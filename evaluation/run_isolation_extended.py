"""Extended user-isolation benchmark: 10 users x 20 memories, 100 random queries.

Every user stores the *same* 20 summary templates (identical text), so the only
thing that separates one user's episode from another's is the ``user_id`` in
metadata. A query by ``user_i`` must return only ``user_i``'s memories.

The improved version scopes vector search with ``where={"user_id": user_id}``
(and re-verifies ownership in SQL); the baseline has neither boundary, so it
leaks other users' identically-titled memories. This benchmark is independent of
the embedding model (hash is fine): the leak is a filtering bug, not a semantic
one.

Run once per version::

    PYTHONPATH=<repo-root>         venv/bin/python3 run_isolation_extended.py
    PYTHONPATH=<baseline-checkout> venv/bin/python3 run_isolation_extended.py
    venv/bin/python3 run_isolation_extended.py --compare baseline.json improved.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import tempfile
import uuid

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from database.migrations import get_engine, get_session_factory, migrate
from database.models import Fact
from memory.episodic import EpisodicMemoryStore
from memory.models import MemoryFact
from memory.semantic import SemanticMemoryStore
from memory.validator import FactValidator
from memory.vector import ChromaVectorStore, HashingEmbedding

N_USERS = 10
MEMORIES_PER_USER = 20
N_QUERIES = 100
SEED = 42
TOP_K = 20  # large enough to surface all identically-titled memories on baseline

# 20 shared summary templates: identical text across every user, so only the
# user_id metadata distinguishes ownership.
TEMPLATES = [
    "Quality control report for sequencing run",
    "Differential expression analysis results",
    "Sample processing log entry",
    "Variant calling summary",
    "Clustering analysis output",
    "Data normalization report",
    "Gene set enrichment analysis results",
    "Statistical test summary",
    "Batch correction report",
    "Read alignment statistics",
    "Peak calling output",
    "Phylogenetic tree construction result",
    "Protein structure prediction output",
    "Pathway analysis conclusion",
    "Drug screening results",
    "Survival analysis summary",
    "Copy number variation report",
    "Expression heatmap generation",
    "Metadata quality assessment",
    "Final analysis report",
]


def _has_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def detect_version() -> str:
    return "improved" if hasattr(Fact, "status") else "baseline"


def build_stores():
    tmp = tempfile.mkdtemp(prefix="membench_iso_")
    engine = get_engine(f"sqlite:///{tmp}/memory.db")
    migrate(engine)
    sf = get_session_factory(engine)
    validator = FactValidator(min_confidence=0.0, require_source=False, rejected_sources=set())
    if _has_param(SemanticMemoryStore.__init__, "feedback_retract_threshold"):
        semantic = SemanticMemoryStore(sf, validator, feedback_retract_threshold=3)
    else:
        semantic = SemanticMemoryStore(sf, validator)
    vs = ChromaVectorStore(
        persist_dir=os.path.join(tmp, "chroma"),
        collection_name=f"col_{uuid.uuid4().hex[:12]}",
    )
    episodic = EpisodicMemoryStore(vs, HashingEmbedding())
    return semantic, episodic


def ingest(semantic, episodic) -> None:
    """Store, for every user, all 20 templates as separate memories."""
    for u in range(N_USERS):
        user_id = f"user_{u}"
        for j, template in enumerate(TEMPLATES):
            mid = semantic.create_memory(user_id, template)
            episodic.store_summary(str(mid), template, metadata={"user_id": user_id})
            semantic.save_fact(
                mid,
                MemoryFact(
                    entity=user_id,
                    relation="report",
                    value=f"report_{j}",
                    confidence=0.9,
                    source="tool_result",
                ),
            )


def run(semantic, episodic) -> dict:
    rng = random.Random(SEED)
    queries = [(f"user_{rng.randrange(N_USERS)}", rng.randrange(len(TEMPLATES))) for _ in range(N_QUERIES)]

    leakage_count = 0
    affected_users: set[str] = set()
    total_returned = 0

    for user_id, tpl_idx in queries:
        query = TEMPLATES[tpl_idx]
        if _has_param(episodic.search_memory, "user_id"):
            hits = episodic.search_memory(query, user_id, k=TOP_K)
        else:
            hits = episodic.search_memory(query, k=TOP_K)

        returned_users = [h.metadata.get("user_id") for h in hits]
        total_returned += len(hits)
        leaked = [u for u in returned_users if u != user_id]
        if leaked:
            leakage_count += 1
            affected_users.update(leaked)

    return {
        "version": detect_version(),
        "n_users": N_USERS,
        "memories_per_user": MEMORIES_PER_USER,
        "total_memories": N_USERS * MEMORIES_PER_USER,
        "total_queries": N_QUERIES,
        "leakage_count": leakage_count,
        "leakage_rate": round(leakage_count / N_QUERIES, 4),
        "affected_users": len(affected_users),
        "affected_user_ids": sorted(affected_users),
    }


def _fmt(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def compare(baseline_path, improved_path) -> str:
    base = json.loads(open(baseline_path).read())
    impr = json.loads(open(improved_path).read())
    rows = [
        ("total_queries", base["total_queries"], impr["total_queries"]),
        ("leakage_count", base["leakage_count"], impr["leakage_count"]),
        ("leakage_rate", base["leakage_rate"], impr["leakage_rate"]),
        ("affected_users", base["affected_users"], impr["affected_users"]),
    ]
    lines = ["| metric | baseline | improved | delta |", "|--------|---------:|---------:|------:|"]
    for name, b, i in rows:
        d = i - b if isinstance(b, (int, float)) and isinstance(i, (int, float)) else ""
        lines.append(f"| {name} | {_fmt(b)} | {_fmt(i)} | {_fmt(d) if d != '' else '—'} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "IMPROVED"), default=None)
    args = ap.parse_args()

    if args.compare:
        print(compare(args.compare[0], args.compare[1]))
        return

    semantic, episodic = build_stores()
    ingest(semantic, episodic)
    print(json.dumps(run(semantic, episodic), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
