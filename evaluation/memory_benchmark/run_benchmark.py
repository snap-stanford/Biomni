"""Differentiated Memory benchmark runner (Baseline vs Improved).

Run this script once against each version's source tree (it imports whatever
``memory`` / ``database`` package is on ``sys.path``), and it emits a flat JSON
dict of metrics to stdout. A separate compare step turns two JSON outputs into
the |metric|baseline|improved|delta| table.

The runner is version-agnostic: it adapts to the API differences between the
baseline commit (4349ab3) and the improved working tree via ``hasattr`` /
``inspect.signature`` probes. Capabilities that only exist in the improved
version (feedback retract, TTL expiry) are measured on improved and reported as
the worst-case (fully exposed) on baseline, because baseline simply cannot
invalidate a fact.

It does NOT touch production code and uses a throwaway SQLite + Chroma store in
a temp directory for every run.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta

from database.migrations import get_engine, get_session_factory, migrate
from database.models import Fact
from memory.episodic import EpisodicMemoryStore
from memory.models import MemoryFact
from memory.retriever import MemoryRetriever
from memory.semantic import SemanticMemoryStore
from memory.validator import FactValidator
from memory.vector import ChromaVectorStore, HashingEmbedding


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _has_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def F(entity, relation, value, confidence=0.9, source="tool_result") -> MemoryFact:
    return MemoryFact(
        entity=entity,
        relation=relation,
        value=value,
        confidence=confidence,
        source=source,
    )


def _val(rows, entity, relation):
    """Return the set of values among rows for a given (entity, relation)."""
    return {r["value"] for r in rows if r["entity"] == entity and r["relation"] == relation}


def _has(rows, entity, relation, value):
    return any(r["entity"] == entity and r["relation"] == relation and r["value"] == value for r in rows)


class Env:
    """A version-agnostic handle over the semantic + episodic + retriever stack."""

    def __init__(self):
        tmp = tempfile.mkdtemp(prefix="membench_")
        db_path = os.path.join(tmp, "memory.db")
        chroma_dir = os.path.join(tmp, "chroma")
        collection = f"col_{uuid.uuid4().hex[:12]}"  # 3..512 chars, unique per run

        self.engine = get_engine(f"sqlite:///{db_path}")
        migrate(self.engine)
        self.sf = get_session_factory(self.engine)

        # Accept everything non-empty so the benchmark isolates storage/retrieval
        # mechanics from validation (both versions share this validator API).
        self.validator = FactValidator(min_confidence=0.0, require_source=False, rejected_sources=set())

        if _has_param(SemanticMemoryStore.__init__, "feedback_retract_threshold"):
            self.semantic = SemanticMemoryStore(self.sf, self.validator, feedback_retract_threshold=3)
        else:
            self.semantic = SemanticMemoryStore(self.sf, self.validator)

        vs = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection)
        emb = HashingEmbedding()
        self.episodic = EpisodicMemoryStore(vs, emb)
        self.retriever = MemoryRetriever(self.episodic, self.semantic, top_k=10)

    # ---- version-agnostic primitives ------------------------------------ #
    def create_memory(self, user_id, summary):
        return self.semantic.create_memory(user_id, summary)

    def save_fact(self, memory_id, entity, relation, value, confidence=0.9, source="tool_result"):
        return self.semantic.save_fact(memory_id, F(entity, relation, value, confidence, source))

    def store_summary(self, memory_id, summary, user_id):
        self.episodic.store_summary(str(memory_id), summary, metadata={"user_id": user_id})

    def search_memories(self, query, user_id, k=10):
        if _has_param(self.episodic.search_memory, "user_id"):
            return self.episodic.search_memory(query, user_id, k=k)
        return self.episodic.search_memory(query, k=k)

    def retrieve(self, query, user_id):
        if _has_param(self.retriever.retrieve, "user_id"):
            return self.retriever.retrieve(query, user_id)
        return self.retriever.retrieve(query)

    def facts_for_query(self, query, user_id):
        """Facts surfaced for a query as ``(entity, relation, value)`` tuples.

        Improved: the real ``retrieve`` path (status filter + ranking + cap).
        Baseline: ``retrieve`` crashes at runtime (passes a string memory_id to a
        ``Uuid`` column, then ``dataclasses.asdict`` on ORM rows). Reproduce its
        *intended* semantics — top-k summaries -> all their facts, no filtering —
        via the robust primitives below, so the comparison stays numeric.
        """
        if _has_param(self.retriever.retrieve, "user_id"):
            ctx = self.retriever.retrieve(query, user_id)
            return [(f.entity, f.relation, f.value) for f in ctx.facts]
        hits = self.search_memories(query, user_id, k=self.retriever.top_k)
        ids = [h.metadata.get("memory_id") for h in hits if h.metadata.get("memory_id")]
        rows = self.exposed_facts(ids, user_id)
        return [(r["entity"], r["relation"], r["value"]) for r in rows]

    def exposed_facts(self, memory_ids, user_id):
        """Mirror the retriever's fact-fetch path for the given version."""
        if hasattr(self.semantic, "get_active_facts_by_memories"):
            return self.semantic.get_active_facts_by_memories(memory_ids, user_id)
        # Baseline: get_facts_by_memory calls dataclasses.asdict on SQLAlchemy ORM
        # rows, which raises TypeError. Read the stored facts directly and map the
        # mapped columns explicitly — exactly what a correct get_facts_by_memory
        # should return, without touching production code.
        from sqlalchemy import select

        out = []
        with self.sf() as session:
            for mid in memory_ids:
                mid_uuid = uuid.UUID(str(mid))
                rows = session.scalars(select(Fact).where(Fact.memory_id == mid_uuid)).all()
                for r in rows:
                    out.append({c.name: getattr(r, c.name) for c in Fact.__table__.columns})
        return out


# --------------------------------------------------------------------------- #
# benchmark sections
# --------------------------------------------------------------------------- #
def bench_conflict(env: Env) -> dict:
    """Supersede + dedup via the shared save_fact API."""
    mid = env.create_memory("alice", "TP53 status analysis")
    env.save_fact(mid, "TP53", "current_status", "unknown", 0.80)
    env.save_fact(mid, "TP53", "current_status", "pathogenic", 0.95)
    exposed = env.exposed_facts([mid], "alice")
    vals = _val(exposed, "TP53", "current_status")
    total = len(vals)
    stale = len(vals - {"pathogenic"})
    correct = len(vals & {"pathogenic"})
    stale_exposure = stale / total if total else 0.0
    active_accuracy = correct / total if total else 1.0

    # dedup: identical fact saved twice should collapse to one in improved
    mid2 = env.create_memory("alice", "duplicate check")
    env.save_fact(mid2, "EGFR", "log2FC", "2.3", 0.90)
    env.save_fact(mid2, "EGFR", "log2FC", "2.3", 0.90)
    exp2 = env.exposed_facts([mid2], "alice")
    dup = sum(1 for r in exp2 if r["entity"] == "EGFR" and r["relation"] == "log2FC" and r["value"] == "2.3")
    dup_exposure = (dup - 1) / dup if dup else 0.0  # 0 = fully deduped

    return {
        "stale_fact_exposure_rate": round(stale_exposure, 4),
        "active_fact_accuracy": round(active_accuracy, 4),
        "duplicate_fact_exposure_rate": round(dup_exposure, 4),
    }


def bench_lifecycle(env: Env) -> dict:
    """Retract (feedback) and expire (TTL). Improved-only; baseline can't invalidate."""
    # ---- retract via negative feedback (improved only) ----
    retract_exposure = 1.0
    if hasattr(env.semantic, "update_fact_feedback"):
        mid = env.create_memory("alice", "feedback retract")
        row = env.save_fact(mid, "Drug", "interaction", "safe", 0.9)
        fid = row.id
        for _ in range(3):
            env.semantic.update_fact_feedback(fid, -1, "alice")
        exposed = env.exposed_facts([mid], "alice")
        retract_exposure = 1.0 if _has(exposed, "Drug", "interaction", "safe") else 0.0

    # ---- expire via TTL (improved only) ----
    expire_exposure = 1.0
    if hasattr(env.semantic, "expire_facts"):
        mid = env.create_memory("alice", "ttl expire")
        env.save_fact(mid, "Sample", "qc_status", "pass", 0.9)
        # age the fact's created_at back 400 days, then expire with ttl=365
        with env.sf() as s:
            from sqlalchemy import update as _upd

            s.execute(_upd(Fact).values(created_at=datetime.now(UTC) - timedelta(days=400)))
            s.commit()
        env.semantic.expire_facts(now=datetime.now(UTC), ttl_days=365)
        exposed = env.exposed_facts([mid], "alice")
        expire_exposure = 1.0 if _has(exposed, "Sample", "qc_status", "pass") else 0.0

    return {
        "retract_exposure_rate": round(retract_exposure, 4),
        "expire_exposure_rate": round(expire_exposure, 4),
    }


def bench_isolation(env: Env) -> dict:
    """Cross-user leakage at the vector layer and the SQL layer."""
    mid_a = env.create_memory("alice", "BRCA1 mutation analysis result")
    env.store_summary(mid_a, "BRCA1 mutation analysis result", "alice")
    env.save_fact(mid_a, "BRCA1", "pathogenic_mutation", "185delAG", 0.9)

    mid_b = env.create_memory("bob", "BRCA1 mutation analysis result")
    env.store_summary(mid_b, "BRCA1 mutation analysis result", "bob")
    env.save_fact(mid_b, "BRCA1", "pathogenic_mutation", "999delX", 0.9)

    # vector leakage: alice queries, count hits that aren't alice's
    hits = env.search_memories("BRCA1 mutation analysis", "alice", k=10)
    leak = sum(1 for h in hits if h.metadata.get("user_id") != "alice")
    vector_leak = leak / len(hits) if hits else 0.0

    # sql leakage: hand bob's memory_id to alice's fact-fetch path
    exposed = env.exposed_facts([mid_b], "alice")
    sql_leak = 1.0 if exposed else 0.0

    return {
        "vector_leakage_rate": round(vector_leak, 4),
        "sql_leakage_rate": round(sql_leak, 4),
    }


def bench_retrieval(env: Env) -> dict:
    """Memory-level Recall@K / MRR and fact-level Precision@K via the retriever."""
    # alice: one clean memory (recall target) + one stale/dup memory (precision target)
    m1 = env.create_memory("alice", "BRCA1 mutation analysis result")
    env.store_summary(m1, "BRCA1 mutation analysis result", "alice")
    env.save_fact(m1, "BRCA1", "pathogenic_mutation", "185delAG", 0.9)
    env.save_fact(m1, "BRCA1", "pathogenic_mutation", "5382insC", 0.9)
    env.save_fact(m1, "BRCA1", "current_status", "unknown", 0.7)
    env.save_fact(m1, "BRCA1", "current_status", "pathogenic", 0.95)

    m2 = env.create_memory("alice", "EGFR differential expression result")
    env.store_summary(m2, "EGFR differential expression result", "alice")
    env.save_fact(m2, "EGFR", "log2FC", "2.3", 0.9)

    # bob: same-topic memory to test isolation through the retriever
    m3 = env.create_memory("bob", "BRCA1 mutation analysis result")
    env.store_summary(m3, "BRCA1 mutation analysis result", "bob")
    env.save_fact(m3, "BRCA1", "pathogenic_mutation", "999delX", 0.9)

    query = "BRCA1 mutation analysis"
    relevant = {str(m1)}

    # memory-level recall + MRR from the vector search
    hits = env.search_memories(query, "alice", k=5)
    hit_ids = [str(h.id) for h in hits]
    recalled = relevant & set(hit_ids)
    recall_at_k = len(recalled) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for rank, h in enumerate(hits, start=1):
        if str(h.id) in relevant:
            mrr = 1.0 / rank
            break

    # fact-level precision through the full retriever (filtering + isolation)
    facts = env.facts_for_query(query, "alice")
    correct = {
        ("BRCA1", "pathogenic_mutation", "185delAG"),
        ("BRCA1", "pathogenic_mutation", "5382insC"),
        ("BRCA1", "current_status", "pathogenic"),
    }
    n_correct = sum(1 for f in facts if f in correct)
    precision = n_correct / len(facts) if facts else 0.0

    return {
        "recall_at_5": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "fact_precision": round(precision, 4),
    }


def bench_continuation(env: Env) -> dict:
    """Session A persists specific values; session B must surface them."""
    mid = env.create_memory("alice", "BRCA1 pathogenic mutation analysis result")
    env.store_summary(mid, "BRCA1 pathogenic mutation analysis result", "alice")
    env.save_fact(mid, "BRCA1", "pathogenic_mutation", "185delAG", 0.9)
    env.save_fact(mid, "BRCA1", "pathogenic_mutation", "5382insC", 0.9)
    # a stale single-value fact that should be filtered out on improved
    env.save_fact(mid, "BRCA1", "current_status", "unknown", 0.7)
    env.save_fact(mid, "BRCA1", "current_status", "pathogenic", 0.95)

    facts = env.facts_for_query("BRCA1 pathogenic mutation continuation", "alice")

    required = [
        ("BRCA1", "pathogenic_mutation", "185delAG"),
        ("BRCA1", "pathogenic_mutation", "5382insC"),
    ]
    hit = sum(1 for r in required if r in facts) / len(required)

    specific_values = ["185delAG", "5382insC"]
    retained = sum(1 for v in specific_values if any(f[2] == v for f in facts)) / len(specific_values)

    return {
        "required_fact_hit_rate": round(hit, 4),
        "specific_value_retention_rate": round(retained, 4),
    }


def detect_version() -> str:
    # The improved Fact model gained a lifecycle `status` column; baseline lacks it.
    return "improved" if hasattr(Fact, "status") else "baseline"


def main() -> None:
    version = detect_version()
    # Each section gets a fresh Env (its own SQLite + Chroma collection) so one
    # benchmark's stored summaries cannot leak into another's retrieval ranking.
    metrics = {
        "version": version,
        "conflict": bench_conflict(Env()),
        "lifecycle": bench_lifecycle(Env()),
        "isolation": bench_isolation(Env()),
        "retrieval": bench_retrieval(Env()),
        "continuation": bench_continuation(Env()),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
