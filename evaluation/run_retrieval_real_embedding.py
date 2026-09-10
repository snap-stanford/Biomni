"""Retrieval benchmark with a configurable (incl. real semantic) embedding.

Run once per version (baseline vs improved), pointed at its source tree via
``PYTHONPATH``. Emits a flat JSON of retrieval metrics to stdout. A second
``--compare`` invocation turns two JSON outputs into ``comparison.md``.

Usage::

    # improved (current tree)
    PYTHONPATH=/home/ytz/Biomni1      venv/bin/python3 run_retrieval_real_embedding.py \
        --embedding huggingface > results/retrieval_real_embedding/improved.json

    # baseline
    PYTHONPATH=/tmp/membench/baseline venv/bin/python3 run_retrieval_real_embedding.py \
        --embedding huggingface > results/retrieval_real_embedding/baseline.json

    # comparison
    venv/bin/python3 run_retrieval_real_embedding.py --compare \
        results/retrieval_real_embedding/baseline.json \
        results/retrieval_real_embedding/improved.json

``--embedding`` accepts ``hash``, ``huggingface`` (real semantic: HuggingFace
``all-MiniLM-L6-v2``), or ``openai``. This does NOT modify production ``vector.py``;
it builds a provider object here and hands it to ``EpisodicMemoryStore``.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

# Allow ``import metrics`` regardless of the caller's cwd.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import metrics as M  # noqa: E402

from database.migrations import get_engine, get_session_factory, migrate  # noqa: E402
from database.models import Fact  # noqa: E402
from memory.episodic import EpisodicMemoryStore  # noqa: E402
from memory.models import MemoryFact  # noqa: E402
from memory.retriever import MemoryRetriever  # noqa: E402
from memory.semantic import SemanticMemoryStore  # noqa: E402
from memory.validator import FactValidator  # noqa: E402
from memory.vector import ChromaVectorStore, HashingEmbedding  # noqa: E402

CORPUS_PATH = os.path.join(_EVAL_DIR, "datasets", "memory_corpus.json")
TOP_K = 5  # enough to compute Precision@1/3/5 and Recall@5


# --------------------------------------------------------------------------- #
# embedding providers (built here, never touching production vector.py)
# --------------------------------------------------------------------------- #
def _has_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


class _GtMiniLMEmbedding:
    """Real semantic embedding: HuggingFace ``all-MiniLM-L6-v2``.

    Primary path: the PyPI package ``gt-all-minilm-l6-v2`` bundles the model
    weights inside the wheel (no network needed at inference time), exposing a
    sentence-transformers-style model via ``load_model()``. It requires torch.

    Fallback path: chromadb's ONNX backend (``ONNXMiniLM_L6_V2``) — no torch, but
    it downloads the model archive from chromadb's S3 mirror on first use.
    """

    def __init__(self, model: str | None = None):
        self._model = model or "all-MiniLM-L6-v2"
        self._backend = None

    def _ensure(self):
        if self._backend is not None:
            return self._backend
        # Preferred: bundled torch model (offline, exact sentence-transformers weights).
        try:
            import gt_all_minilm_l6_v2 as gt

            st = gt.load_model()
            import numpy as np

            class _Backend:
                def encode(self, texts):
                    vecs = st.encode(list(texts), normalize_embeddings=True)
                    return np.asarray(vecs, dtype="float32")

            self._backend = _Backend()
            return self._backend
        except Exception as exc:  # pragma: no cover - depends on env
            import logging

            logging.getLogger(__name__).warning(
                "gt-all-minilm-l6-v2 unavailable (%s); falling back to "
                "chromadb ONNX backend", exc,
            )
        # Fallback: chromadb ONNX (downloads model on first use).
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        onnx_fn = ONNXMiniLM_L6_V2()

        class _Backend:
            def encode(self, texts):
                return onnx_fn(list(texts))

        self._backend = _Backend()
        return self._backend

    def embed_documents(self, texts):
        backend = self._ensure()
        vecs = backend.encode(list(texts))
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class _OpenAIEmbedding:
    """Optional provider: requires ``langchain_openai`` + ``OPENAI_API_KEY``."""

    def __init__(self, model: str):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "openai provider requires `pip install langchain-openai` "
                "and an OPENAI_API_KEY."
            ) from exc
        self._emb = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts):
        return self._emb.embed_documents(list(texts))

    def embed_query(self, text):
        return self._emb.embed_query(text)


def build_embedding(provider: str, model: str | None):
    if provider == "hash":
        return HashingEmbedding()
    if provider == "huggingface":
        return _GtMiniLMEmbedding(model)
    if provider == "openai":
        return _OpenAIEmbedding(model or "text-embedding-3-small")
    raise SystemExit(f"unknown embedding provider: {provider}")


# --------------------------------------------------------------------------- #
# version-agnostic stack handle
# --------------------------------------------------------------------------- #
class Env:
    def __init__(self, embedding):
        tmp = tempfile.mkdtemp(prefix="membench_retr_")
        self.engine = get_engine(f"sqlite:///{tmp}/memory.db")
        migrate(self.engine)
        self.sf = get_session_factory(self.engine)
        self.validator = FactValidator(
            min_confidence=0.0, require_source=False, rejected_sources=set()
        )
        if _has_param(SemanticMemoryStore.__init__, "feedback_retract_threshold"):
            self.semantic = SemanticMemoryStore(
                self.sf, self.validator, feedback_retract_threshold=3
            )
        else:
            self.semantic = SemanticMemoryStore(self.sf, self.validator)
        vs = ChromaVectorStore(
            persist_dir=os.path.join(tmp, "chroma"),
            collection_name=f"col_{uuid.uuid4().hex[:12]}",
        )
        self.episodic = EpisodicMemoryStore(vs, embedding)
        self.retriever = MemoryRetriever(self.episodic, self.semantic, top_k=TOP_K)

    def create_memory(self, user_id, summary):
        return self.semantic.create_memory(user_id, summary)

    def save_fact(self, memory_id, entity, relation, value, confidence=0.9):
        return self.semantic.save_fact(
            memory_id,
            MemoryFact(entity=entity, relation=relation, value=value,
                       confidence=confidence, source="tool_result"),
        )

    def store_summary(self, memory_id, summary, user_id):
        self.episodic.store_summary(str(memory_id), summary, metadata={"user_id": user_id})

    def search_memories(self, query, user_id, k=TOP_K):
        if _has_param(self.episodic.search_memory, "user_id"):
            return self.episodic.search_memory(query, user_id, k=k)
        return self.episodic.search_memory(query, k=k)

    def facts_for_query(self, query, user_id):
        """Facts surfaced for a query as ``(entity, relation, value)`` tuples."""
        if _has_param(self.retriever.retrieve, "user_id"):
            ctx = self.retriever.retrieve(query, user_id)
            return [(f.entity, f.relation, f.value) for f in ctx.facts]
        # Baseline retrieve() crashes (string memory_id + asdict on ORM rows);
        # reproduce its intended semantics via robust primitives.
        hits = self.search_memories(query, user_id, k=self.retriever.top_k)
        ids = [h.metadata.get("memory_id") for h in hits if h.metadata.get("memory_id")]
        rows = self.exposed_facts(ids, user_id)
        return [(r["entity"], r["relation"], r["value"]) for r in rows]

    def exposed_facts(self, memory_ids, user_id):
        if hasattr(self.semantic, "get_active_facts_by_memories"):
            return self.semantic.get_active_facts_by_memories(memory_ids, user_id)
        from sqlalchemy import select

        out = []
        with self.sf() as session:
            for mid in memory_ids:
                rows = session.scalars(
                    select(Fact).where(Fact.memory_id == uuid.UUID(str(mid)))
                ).all()
                for r in rows:
                    out.append({c.name: getattr(r, c.name) for c in Fact.__table__.columns})
        return out

    def retract_all(self, fact_rows, user_id):
        """Retract facts via 3 negative feedback votes (improved-only)."""
        if not hasattr(self.semantic, "update_fact_feedback"):
            return
        for row in fact_rows:
            for _ in range(3):
                self.semantic.update_fact_feedback(row.id, -1, user_id)

    def expire_memory(self, memory_id):
        """Age a memory's facts past TTL and expire them (improved-only)."""
        if not hasattr(self.semantic, "expire_facts"):
            return
        from sqlalchemy import update as _upd

        with self.sf() as session:
            session.execute(
                _upd(Fact)
                .where(Fact.memory_id == uuid.UUID(str(memory_id)))
                .values(created_at=datetime.now(timezone.utc) - timedelta(days=400))
            )
            session.commit()
        self.semantic.expire_facts(now=datetime.now(timezone.utc), ttl_days=365)


def detect_version() -> str:
    return "improved" if hasattr(Fact, "status") else "baseline"


# --------------------------------------------------------------------------- #
# ingest + evaluate
# --------------------------------------------------------------------------- #
def ingest_corpus(env: Env, corpus: dict) -> dict[str, str]:
    """Store every memory + facts; return ``{memory_id_str: memory_key}``."""
    id2key: dict[str, str] = {}
    for mem in corpus["memories"]:
        mid = env.create_memory(mem["user_id"], mem["summary"])
        env.store_summary(str(mid), mem["summary"], mem["user_id"])
        fact_rows = []
        for f in mem.get("facts", []):
            row = env.save_fact(
                mid, f["entity"], f["relation"], f["value"], f.get("confidence", 0.9)
            )
            if row is not None:
                fact_rows.append(row)
        id2key[str(mid)] = mem["memory_key"]

        lc = mem.get("lifecycle")
        if lc == "retracted":
            env.retract_all(fact_rows, mem["user_id"])
        elif lc == "expired":
            env.expire_memory(mid)
        # "superseded" needs no action: its single-value facts supersede naturally.
    return id2key


def evaluate(env: Env, corpus: dict, id2key: dict[str, str]) -> dict:
    p1, p3, p5, r5, rr = [], [], [], [], []
    low_lex_overlap_hits: list[float] = []
    fact_precisions: list[float] = []
    fact_query_count = 0

    for q in corpus["queries"]:
        hits = env.search_memories(q["query"], q["user_id"], k=TOP_K)
        ranked = [id2key[str(h.id)] for h in hits if str(h.id) in id2key]
        expected = q["expected_memory_keys"]

        p1.append(M.precision_at_k(ranked, expected, 1))
        p3.append(M.precision_at_k(ranked, expected, 3))
        p5.append(M.precision_at_k(ranked, expected, 5))
        r5.append(M.recall_at_k(ranked, expected, 5))
        rr.append(M.mrr(ranked, expected))

        if q.get("category") == "low_lexical_overlap":
            low_lex_overlap_hits.append(M.hit_rate(ranked, expected))

        if "expected_facts" in q:
            fact_query_count += 1
            returned = env.facts_for_query(q["query"], q["user_id"])
            expected_facts = {tuple(t) for t in q["expected_facts"]}
            fact_precisions.append(M.fact_precision(returned, expected_facts))

    return {
        "version": detect_version(),
        "corpus": {
            "n_memories": len(corpus["memories"]),
            "n_queries": len(corpus["queries"]),
        },
        "memory_level": {
            "precision_at_1": round(M.mean(p1), 4),
            "precision_at_3": round(M.mean(p3), 4),
            "precision_at_5": round(M.mean(p5), 4),
            "recall_at_5": round(M.mean(r5), 4),
            "mrr": round(M.mean(rr), 4),
        },
        "fact_level": {
            "fact_precision": round(M.mean(fact_precisions), 4),
            "n_fact_queries": fact_query_count,
        },
        "low_lexical_overlap_success_rate": round(M.mean(low_lex_overlap_hits), 4),
        "n_low_lexical_overlap_queries": len(low_lex_overlap_hits),
    }


# --------------------------------------------------------------------------- #
# comparison table
# --------------------------------------------------------------------------- #
def _flatten(d: dict) -> dict[str, float]:
    out = {}
    out["memory_level.precision_at_1"] = d["memory_level"]["precision_at_1"]
    out["memory_level.precision_at_3"] = d["memory_level"]["precision_at_3"]
    out["memory_level.precision_at_5"] = d["memory_level"]["precision_at_5"]
    out["memory_level.recall_at_5"] = d["memory_level"]["recall_at_5"]
    out["memory_level.mrr"] = d["memory_level"]["mrr"]
    out["fact_level.fact_precision"] = d["fact_level"]["fact_precision"]
    out["low_lexical_overlap_success_rate"] = d["low_lexical_overlap_success_rate"]
    return out


def _fmt(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def compare(baseline_path, improved_path) -> str:
    base = _flatten(json.loads(open(baseline_path).read()))
    impr = _flatten(json.loads(open(improved_path).read()))
    lines = ["| metric | baseline | improved | delta |", "|--------|---------:|---------:|------:|"]
    for k in base:
        b, i = base[k], impr[k]
        d = i - b
        lines.append(f"| {k} | {_fmt(b)} | {_fmt(i)} | {_fmt(d)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="huggingface",
                    choices=["hash", "huggingface", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "IMPROVED"), default=None)
    args = ap.parse_args()

    if args.compare:
        print(compare(args.compare[0], args.compare[1]))
        return

    corpus = json.loads(open(CORPUS_PATH).read())
    embedding = build_embedding(args.embedding, args.model)
    env = Env(embedding)
    id2key = ingest_corpus(env, corpus)
    result = evaluate(env, corpus, id2key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
