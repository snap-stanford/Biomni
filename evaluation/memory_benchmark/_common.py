"""Shared helpers for the memory benchmarks (retrieval + continuation).

This module is benchmark-only. It does NOT touch production ``memory``/``database``
code beyond importing it (the version under ``sys.path`` — baseline or improved).

Two responsibilities:

1. **Corpus content** — derive, for every ``memory_key`` in
   ``memory_corpus.json``, a deterministic ``summary`` and ``facts`` used to ingest
   a memory. The summary is built from the human-authored extraction trace (task +
   observations + solution); the facts come from the human-confirmed extraction
   ground truth plus the continuation cases' ``facts_must_contain``. This stands in
   for the LLM extractor (out of scope for retrieval/continuation) so the benchmarks
   can be run deterministically with no LLM and no network beyond the embedding model.

2. **Version-agnostic stack** — an ``Env`` handle over ``SemanticMemoryStore`` +
   ``EpisodicMemoryStore`` + ``MemoryRetriever`` that adapts to the API differences
   between the baseline commit and the improved tree via ``inspect.signature`` /
   ``hasattr`` probes (the improved version added ``user_id`` scoping, lifecycle
   ``status``, and query-aware reranking; the baseline ``retrieve``/``get_facts_by_memory``
   crash at runtime, so the baseline's *intended* semantics are reproduced via the
   robust primitives).
"""

from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))

CORPUS = os.path.join(_HERE, "memory_corpus.json")
EXTRACTION_CASES = os.path.join(_HERE, "extraction", "extraction_cases.json")
EXTRACTION_GT = os.path.join(_HERE, "ground_truth", "extraction_ground_truth.json")
CONTINUATION_CASES = os.path.join(_HERE, "continuation", "continuation_cases.json")
RETRIEVAL_CASES = os.path.join(_HERE, "retrieval", "retrieval_cases.json")

_TAG = re.compile(r"</?[a-zA-Z_][^>]*>")


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def strip_tags(s: str) -> str:
    """Remove the trace's inline ``<execute>`` / ``<solution>`` markup."""
    return _TAG.sub("", s).strip()


def build_summary(trace: list[dict]) -> str:
    """A deterministic, human-authored summary from an extraction trace.

    Concatenates the human task, the observations, and the final solution line,
    joined with "；". These are the messages that carry the task's facts, so the
    summary contains the specific values the continuation criteria expect.
    """
    parts: list[str] = []
    for m in trace:
        c = (m.get("content") or "").strip()
        if not c:
            continue
        t = m.get("type")
        if t == "human":
            parts.append(c)
        elif t == "observation":
            parts.append(strip_tags(c))
        elif t == "ai" and "<solution>" in c:
            parts.append(strip_tags(c))
    return "；".join(parts)


def _facts_from_extraction_gt() -> dict[str, list[tuple[str, str, str]]]:
    """``case_id -> [(entity, relation, value)]`` from human-confirmed extraction GT."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for case in _load(EXTRACTION_GT).get("cases", []):
        out[case["case_id"]] = [(f["entity"], f["relation"], f["value"]) for f in case.get("expected_facts", [])]
    return out


def _facts_from_continuation() -> dict[str, list[tuple[str, str, str]]]:
    """``memory_key -> [(entity, relation, value)]`` from the continuation cases."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for case in _load(CONTINUATION_CASES).get("cases", []):
        keys = case.get("session_a", {}).get("memory_keys", [])
        facts = [
            (f["entity"], f["relation"], f["value"])
            for f in case.get("expected_memory", {}).get("facts_must_contain", [])
        ]
        if not facts:
            continue
        if len(keys) == 1:
            out.setdefault(keys[0], []).extend(facts)
        else:
            # multi-memory continuation: assign each fact to the key whose name
            # mentions the fact's entity (e.g. BRCA1 -> brca1_*). Fallback: first key.
            for f in facts:
                target = next((k for k in keys if f[0].lower() in k.lower()), keys[0])
                out.setdefault(target, []).append(f)
    return out


def build_corpus_content() -> dict[str, dict]:
    """``memory_key -> {"user_id", "summary", "facts": [(e,r,v), ...]}``."""
    corpus = _load(CORPUS)
    traces = {c["case_id"]: c["trace"] for c in _load(EXTRACTION_CASES)["cases"]}
    gt_facts = _facts_from_extraction_gt()
    cont_facts = _facts_from_continuation()

    def _dedup(items):
        seen = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    content: dict[str, dict] = {}
    for mem in corpus["memories"]:
        key = mem["memory_key"]
        content[key] = {
            "user_id": mem["user_id"],
            "summary": build_summary(traces.get(mem["trace_ref"], [])),
            "facts": _dedup(list(cont_facts.get(key, [])) + gt_facts.get(mem["trace_ref"], [])),
        }
    return content


# --------------------------------------------------------------------------- #
# embedding (built here — never touching production vector.py's default)
# --------------------------------------------------------------------------- #
class _OnnxSemanticEmbedding:
    """The production ``sentence_transformer`` model ``all-MiniLM-L6-v2`` (384-d),
    loaded via chromadb's bundled ONNX backend (torch is not installed in this env).
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self._fn(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def build_embedding(provider: str = "sentence_transformer"):
    from memory.vector import HashingEmbedding

    if provider == "hash":
        return HashingEmbedding()
    if provider == "sentence_transformer":
        return _OnnxSemanticEmbedding()
    raise SystemExit(f"unknown embedding provider: {provider}")


# --------------------------------------------------------------------------- #
# version-agnostic stack handle
# --------------------------------------------------------------------------- #
def _has_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def detect_version() -> str:
    from database.models import Fact

    return "improved" if hasattr(Fact, "status") else "baseline"


class Env:
    """A handle over the semantic + episodic + retriever stack.

    Works against both the baseline and the improved tree. For the improved tree it
    passes the embedding into the retriever (so query-aware fact reranking — the
    production behaviour — is exercised); the baseline retriever has no such param.
    """

    def __init__(self, embedding, top_k: int = 5) -> None:
        from database.migrations import get_engine, get_session_factory, migrate
        from memory.episodic import EpisodicMemoryStore
        from memory.models import MemoryFact
        from memory.retriever import MemoryRetriever
        from memory.semantic import SemanticMemoryStore
        from memory.validator import FactValidator
        from memory.vector import ChromaVectorStore

        tmp = tempfile.mkdtemp(prefix="membench_")
        self.engine = get_engine(f"sqlite:///{tmp}/memory.db")
        migrate(self.engine)
        self.sf = get_session_factory(self.engine)
        self.validator = FactValidator(min_confidence=0.0, require_source=False, rejected_sources=set())
        if _has_param(SemanticMemoryStore.__init__, "feedback_retract_threshold"):
            self.semantic = SemanticMemoryStore(self.sf, self.validator, feedback_retract_threshold=3)
        else:
            self.semantic = SemanticMemoryStore(self.sf, self.validator)
        vs = ChromaVectorStore(
            persist_dir=os.path.join(tmp, "chroma"),
            collection_name=f"col_{uuid.uuid4().hex[:12]}",
        )
        self.episodic = EpisodicMemoryStore(vs, embedding)
        if _has_param(MemoryRetriever.__init__, "embedding"):
            self.retriever = MemoryRetriever(self.episodic, self.semantic, top_k=top_k, embedding=embedding)
        else:
            self.retriever = MemoryRetriever(self.episodic, self.semantic, top_k=top_k)
        self._MemoryFact = MemoryFact

    def create_memory(self, user_id, summary):
        return self.semantic.create_memory(user_id, summary)

    def save_fact(self, memory_id, entity, relation, value, confidence=0.9):
        return self.semantic.save_fact(
            memory_id,
            self._MemoryFact(
                entity=entity,
                relation=relation,
                value=value,
                confidence=confidence,
                source="tool_result",
            ),
        )

    def store_summary(self, memory_id, summary, user_id):
        self.episodic.store_summary(str(memory_id), summary, metadata={"user_id": user_id})

    def search_memories(self, query, user_id, k=5):
        if _has_param(self.episodic.search_memory, "user_id"):
            return self.episodic.search_memory(query, user_id, k=k)
        return self.episodic.search_memory(query, k=k)

    def retrieve(self, query, user_id):
        if _has_param(self.retriever.retrieve, "user_id"):
            return self.retriever.retrieve(query, user_id)
        # Baseline retrieve() crashes (get_facts_by_memory asdict on ORM rows);
        # reproduce its intended semantics via robust primitives.
        from memory.models import MemoryContext

        ctx = MemoryContext()
        hits = self.search_memories(query, user_id, k=self.retriever.top_k)
        for h in hits:
            s = h.text or h.metadata.get("summary", "")
            if s and s not in ctx.previous_tasks:
                ctx.previous_tasks.append(s)
        for r in self.exposed_facts(
            [h.metadata.get("memory_id") for h in hits if h.metadata.get("memory_id")],
            user_id,
        ):
            ctx.facts.append(
                self._MemoryFact(
                    entity=r["entity"],
                    relation=r["relation"],
                    value=r["value"],
                    confidence=float(r.get("confidence", 0.0)),
                    source=r.get("source") or "",
                    created_at=r.get("created_at"),
                    updated_at=r.get("updated_at"),
                    status=r.get("status") or "active",
                )
            )
        return ctx

    def facts_for_query(self, query, user_id):
        ctx = self.retrieve(query, user_id)
        return [(f.entity, f.relation, f.value) for f in ctx.facts]

    def exposed_facts(self, memory_ids, user_id):
        if hasattr(self.semantic, "get_active_facts_by_memories"):
            return self.semantic.get_active_facts_by_memories(memory_ids, user_id)
        # Baseline: read the stored facts directly (bypass the crashing asdict).
        from database.models import Fact
        from sqlalchemy import select

        out = []
        with self.sf() as session:
            for mid in memory_ids:
                rows = session.scalars(select(Fact).where(Fact.memory_id == uuid.UUID(str(mid)))).all()
                for r in rows:
                    out.append({c.name: getattr(r, c.name) for c in Fact.__table__.columns})
        return out
