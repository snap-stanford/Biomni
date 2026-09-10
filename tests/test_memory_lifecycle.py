"""Tests for fact lifecycle, scoring, retrieval ranking, and cleanup.

Covers the 15 behaviours requested in the memory-system enhancement:
timestamps, status, TTL expiry, access_count accounting, importance/recency
scoring, two-stage ranking, and cross-store cleanup via ``memory_id``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

import pytest
from database.migrations import get_engine, get_session_factory, migrate
from database.models import Fact, Memory
from memory.episodic import EpisodicMemoryStore
from memory.models import MemoryConfig, MemoryFact
from memory.retriever import MemoryRetriever
from memory.scoring import feedback_score, importance_score, recency_score, usage_score
from memory.semantic import SemanticMemoryStore
from memory.system import MemorySystem
from memory.validator import FactValidator
from memory.vector import HashingEmbedding, SearchResult, build_embedding_provider, build_vector_store
from sqlalchemy import inspect, text
from sqlalchemy import update as sa_update


class InMemoryVectorStore:
    """Minimal in-memory VectorStore so tests don't need chromadb."""

    def __init__(self) -> None:
        self._items: dict = {}

    def add(self, ids, texts, embeddings, metadatas=None) -> None:
        metadatas = metadatas or [{}] * len(ids)
        for i, cid in enumerate(ids):
            self._items[cid] = (texts[i], embeddings[i], metadatas[i])

    def search(self, query_embedding, k: int = 5, where: dict | None = None) -> list[SearchResult]:
        results = []
        for cid, (text, emb, meta) in self._items.items():
            if where and any(meta.get(kk) != vv for kk, vv in where.items()):
                continue
            results.append(
                SearchResult(
                    id=cid,
                    score=sum(a * b for a, b in zip(query_embedding, emb, strict=False)),
                    metadata=meta,
                    text=text,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get(self, ids) -> list[SearchResult]:
        return [
            SearchResult(id=cid, score=0.0, metadata=self._items[cid][2], text=self._items[cid][0])
            for cid in ids
            if cid in self._items
        ]

    def delete(self, ids) -> None:
        for cid in ids:
            self._items.pop(cid, None)


def _fact(
    entity="BRCA1",
    relation="has_mutation",
    value="185delAG",
    confidence=0.9,
    source="tool_result",
) -> MemoryFact:
    return MemoryFact(entity=entity, relation=relation, value=value, confidence=confidence, source=source)


@pytest.fixture
def session_factory(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/memory.db")
    migrate(engine)
    return get_session_factory(engine)


@pytest.fixture
def semantic(session_factory):
    return SemanticMemoryStore(session_factory, FactValidator())


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


@pytest.fixture
def episodic(vector_store):
    return EpisodicMemoryStore(vector_store, HashingEmbedding())


# ---- lifecycle -----------------------------------------------------------


def test_fact_gets_timestamps(semantic):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    assert row.created_at is not None
    assert row.updated_at is not None


def test_created_at_unchanged_on_update(semantic):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    original = row.created_at
    updated = semantic.update_fact(row.id, value="changed_value")
    assert updated is not None
    assert updated.created_at == original


def test_updated_at_changes_on_update(semantic, session_factory):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    # Force updated_at far into the past so the onupdate bump is observable.
    with session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.id == row.id).values(updated_at=datetime(2000, 1, 1)))
        session.commit()
    updated = semantic.update_fact(row.id, value="changed_again")
    assert updated is not None
    assert updated.updated_at > datetime(2000, 1, 1)


def test_status_defaults_to_active(semantic):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    assert row.status == "active"


def test_expires_after_ttl(semantic, session_factory):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    with session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.id == row.id).values(created_at=datetime(2020, 1, 1)))
        session.commit()
    count = semantic.expire_facts(now=datetime.now(UTC), ttl_days=365)
    assert count >= 1
    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["status"] == "expired"


def test_not_expired_within_ttl(semantic):
    memory_id = semantic.create_memory("u1", "summary")
    semantic.save_fact(memory_id, _fact())
    semantic.expire_facts(now=datetime.now(UTC), ttl_days=365)
    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["status"] == "active"


def test_access_count_starts_at_zero(semantic):
    memory_id = semantic.create_memory("u1", "summary")
    row = semantic.save_fact(memory_id, _fact())
    assert row.access_count == 0


# ---- retrieval accounting -----------------------------------------------


def test_access_count_increments_on_retrieval(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=5)
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(confidence=0.9))
    episodic.store_summary(str(memory_id), "summary", metadata={"user_id": "alice"})

    context = retriever.retrieve("query", user_id="alice")
    assert len(context.facts) == 1
    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["access_count"] == 1


def test_unselected_candidate_not_incremented(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=5, max_facts=1)
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="A", value="va", confidence=0.9))
    semantic.save_fact(memory_id, _fact(entity="B", value="vb", confidence=0.7))
    episodic.store_summary(str(memory_id), "summary", metadata={"user_id": "alice"})

    context = retriever.retrieve("query", user_id="alice")
    assert len(context.facts) == 1
    counts = {f["entity"]: f["access_count"] for f in semantic.get_facts_by_memory(memory_id)}
    assert counts["A"] == 1  # higher confidence selected
    assert counts["B"] == 0  # candidate but not selected


# ---- scoring -------------------------------------------------------------


def test_importance_score_bounded():
    now = datetime.now(UTC)
    for conf in (0.0, 0.5, 1.0):
        for ac in (0, 1, 10, 1000):
            score = importance_score(
                conf,
                ac,
                now - timedelta(days=10),
                now,
                confidence_weight=0.5,
                usage_weight=0.3,
                recency_weight=0.2,
                usage_saturation=100.0,
                recency_lambda=0.01,
            )
            assert 0.0 <= score <= 1.0


def test_recency_and_usage_are_bounded():
    now = datetime.now(UTC)
    assert 0.0 <= recency_score(now - timedelta(days=400), now, 0.01) <= 1.0
    assert 0.0 <= usage_score(0, 100.0) <= 1.0
    assert 0.0 <= usage_score(100000, 100.0) <= 1.0


def test_cold_start_ranks_by_confidence():
    retriever = MemoryRetriever(None, None, scoring_threshold=3, max_facts=10)
    now = datetime.now(UTC)
    rows = [
        {"id": "low", "entity": "low", "confidence": 0.5, "access_count": 1, "created_at": now},
        {"id": "high", "entity": "high", "confidence": 0.9, "access_count": 1, "created_at": now},
    ]
    selected = retriever._rank_and_select(rows)
    assert [r["entity"] for r in selected] == ["high", "low"]


def test_mature_ranks_by_importance():
    retriever = MemoryRetriever(None, None, scoring_threshold=3, max_facts=10)
    now = datetime.now(UTC)
    rows = [
        # Higher confidence but stale and rarely used.
        {"id": "x", "entity": "x", "confidence": 0.95, "access_count": 4, "created_at": now - timedelta(days=400)},
        # Lower confidence but heavily used and fresh -> higher importance.
        {"id": "y", "entity": "y", "confidence": 0.7, "access_count": 100, "created_at": now},
    ]
    selected = retriever._rank_and_select(rows)
    assert [r["entity"] for r in selected] == ["y", "x"]


# ---- cleanup -------------------------------------------------------------


def _make_system(tmp_path):
    config = MemoryConfig(
        database_url=f"sqlite:///{tmp_path}/cleanup.db",
        persist_dir=str(tmp_path / "chroma"),
        embedding_provider="hash",  # deterministic, no external deps in tests
    )
    system = MemorySystem(config=config, llm=None)
    return system


def _age_memory(system, memory_id):
    with system.semantic.session_factory() as session:
        session.execute(sa_update(Memory).where(Memory.id == memory_id).values(created_at=datetime(2020, 1, 1)))
        session.execute(sa_update(Fact).where(Fact.memory_id == memory_id).values(created_at=datetime(2020, 1, 1)))
        session.commit()


def test_cleanup_deletes_sql_and_vector(tmp_path):
    system = _make_system(tmp_path)
    vec = InMemoryVectorStore()
    system.episodic = EpisodicMemoryStore(vec, HashingEmbedding())

    memory_id = system.semantic.create_memory("u1", "old summary")
    system.semantic.save_fact(memory_id, _fact())
    system.episodic.store_summary(str(memory_id), "old summary", metadata={"user_id": "u1"})
    _age_memory(system, memory_id)

    report = system.cleanup()
    assert str(memory_id) in report["deleted"]
    # SQL Memory + Facts gone.
    assert system.semantic.get_facts_by_memory(memory_id) == []
    # Vector summary gone.
    assert vec._items == {}


def test_cleanup_logs_vector_failure(tmp_path, caplog):
    class FailingVectorStore(InMemoryVectorStore):
        def delete(self, ids):
            raise RuntimeError("vector backend down")

    system = _make_system(tmp_path)
    system.episodic = EpisodicMemoryStore(FailingVectorStore(), HashingEmbedding())

    memory_id = system.semantic.create_memory("u1", "old summary")
    system.semantic.save_fact(memory_id, _fact())
    system.episodic.store_summary(str(memory_id), "old summary", metadata={"user_id": "u1"})
    _age_memory(system, memory_id)

    with caplog.at_level(logging.ERROR):
        report = system.cleanup()

    assert str(memory_id) in report["failed"]
    # SQL was NOT deleted (vector failure aborts before touching SQL -> retryable).
    assert system.semantic.get_facts_by_memory(memory_id) != []
    assert any(str(memory_id) in rec.message for rec in caplog.records)


# ---- migration -----------------------------------------------------------


def test_migration_adds_missing_columns(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/legacy.db")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE memory (id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(255), "
                "summary TEXT, created_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE facts (id VARCHAR(36) PRIMARY KEY, memory_id VARCHAR(36), "
                "entity VARCHAR(255), relation VARCHAR(255), value TEXT, "
                "confidence FLOAT, source VARCHAR(255))"
            )
        )

    migrate(engine)

    cols = {c["name"] for c in inspect(engine).get_columns("facts")}
    for name in (
        "created_at",
        "updated_at",
        "status",
        "access_count",
        "positive_feedback_count",
        "negative_feedback_count",
    ):
        assert name in cols


# ---- lifecycle decoupling ------------------------------------------------


def test_cleanup_keeps_memory_with_partial_active_facts(tmp_path):
    system = _make_system(tmp_path)
    vec = InMemoryVectorStore()
    system.episodic = EpisodicMemoryStore(vec, HashingEmbedding())

    memory_id = system.semantic.create_memory("u1", "summary")
    fact_a = system.semantic.save_fact(memory_id, _fact(entity="A", value="va"))
    system.semantic.save_fact(memory_id, _fact(entity="B", value="vb"))
    system.episodic.store_summary(str(memory_id), "summary", metadata={"user_id": "u1"})

    # Age only fact A so it expires; fact B stays active.
    with system.semantic.session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.id == fact_a.id).values(created_at=datetime(2020, 1, 1)))
        session.commit()

    report = system.cleanup()

    assert str(memory_id) not in report["deleted"]
    assert len(system.semantic.get_facts_by_memory(memory_id)) == 2
    assert str(memory_id) in vec._items  # vector summary kept
    statuses = {f["entity"]: f["status"] for f in system.semantic.get_facts_by_memory(memory_id)}
    assert statuses["A"] == "expired"
    assert statuses["B"] == "active"


def test_cleanup_deletes_when_all_facts_inactive(tmp_path):
    system = _make_system(tmp_path)
    vec = InMemoryVectorStore()
    system.episodic = EpisodicMemoryStore(vec, HashingEmbedding())

    memory_id = system.semantic.create_memory("u1", "summary")
    system.semantic.save_fact(memory_id, _fact(entity="A", value="va"))
    system.semantic.save_fact(memory_id, _fact(entity="B", value="vb"))
    system.episodic.store_summary(str(memory_id), "summary", metadata={"user_id": "u1"})

    # All facts age past TTL.
    with system.semantic.session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.memory_id == memory_id).values(created_at=datetime(2020, 1, 1)))
        session.commit()

    report = system.cleanup()
    assert str(memory_id) in report["deleted"]
    assert system.semantic.get_facts_by_memory(memory_id) == []
    assert vec._items == {}


# ---- retrieval filtering -------------------------------------------------


def test_retriever_excludes_non_active_facts(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=5)
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="active", value="v1"))
    expired = semantic.save_fact(memory_id, _fact(entity="expired", value="v2"))
    superseded = semantic.save_fact(memory_id, _fact(entity="superseded", value="v3"))
    retracted = semantic.save_fact(memory_id, _fact(entity="retracted", value="v4"))

    semantic.update_fact(expired.id, status="expired")
    semantic.update_fact(superseded.id, status="superseded")
    semantic.update_fact(retracted.id, status="retracted")
    episodic.store_summary(str(memory_id), "summary", metadata={"user_id": "alice"})

    context = retriever.retrieve("query", user_id="alice")
    assert [f.entity for f in context.facts] == ["active"]


def test_batch_fetch_and_bulk_access_count(semantic, episodic, session_factory):
    from sqlalchemy import event

    memory_ids = []
    for i in range(3):
        mid = semantic.create_memory("alice", f"summary {i}")
        semantic.save_fact(mid, _fact(entity=f"E{i}", value=f"v{i}", confidence=0.8))
        episodic.store_summary(str(mid), f"summary {i}", metadata={"user_id": "alice"})
        memory_ids.append(mid)

    engine = session_factory.kw["bind"]
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        retriever = MemoryRetriever(episodic, semantic, top_k=5)
        context = retriever.retrieve("query", user_id="alice")
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert len(context.facts) == 3
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT") and "facts" in s.lower()]
    updates = [s for s in statements if s.lstrip().upper().startswith("UPDATE") and "facts" in s.lower()]
    assert len(selects) == 1  # one batched SELECT for all memories
    assert len(updates) == 1  # one batched UPDATE for all access_count bumps
    for mid in memory_ids:
        for f in semantic.get_facts_by_memory(mid):
            assert f["access_count"] == 1


# ---- user_id isolation ----------------------------------------------------


def test_user_isolation_vector_and_sql(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=10)

    a_mid = semantic.create_memory("alice", "alice task summary")
    semantic.save_fact(a_mid, _fact(entity="gene_A", value="alice-only"))
    episodic.store_summary(str(a_mid), "alice task summary", metadata={"user_id": "alice"})

    b_mid = semantic.create_memory("bob", "bob task summary")
    semantic.save_fact(b_mid, _fact(entity="gene_B", value="bob-only"))
    episodic.store_summary(str(b_mid), "bob task summary", metadata={"user_id": "bob"})

    assert {f.entity for f in retriever.retrieve("task", user_id="alice").facts} == {"gene_A"}
    assert {f.entity for f in retriever.retrieve("task", user_id="bob").facts} == {"gene_B"}


def test_similar_query_cannot_leak_across_users(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=10)

    # Two users store near-identical summaries; a shared query must not cross over.
    a_mid = semantic.create_memory("alice", "find EGFR differentially expressed genes")
    semantic.save_fact(a_mid, _fact(entity="EGFR", value="alice result"))
    episodic.store_summary(str(a_mid), "find EGFR differentially expressed genes", metadata={"user_id": "alice"})

    b_mid = semantic.create_memory("bob", "find EGFR differentially expressed genes")
    semantic.save_fact(b_mid, _fact(entity="EGFR", value="bob result"))
    episodic.store_summary(str(b_mid), "find EGFR differentially expressed genes", metadata={"user_id": "bob"})

    ctx = retriever.retrieve("EGFR differentially expressed genes", user_id="alice")
    assert {f.value for f in ctx.facts} == {"alice result"}


def test_sql_rejects_cross_user_memory_id(semantic):
    a_mid = semantic.create_memory("alice", "alice summary")
    semantic.save_fact(a_mid, _fact(entity="gene_A", value="alice-only"))

    # Even with Alice's memory_id, Bob's user_id returns nothing.
    assert semantic.get_active_facts_by_memories([a_mid], user_id="bob") == []
    # The owner still sees their own facts.
    rows = semantic.get_active_facts_by_memories([a_mid], user_id="alice")
    assert [r["entity"] for r in rows] == ["gene_A"]


def test_single_user_retrieval_unchanged(semantic, episodic):
    retriever = MemoryRetriever(episodic, semantic, top_k=10, max_facts=10)
    m1 = semantic.create_memory("alice", "summary one")
    semantic.save_fact(m1, _fact(entity="A", value="va", confidence=0.9))
    episodic.store_summary(str(m1), "summary one", metadata={"user_id": "alice"})
    m2 = semantic.create_memory("alice", "summary two")
    semantic.save_fact(m2, _fact(entity="B", value="vb", confidence=0.7))
    episodic.store_summary(str(m2), "summary two", metadata={"user_id": "alice"})

    ctx = retriever.retrieve("query", user_id="alice")
    assert {f.entity for f in ctx.facts} == {"A", "B"}


def test_store_summary_requires_user_id(episodic):
    with pytest.raises(ValueError, match="user_id"):
        episodic.store_summary("mem-no-owner", "summary")


def test_store_summary_rejects_blank_user_id(episodic):
    with pytest.raises(ValueError, match="user_id"):
        episodic.store_summary("mem-blank-owner", "summary", metadata={"user_id": "   "})


def test_vector_metadata_retains_user_id(episodic, vector_store):
    episodic.store_summary("a-mem", "alice summary", metadata={"user_id": "alice"})
    episodic.store_summary("b-mem", "bob summary", metadata={"user_id": "bob"})

    # The stored metadata carries user_id so search can scope by owner.
    assert vector_store.get(["a-mem"])[0].metadata["user_id"] == "alice"

    # A cross-user search at the vector layer never leaks the other owner.
    bob_hits = episodic.search_memory("summary", user_id="bob", k=10)
    assert {h.id for h in bob_hits} == {"b-mem"}
    alice_hits = episodic.search_memory("summary", user_id="alice", k=10)
    assert {h.id for h in alice_hits} == {"a-mem"}


# ---- fact conflict detection & update ------------------------------------


def test_same_fact_not_duplicated(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    f1 = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG", confidence=0.8))
    f2 = semantic.save_fact(
        memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG", confidence=0.95)
    )
    facts = semantic.get_facts_by_memory(memory_id)
    assert len(facts) == 1  # no duplicate row
    assert str(f2.id) == str(f1.id)  # reused the existing row
    assert facts[0]["confidence"] == 0.95  # confidence updated


def test_different_value_supersedes_old(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    old = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="185delAG"))
    new = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="5382insC"))

    facts = semantic.get_facts_by_memory(memory_id)
    statuses = {f["value"]: f["status"] for f in facts}
    assert statuses == {"185delAG": "superseded", "5382insC": "active"}
    assert len(facts) == 2  # history preserved
    assert str(new.id) != str(old.id)


def test_non_active_facts_do_not_participate_in_conflict(semantic):
    memory_id = semantic.create_memory("alice", "summary")

    # superseded chain: v1 -> superseded by v2
    semantic.save_fact(memory_id, _fact(entity="TP53", relation="current_status", value="v1"))
    semantic.save_fact(memory_id, _fact(entity="TP53", relation="current_status", value="v2"))

    r = semantic.save_fact(memory_id, _fact(entity="EGFR", relation="current_status", value="r1"))
    semantic.update_fact(r.id, status="retracted")
    e = semantic.save_fact(memory_id, _fact(entity="KRAS", relation="current_status", value="e1"))
    semantic.update_fact(e.id, status="expired")

    # New values: only the *active* v2 is superseded; retracted/expired stay put.
    semantic.save_fact(memory_id, _fact(entity="TP53", relation="current_status", value="v3"))
    semantic.save_fact(memory_id, _fact(entity="EGFR", relation="current_status", value="r2"))
    semantic.save_fact(memory_id, _fact(entity="KRAS", relation="current_status", value="e2"))

    by_key = {(f["entity"], f["value"]): f["status"] for f in semantic.get_facts_by_memory(memory_id)}
    assert by_key[("TP53", "v1")] == "superseded"
    assert by_key[("TP53", "v2")] == "superseded"
    assert by_key[("TP53", "v3")] == "active"
    assert by_key[("EGFR", "r1")] == "retracted"  # not flipped to superseded
    assert by_key[("EGFR", "r2")] == "active"
    assert by_key[("KRAS", "e1")] == "expired"  # not flipped to superseded
    assert by_key[("KRAS", "e2")] == "active"


def test_same_fact_across_users_isolated(semantic):
    a_mid = semantic.create_memory("alice", "summary")
    b_mid = semantic.create_memory("bob", "summary")

    semantic.save_fact(a_mid, _fact(entity="BRCA1", relation="has_mutation", value="185delAG"))
    semantic.save_fact(b_mid, _fact(entity="BRCA1", relation="has_mutation", value="185delAG"))

    a_facts = semantic.get_facts_by_memory(a_mid)
    b_facts = semantic.get_facts_by_memory(b_mid)
    assert len(a_facts) == 1 and a_facts[0]["status"] == "active"
    assert len(b_facts) == 1 and b_facts[0]["status"] == "active"

    # A different value for Alice must not touch Bob's fact.
    semantic.save_fact(a_mid, _fact(entity="BRCA1", relation="has_mutation", value="5382insC"))
    b_facts = semantic.get_facts_by_memory(b_mid)
    assert len(b_facts) == 1
    assert b_facts[0]["status"] == "active"
    assert b_facts[0]["value"] == "185delAG"


def test_same_fact_preserves_created_at(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    f1 = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG", confidence=0.8))
    original_created = f1.created_at
    assert original_created is not None

    f2 = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG", confidence=0.9))
    assert str(f2.id) == str(f1.id)
    assert f2.created_at == original_created  # created_at never overwritten


def test_supersede_updates_updated_at(semantic, session_factory):
    memory_id = semantic.create_memory("alice", "summary")
    old = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="v1"))

    # Force updated_at far into the past so the onupdate bump is observable.
    with session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.id == old.id).values(updated_at=datetime(2000, 1, 1)))
        session.commit()

    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="v2"))

    old_row = [f for f in semantic.get_facts_by_memory(memory_id) if f["value"] == "v1"][0]
    assert old_row["status"] == "superseded"
    assert old_row["updated_at"] > datetime(2000, 1, 1)  # status change bumped updated_at


def test_conflict_supersede_and_insert_atomic(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="v1"))
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="current_status", value="v2"))

    facts = semantic.get_facts_by_memory(memory_id)
    statuses = {f["value"]: f["status"] for f in facts}
    # Old superseded AND new active exist together — the two sides never diverge.
    assert statuses == {"v1": "superseded", "v2": "active"}
    assert len(facts) == 2


def test_save_facts_uses_conflict_detection(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    rows = semantic.save_facts(
        memory_id,
        [
            _fact(entity="BRCA1", relation="current_status", value="v1"),
            _fact(entity="BRCA1", relation="current_status", value="v2"),
        ],
    )
    assert len(rows) == 2
    statuses = {f["value"]: f["status"] for f in semantic.get_facts_by_memory(memory_id)}
    assert statuses == {"v1": "superseded", "v2": "active"}


# ---- relation cardinality ------------------------------------------------


def test_single_value_relation_supersedes(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="TP53", relation="current_status", value="active"))
    semantic.save_fact(memory_id, _fact(entity="TP53", relation="current_status", value="inactive"))

    statuses = {f["value"]: f["status"] for f in semantic.get_facts_by_memory(memory_id)}
    assert statuses == {"active": "superseded", "inactive": "active"}


def test_multi_value_relation_keeps_both_active(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG"))
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="5382insC"))

    statuses = {f["value"]: f["status"] for f in semantic.get_facts_by_memory(memory_id)}
    assert statuses == {"185delAG": "active", "5382insC": "active"}


def test_multi_value_relation_same_value_dedup(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    f1 = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG"))
    f2 = semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="has_mutation", value="185delAG"))

    facts = semantic.get_facts_by_memory(memory_id)
    assert len(facts) == 1
    assert str(f1.id) == str(f2.id)


def test_unknown_relation_defaults_to_multi(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="some_unknown_relation", value="v1"))
    semantic.save_fact(memory_id, _fact(entity="BRCA1", relation="some_unknown_relation", value="v2"))

    statuses = {f["value"]: f["status"] for f in semantic.get_facts_by_memory(memory_id)}
    assert statuses == {"v1": "active", "v2": "active"}


# ---- user feedback --------------------------------------------------------


def test_positive_feedback_records_count(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    updated = semantic.update_fact_feedback(row.id, 1, "alice")
    assert updated.positive_feedback_count == 1
    assert updated.negative_feedback_count == 0


def test_negative_feedback_records_count(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    updated = semantic.update_fact_feedback(row.id, -1, "alice")
    assert updated.negative_feedback_count == 1
    assert updated.positive_feedback_count == 0


def test_invalid_feedback_rejected(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    with pytest.raises(ValueError):
        semantic.update_fact_feedback(row.id, 0, "alice")
    with pytest.raises(ValueError):
        semantic.update_fact_feedback(row.id, 2, "alice")


def test_feedback_affects_importance_score():
    now = datetime.now(UTC)
    kwargs = {
        "confidence": 0.5,
        "access_count": 0,
        "created_at": now,
        "now": now,
        "confidence_weight": 0.4,
        "usage_weight": 0.2,
        "recency_weight": 0.2,
        "feedback_weight": 0.2,
        "usage_saturation": 100.0,
        "recency_lambda": 0.01,
    }
    neutral = importance_score(**kwargs, positive_feedback=0, negative_feedback=0)
    praised = importance_score(**kwargs, positive_feedback=10, negative_feedback=0)
    criticized = importance_score(**kwargs, positive_feedback=0, negative_feedback=10)

    assert feedback_score(0, 0) == 0.5  # no feedback is neutral, not "low quality"
    assert praised > neutral
    assert criticized < neutral
    assert praised > criticized


def test_negative_feedback_does_not_delete(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    semantic.update_fact_feedback(row.id, -1, "alice")

    facts = semantic.get_facts_by_memory(memory_id)
    assert len(facts) == 1  # still exists
    assert facts[0]["status"] == "active"  # one negative does not retract
    assert facts[0]["negative_feedback_count"] == 1


def test_retract_at_configured_threshold(session_factory):
    semantic = SemanticMemoryStore(session_factory, FactValidator(), feedback_retract_threshold=2)
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())

    semantic.update_fact_feedback(row.id, -1, "alice")
    assert semantic.get_facts_by_memory(memory_id)[0]["status"] == "active"

    semantic.update_fact_feedback(row.id, -1, "alice")
    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["status"] == "retracted"
    assert facts[0]["negative_feedback_count"] == 2
    assert len(facts) == 1  # retracted, never deleted


def test_cross_user_cannot_feedback(semantic):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())

    assert semantic.update_fact_feedback(row.id, 1, "bob") is None
    assert semantic.update_fact_feedback(row.id, -1, "bob") is None

    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["positive_feedback_count"] == 0
    assert facts[0]["negative_feedback_count"] == 0
    assert facts[0]["status"] == "active"


# ---- feedback timestamp semantics ----------------------------------------


def _pin_updated_at(session_factory, fact_id):
    """Force updated_at far into the past so any bump is observable."""
    with session_factory() as session:
        session.execute(sa_update(Fact).where(Fact.id == fact_id).values(updated_at=datetime(2000, 1, 1)))
        session.commit()


def test_positive_feedback_does_not_bump_updated_at(semantic, session_factory):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    _pin_updated_at(session_factory, row.id)

    semantic.update_fact_feedback(row.id, 1, "alice")

    assert semantic.get_facts_by_memory(memory_id)[0]["updated_at"] == datetime(2000, 1, 1)


def test_negative_feedback_does_not_bump_updated_at(semantic, session_factory):
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    _pin_updated_at(session_factory, row.id)

    # One negative is below the default threshold of 3, so no retract, no bump.
    semantic.update_fact_feedback(row.id, -1, "alice")

    assert semantic.get_facts_by_memory(memory_id)[0]["updated_at"] == datetime(2000, 1, 1)


def test_retract_bumps_updated_at(session_factory):
    semantic = SemanticMemoryStore(session_factory, FactValidator(), feedback_retract_threshold=2)
    memory_id = semantic.create_memory("alice", "summary")
    row = semantic.save_fact(memory_id, _fact())
    _pin_updated_at(session_factory, row.id)

    # First negative: below threshold -> still active, updated_at unchanged.
    semantic.update_fact_feedback(row.id, -1, "alice")
    assert semantic.get_facts_by_memory(memory_id)[0]["updated_at"] == datetime(2000, 1, 1)

    # Second negative: crosses threshold -> retracted (lifecycle change) bumps updated_at.
    semantic.update_fact_feedback(row.id, -1, "alice")
    facts = semantic.get_facts_by_memory(memory_id)
    assert facts[0]["status"] == "retracted"
    assert facts[0]["updated_at"] > datetime(2000, 1, 1)


# ---- query-aware fact reranking ------------------------------------------


class _FakeSemanticEmbedding:
    """Deterministic fake embedding for query-aware reranking (no external API).

    Gives high cosine similarity to texts sharing BRCA1/mutation vocabulary and
    zero similarity to unrelated texts, so threshold filtering is testable.
    """

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "brca1" in t else 0.0,
            1.0 if "mutation" in t else 0.0,
            1.0 if "185delag" in t else 0.0,
        ]

    def embed_documents(self, texts) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_fact_semantic_reranking_ranks_but_does_not_drop(semantic, episodic):
    """Query-aware reranking ranks the relevant fact first but no longer drops the
    low-similarity facts: semantic similarity only influences rank, never deletion.

    The two high-confidence-but-unrelated facts (``tool``, ``user``) have a cosine
    similarity of ~0 to the query, which is below the old ``fact_similarity_threshold``
    of 0.45; they must still appear in the context (ranked below the relevant fact).
    """
    retriever = MemoryRetriever(episodic, semantic, top_k=5, embedding=_FakeSemanticEmbedding())
    memory_id = semantic.create_memory("alice", "BRCA1 analysis")

    semantic.save_fact(
        memory_id,
        _fact(entity="BRCA1", relation="pathogenic_mutation", value="185delAG", confidence=0.8),
    )
    semantic.save_fact(memory_id, _fact(entity="tool", relation="used", value="parse_vcf", confidence=0.95))
    semantic.save_fact(memory_id, _fact(entity="user", relation="likes", value="Python", confidence=0.95))
    episodic.store_summary(str(memory_id), "BRCA1 analysis", metadata={"user_id": "alice"})

    context = retriever.retrieve("BRCA1 mutations", user_id="alice")
    entities = [f.entity for f in context.facts]

    # The relevant fact is still ranked first...
    assert entities[0] == "BRCA1"
    assert context.facts[0].value == "185delAG"
    # ...but the low-similarity facts are no longer filtered out.
    assert set(entities) == {"BRCA1", "tool", "user"}


def test_fact_score_formula(semantic, episodic):
    """FactScore is ``0.7 * semantic_similarity + 0.3 * importance_score``.

    A fixed embedding forces an exact cosine similarity of 0.2, and a cold-start
    fact's importance equals its confidence (0.8), giving the expected score 0.38.
    """

    class _FixedVectors:
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[0.2, math.sqrt(0.96)] for _ in texts]

    retriever = MemoryRetriever(episodic, semantic, embedding=_FixedVectors())
    now = datetime.now(UTC)
    rows = [
        {
            "entity": "gene",
            "relation": "log2fc",
            "value": "2.3",
            "confidence": 0.8,
            "access_count": 0,
            "created_at": now,
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
        }
    ]
    scored = retriever._score_query_aware(rows, "expression analysis", now)
    score, similarity, _row = scored[0]

    assert abs(similarity - 0.2) < 1e-6
    # cold-start importance == confidence == 0.8
    assert abs(score - (0.7 * 0.2 + 0.3 * 0.8)) < 1e-9
    assert abs(score - 0.38) < 1e-9


def test_reranking_still_caps_to_top_n(semantic, episodic):
    """The query-aware path still returns exactly the top ``max_facts`` by score.

    Truncation is by rank (top-N), not by a similarity threshold: the ``tool`` fact
    (similarity ~0) survives because it ranks 2nd, while ``user`` (also similarity ~0,
    lower confidence) is dropped only because it ranks 3rd.
    """
    retriever = MemoryRetriever(episodic, semantic, top_k=5, max_facts=2, embedding=_FakeSemanticEmbedding())
    memory_id = semantic.create_memory("alice", "BRCA1 analysis")
    semantic.save_fact(
        memory_id,
        _fact(entity="BRCA1", relation="pathogenic_mutation", value="185delAG", confidence=0.8),
    )
    semantic.save_fact(memory_id, _fact(entity="tool", relation="used", value="parse_vcf", confidence=0.95))
    semantic.save_fact(memory_id, _fact(entity="user", relation="likes", value="Python", confidence=0.6))
    episodic.store_summary(str(memory_id), "BRCA1 analysis", metadata={"user_id": "alice"})

    context = retriever.retrieve("BRCA1 mutations", user_id="alice")

    assert [f.entity for f in context.facts] == ["BRCA1", "tool"]


def test_query_aware_reranking_preserves_user_isolation(semantic, episodic):
    """Query-aware reranking must not leak facts across users, even when both users
    store the same ``entity relation value`` text (identical semantic similarity)."""
    retriever = MemoryRetriever(episodic, semantic, top_k=10, embedding=_FakeSemanticEmbedding())

    for user, value in (("alice", "alice result"), ("bob", "bob result")):
        mid = semantic.create_memory(user, "BRCA1 analysis")
        semantic.save_fact(mid, _fact(entity="BRCA1", relation="pathogenic_mutation", value=value))
        episodic.store_summary(str(mid), "BRCA1 analysis", metadata={"user_id": user})

    ctx_alice = retriever.retrieve("BRCA1 mutations", user_id="alice")
    ctx_bob = retriever.retrieve("BRCA1 mutations", user_id="bob")

    assert {f.value for f in ctx_alice.facts} == {"alice result"}
    assert {f.value for f in ctx_bob.facts} == {"bob result"}


# ---- embedding provider configuration -------------------------------------


def test_default_embedding_provider_is_semantic():
    """The out-of-the-box provider is semantic, not the deterministic hash."""
    assert MemoryConfig().embedding_provider == "sentence_transformer"
    assert MemoryConfig().embedding_provider != "hash"


def test_explicit_hash_provider_still_works():
    """Explicitly selecting ``hash`` still yields a HashingEmbedding."""
    provider = build_embedding_provider(MemoryConfig(embedding_provider="hash"))
    assert isinstance(provider, HashingEmbedding)


def test_unimplemented_vector_backends_raise_clear_error():
    """FAISS/Milvus are planned, not silently ignored: they must raise clearly."""
    for backend in ("faiss", "milvus"):
        with pytest.raises(NotImplementedError, match="Current supported backend: Chroma"):
            build_vector_store(MemoryConfig(vector_db=backend))
