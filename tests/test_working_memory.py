"""Tests for Working Memory: CRUD, immutability, persistence, backend wiring,
TTL cleanup, structured variables, isolation and optimistic locking.

Working memory is *current-task* state (unlike long-term episodic/semantic
memory), so these tests exercise the manager + both backends in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from database.migrations import get_engine, get_session_factory, migrate
from memory.models import MemoryConfig, MemoryFact, WorkingMemoryState
from memory.system import MemorySystem
from memory.working import (
    ConcurrentUpdateError,
    LangGraphCheckpointerStore,
    SQLWorkingMemoryStore,
    WorkingMemoryManager,
)


@pytest.fixture
def session_factory(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/memory.db")
    migrate(engine)
    return get_session_factory(engine)


@pytest.fixture
def manager(session_factory):
    return WorkingMemoryManager(SQLWorkingMemoryStore(session_factory))


class _FakeCheckpointer:
    """Minimal in-memory stand-in for a LangGraph checkpointer (put/get)."""

    def __init__(self) -> None:
        self._store: dict = {}

    def put(self, config, checkpoint, metadata=None, new_versions=None) -> None:
        self._store[config["configurable"]["thread_id"]] = checkpoint

    def get(self, config):
        return self._store.get(config["configurable"]["thread_id"])


def _make_system(tmp_path, checkpointer=None):
    config = MemoryConfig(
        database_url=f"sqlite:///{tmp_path}/memory.db",
        persist_dir=str(tmp_path / "chroma"),
        embedding_provider="hash",
    )
    return MemorySystem(config=config, llm=None, checkpointer=checkpointer)


# ---- 1. CRUD -------------------------------------------------------------


def test_working_memory_crud(manager):
    manager.create_state("u1", "t1", current_step="start", next_action="run", variables={"x": 1})
    loaded = manager.load_state("u1", "t1")
    assert loaded is not None
    assert loaded.task_id == "t1"
    assert loaded.user_id == "u1"
    assert loaded.current_step == "start"
    assert loaded.next_action == "run"
    assert loaded.variables == {"x": 1}

    manager.update_state("u1", "t1", current_step="mid")
    assert manager.load_state("u1", "t1").current_step == "mid"

    manager.clear("u1", "t1")
    assert manager.load_state("u1", "t1") is None


# ---- 2. task_id / user_id immutability ------------------------------------


def test_update_state_cannot_change_identity(manager):
    manager.create_state("u1", "t1", current_step="old")
    updated = manager.update_state("u1", "t1", current_step="x")

    # identity is immutable: the update changes only the mutable content.
    assert updated.task_id == "t1"
    assert updated.user_id == "u1"
    assert updated.current_step == "x"
    # (user_id, task_id) is the load key, not an update field: a different key is a
    # *different* entry, never a mutation of this one.
    assert manager.load_state("u1", "t2") is None
    assert manager.load_state("u2", "t1") is None


def test_update_whitelist_only_mutates_content(manager):
    manager.create_state("u1", "t1", current_step="old", next_action="na", variables={"a": 1})
    # Unknown keys are ignored; only the whitelisted content fields change.
    manager.update_state("u1", "t1", current_step="new", unrelated="ignored")
    loaded = manager.load_state("u1", "t1")
    assert loaded.task_id == "t1"
    assert loaded.user_id == "u1"
    assert loaded.current_step == "new"
    assert loaded.next_action == "na"
    assert loaded.variables == {"a": 1}


# ---- 4. persistence round-trip -------------------------------------------


def test_working_memory_persistence_roundtrip(session_factory):
    store = SQLWorkingMemoryStore(session_factory)
    store.save(
        WorkingMemoryState(
            task_id="t1",
            user_id="u1",
            current_step="step",
            variables={"a": 1, "b": "x"},
            next_action="go",
        )
    )
    loaded = store.load("u1", "t1")
    assert loaded is not None
    assert loaded.current_step == "step"
    assert loaded.variables == {"a": 1, "b": "x"}
    assert loaded.next_action == "go"


# ---- 5. checkpointer backend wiring ---------------------------------------


def test_memory_system_uses_checkpointer_store_when_provided(tmp_path):
    system = _make_system(tmp_path, checkpointer=_FakeCheckpointer())
    assert isinstance(system.working.store, LangGraphCheckpointerStore)


def test_memory_system_defaults_to_sql_store_without_checkpointer(tmp_path):
    system = _make_system(tmp_path)
    assert isinstance(system.working.store, SQLWorkingMemoryStore)


def test_checkpointer_store_roundtrip():
    store = LangGraphCheckpointerStore(_FakeCheckpointer())
    store.save(WorkingMemoryState(task_id="t1", user_id="u1", current_step="running", variables={"g": "EGFR"}))
    loaded = store.load("u1", "t1")
    assert loaded is not None
    assert loaded.current_step == "running"
    assert loaded.variables == {"g": "EGFR"}
    # user/task isolation is encoded in the checkpoint thread_id.
    assert store.load("u2", "t1") is None
    assert store.load("u1", "t2") is None


def test_checkpointer_store_delete_is_explicitly_unsupported():
    store = LangGraphCheckpointerStore(_FakeCheckpointer())
    with pytest.raises(NotImplementedError):
        store.delete("u1", "t1")


# ---- 6. cleanup does not touch long-term memory ---------------------------


def test_working_memory_cleanup_does_not_touch_longterm(tmp_path):
    system = _make_system(tmp_path)
    mid = system.semantic.create_memory("u1", "summary")
    system.semantic.save_fact(
        mid,
        MemoryFact(
            entity="BRCA1",
            relation="has_mutation",
            value="185delAG",
            confidence=0.9,
            source="tool_result",
        ),
    )
    system.working.create_state("u1", "t1", current_step="x")

    system.working.clear("u1", "t1")

    assert system.working.load_state("u1", "t1") is None
    # long-term episodic/semantic data is unaffected by working-memory cleanup.
    assert len(system.semantic.get_facts_by_memory(mid)) == 1


# ---- 7. TTL cleanup --------------------------------------------------------


def test_working_memory_ttl_cleanup(session_factory):
    store = SQLWorkingMemoryStore(session_factory, ttl_days=7)
    store.save(WorkingMemoryState(task_id="t1", user_id="u1"))
    loaded = store.load("u1", "t1")
    assert loaded is not None and loaded.expires_at is not None

    # Not yet expired.
    assert store.delete_expired(datetime.now(UTC)) == 0
    # 8 days later the entry has lapsed and is removed.
    future = datetime.now(UTC) + timedelta(days=8)
    assert store.delete_expired(future) == 1
    assert store.load("u1", "t1") is None


def test_manager_cleanup_returns_count(session_factory):
    manager = WorkingMemoryManager(SQLWorkingMemoryStore(session_factory, ttl_days=7))
    manager.create_state("u1", "t1")
    manager.create_state("u1", "t2")
    future = datetime.now(UTC) + timedelta(days=8)
    assert manager.cleanup(future) == 2


# ---- 8. structured variables API ------------------------------------------


def test_working_memory_variables_api(manager):
    manager.set_variable("u1", "t1", "current_gene", "EGFR")
    assert manager.get_variable("u1", "t1", "current_gene") == "EGFR"
    assert manager.get_variable("u1", "t1", "missing") is None
    assert manager.get_variable("u1", "t1", "missing", default="dflt") == "dflt"

    assert manager.delete_variable("u1", "t1", "current_gene") is True
    assert manager.get_variable("u1", "t1", "current_gene", default=None) is None
    # deleting a non-existent key reports False and leaves state intact.
    assert manager.delete_variable("u1", "t1", "current_gene") is False


# ---- 9. user / task isolation ---------------------------------------------


def test_working_memory_user_task_isolation(manager):
    manager.set_variable("uA", "tA", "k", "A")
    manager.set_variable("uB", "tB", "k", "B")
    # Same task_id under different users must not collide.
    manager.set_variable("uA", "tX", "k", "AX")
    manager.set_variable("uB", "tX", "k", "BX")

    assert manager.get_variable("uA", "tX", "k") == "AX"
    assert manager.get_variable("uB", "tX", "k") == "BX"
    # Cross-user / cross-task reads return nothing.
    assert manager.get_variable("uA", "tB", "k") is None
    assert manager.get_variable("uB", "tA", "k") is None
    # The original entries are untouched.
    assert manager.get_variable("uA", "tA", "k") == "A"
    assert manager.get_variable("uB", "tB", "k") == "B"


# ---- 10. optimistic locking ------------------------------------------------


def test_optimistic_lock_detects_stale_version(session_factory):
    store = SQLWorkingMemoryStore(session_factory)
    store.save(WorkingMemoryState(task_id="t1", user_id="u1", current_step="a"))

    # A concurrent writer advances the version to 1.
    WorkingMemoryManager(store).update_state("u1", "t1", current_step="b")

    # A stale writer still on version 0 must fail rather than silently overwrite.
    stale = WorkingMemoryState(task_id="t1", user_id="u1", current_step="stale", version=0)
    with pytest.raises(ConcurrentUpdateError):
        store.save(stale)
