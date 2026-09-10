"""Integration tests for the a1.py memory wiring (recall-on-run, persist-after-run).

These require the full Biomni agent environment (langgraph, langchain, ...). They
are skipped automatically when that environment is not installed, so the memory
subsystem's own tests can still run in isolation.
"""
from __future__ import annotations

import logging

import pytest

try:
    from biomni.agent.a1 import A1
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - depends on optional deps
    pytest.skip(f"a1.py dependencies unavailable: {exc}", allow_module_level=True)

from memory.models import WorkingMemoryState


class _FakeWorkingMemory:
    """Records working-memory load/clear calls so the A1 wiring can be asserted."""

    def __init__(self, state: WorkingMemoryState | None = None) -> None:
        self.state = state
        self.load_calls: list[tuple] = []
        self.clear_calls: list[tuple] = []

    def load_state(self, user_id: str, task_id: str) -> WorkingMemoryState | None:
        self.load_calls.append((user_id, task_id))
        return self.state

    def clear(self, user_id: str, task_id: str) -> None:
        self.clear_calls.append((user_id, task_id))


class _FakeMemory:
    """Records recall/persist calls so the wiring can be asserted without a real DB."""

    def __init__(self, retrieve_result: str = "") -> None:
        self.retrieve_result = retrieve_result
        self.retrieve_calls: list[tuple] = []
        self.ingest_calls: list[tuple] = []
        self.working = _FakeWorkingMemory()

    def retrieve(self, query: str, user_id: str) -> str:
        self.retrieve_calls.append((query, user_id))
        return self.retrieve_result

    def ingest_background(self, trace, user_id=None, task_id=None) -> None:
        self.ingest_calls.append((trace, user_id, task_id))


class _FakeApp:
    """Stands in for the compiled LangGraph app: yields a single completed state."""

    def __init__(self) -> None:
        self.last_inputs = None

    def stream(self, inputs, stream_mode, config):
        self.last_inputs = inputs
        yield {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="<solution>done</solution>"),
            ]
        }


class _RaisingMemory(_FakeMemory):
    """Raises from retrieve/persist to exercise best-effort (non-blocking) failure."""

    def __init__(self, retrieve_error: bool = False, ingest_error: bool = False) -> None:
        super().__init__()
        self.retrieve_error = retrieve_error
        self.ingest_error = ingest_error

    def retrieve(self, query: str, user_id: str) -> str:
        if self.retrieve_error:
            raise RuntimeError("retrieve boom")
        return super().retrieve(query, user_id)

    def ingest_background(self, trace, user_id=None, task_id=None) -> None:
        if self.ingest_error:
            raise RuntimeError("ingest boom")
        super().ingest_background(trace, user_id, task_id)


def _make_agent(memory):
    agent = A1.__new__(A1)
    agent._memory_enabled = True
    agent.memory = memory
    agent.user_id = "default"
    agent.use_tool_retriever = False
    agent.log = []
    return agent


def test_build_initial_messages_injects_memory():
    mem = _FakeMemory(retrieve_result="Previous Task(s):\n- analyzed BRCA1")
    agent = _make_agent(mem)

    messages = agent._build_initial_messages("analyze TP53")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert "BRCA1" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "analyze TP53"
    # user_id must be threaded through (regression for the missing-arg bug).
    assert mem.retrieve_calls == [("analyze TP53", "default")]


def test_build_initial_messages_empty_memory_is_noop():
    mem = _FakeMemory(retrieve_result="")
    agent = _make_agent(mem)

    messages = agent._build_initial_messages("query")

    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)


def test_go_recalls_then_persists():
    mem = _FakeMemory(retrieve_result="Previous Task(s):\n- analyzed BRCA1")
    agent = _make_agent(mem)
    agent.app = _FakeApp()

    agent.go("analyze TP53")

    # recall happened up front, once, with the default user.
    assert mem.retrieve_calls == [("analyze TP53", "default")]
    # persistence fired exactly once after the run completed.
    assert len(mem.ingest_calls) == 1


def test_retrieve_failure_does_not_block(caplog):
    agent = _make_agent(_RaisingMemory(retrieve_error=True))
    with caplog.at_level(logging.WARNING, logger="biomni.agent.a1"):
        messages = agent._build_initial_messages("query")
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "Memory retrieval failed" in caplog.text


def test_persist_failure_does_not_block(caplog):
    agent = _make_agent(_RaisingMemory(ingest_error=True))
    agent.app = _FakeApp()
    with caplog.at_level(logging.WARNING, logger="biomni.agent.a1"):
        log, content = agent.go("query")
    assert content == "<solution>done</solution>"
    assert "Memory persistence failed" in caplog.text


# ---- Working Memory integration -------------------------------------------

def test_go_carries_working_memory_into_state():
    mem = _FakeMemory()
    mem.working = _FakeWorkingMemory(
        state=WorkingMemoryState(task_id="task-123", user_id="default", current_step="idle")
    )
    agent = _make_agent(mem)
    agent.app = _FakeApp()

    agent.go("query", task_id="task-123")

    # Working memory was loaded and carried into the initial AgentState...
    assert mem.working.load_calls == [("default", "task-123")]
    assert agent.app.last_inputs["working"]["task_id"] == "task-123"
    assert agent.app.last_inputs["working"]["current_step"] == "idle"
    # ...and cleared at the end of the run.
    assert mem.working.clear_calls == [("default", "task-123")]
    # The same task_id was threaded into long-term persistence.
    assert mem.ingest_calls[0][2] == "task-123"


def test_go_no_working_memory_is_noop():
    mem = _FakeMemory()
    agent = _make_agent(mem)
    agent.app = _FakeApp()

    agent.go("query")

    # load + clear still fire (best-effort), but working state is absent.
    assert agent.app.last_inputs["working"] is None
    assert len(mem.working.load_calls) == 1
    assert len(mem.working.clear_calls) == 1


def test_working_memory_clear_failure_does_not_block(caplog):
    class _BoomWorking(_FakeWorkingMemory):
        def clear(self, user_id, task_id):
            raise RuntimeError("clear boom")

    mem = _FakeMemory()
    mem.working = _BoomWorking()
    agent = _make_agent(mem)
    agent.app = _FakeApp()

    with caplog.at_level(logging.WARNING, logger="biomni.agent.a1"):
        log, content = agent.go("query", task_id="task-1")

    assert content == "<solution>done</solution>"
    assert "Working memory cleanup failed" in caplog.text
