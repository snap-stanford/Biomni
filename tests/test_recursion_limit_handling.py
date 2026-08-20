"""Tests for A1 agent recursion-limit handling (issue #237).

When the LangGraph execution hits the recursion limit, ``A1.go()`` /
``go_stream()`` must return a user-facing summary of what was accomplished
instead of surfacing a raw ``GraphRecursionError``. Also verifies that
``thread_id`` is isolated per call by default (no state accumulation) and
that ``recursion_limit`` is configurable.
"""

import sys
import types
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The A1 agent module pulls in heavy dependencies (langgraph, langchain,
# pandas, biopython, ...). We stub the missing/broken ones so the module can
# be imported and the agent logic exercised in isolation.
try:
    import numpy

    has_ndarray = hasattr(numpy, "ndarray")
except Exception:
    has_ndarray = False

if not has_ndarray:
    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = type("ndarray", (), {})
    fake_np.__version__ = "99.0"
    sys.modules["numpy"] = fake_np
    fake_pd = types.ModuleType("pandas")
    fake_pd.DataFrame = type("DataFrame", (), {})
    sys.modules["pandas"] = fake_pd

try:
    import Bio  # noqa: F401
except ImportError:
    fake_bio = types.ModuleType("Bio")
    fake_blast = types.ModuleType("Bio.Blast")
    fake_blast.NCBIWWW = types.SimpleNamespace()
    fake_blast.NCBIXML = types.SimpleNamespace()
    fake_blast.__path__ = []
    sys.modules["Bio.Blast"] = fake_blast
    fake_seq = types.ModuleType("Bio.Seq")
    fake_seq.Seq = type("Seq", (), {})
    sys.modules["Bio.Seq"] = fake_seq
    fake_bio.Blast = fake_blast
    fake_bio.Seq = fake_seq
    fake_bio.__path__ = []
    sys.modules["Bio"] = fake_bio

sys.path.insert(0, str(REPO_ROOT))

from biomni.agent.a1 import A1
from biomni.utils import pretty_print
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError


class _NormalStream:
    """Fake LangGraph stream that yields one AIMessage then stops."""

    def __init__(self):
        self.call_count = 0
        self.last_config = None
        self.checkpointer = None

    def __call__(self, inputs, stream_mode="values", config=None):
        self.call_count += 1
        self.last_config = config
        from langchain_core.messages import AIMessage

        yield {"messages": [AIMessage(content="done")]}


class _RecursiveStream:
    """Fake LangGraph stream that always raises GraphRecursionError."""

    def __init__(self):
        self.call_count = 0
        self.last_config = None
        self.checkpointer = None

    def __call__(self, inputs, stream_mode="values", config=None):
        self.call_count += 1
        self.last_config = config
        raise GraphRecursionError("Recursion limit of 500 reached without hitting a stop condition.")


class TestRecursionLimitHandling(unittest.TestCase):
    def _make_agent(self, **kwargs):
        from langgraph.checkpoint.memory import MemorySaver

        agent = A1.__new__(A1)
        # Minimal attributes needed by go()/go_stream()
        agent.use_tool_retriever = False
        agent.critic_count = 0
        agent.user_task = None
        agent.log = []
        agent._conversation_state = None
        agent.recursion_limit = kwargs.get("recursion_limit", 500)
        agent.checkpointer = MemorySaver()  # same as __init__
        return agent

    def _make_app(self):
        """Fake app with a checkpointer attribute (mirrors compiled LangGraph app)."""
        from langgraph.checkpoint.memory import MemorySaver

        app = types.SimpleNamespace()
        app.stream = _NormalStream()
        app.checkpointer = MemorySaver()
        return app

    def test_default_recursion_limit_is_500(self):
        agent = self._make_agent()
        self.assertEqual(agent.recursion_limit, 500)

    def test_custom_recursion_limit(self):
        agent = self._make_agent(recursion_limit=2000)
        self.assertEqual(agent.recursion_limit, 2000)

    def test_go_returns_summary_on_recursion_error(self):
        agent = self._make_agent()
        agent.user_task = "Find trials for breast cancer"
        app = types.SimpleNamespace()
        app.stream = _RecursiveStream()
        app.checkpointer = None
        agent.app = app

        log, content = agent.go("Find trials for breast cancer")

        # Must not raise; must return a summary, not a raw error
        self.assertIn("not completed", content)
        self.assertIn("Find trials for breast cancer", content)
        self.assertIn("Steps executed", content)
        # The summary is appended to the log
        self.assertTrue(any("not completed" in line for line in log))

    def test_go_stream_yields_summary_on_recursion_error(self):
        agent = self._make_agent()
        agent.user_task = "Analyze this dataset"
        app = types.SimpleNamespace()
        app.stream = _RecursiveStream()
        app.checkpointer = None
        agent.app = app

        outputs = list(agent.go_stream("Analyze this dataset"))

        # Generator must not raise; last yield is the summary
        self.assertGreaterEqual(len(outputs), 1)
        self.assertIn("not completed", outputs[-1]["output"])

    def test_normal_execution_unaffected(self):
        agent = self._make_agent()
        app = self._make_app()
        agent.app = app

        log, content = agent.go("Simple task")

        self.assertEqual(content, "done")
        self.assertFalse(any("not completed" in line for line in log))

    def test_thread_id_is_random_by_default(self):
        """Two calls without thread_id use different thread IDs (no accumulation)."""
        agent = self._make_agent()
        app = self._make_app()
        agent.app = app

        agent.go("first")
        first_id = app.stream.last_config["configurable"]["thread_id"]
        agent.go("second")
        second_id = app.stream.last_config["configurable"]["thread_id"]

        self.assertNotEqual(first_id, second_id)

    def test_default_no_checkpointer_attached(self):
        """Without an explicit thread_id, no checkpointer is attached (no leak)."""
        agent = self._make_agent()
        app = self._make_app()
        agent.app = app
        agent.checkpointer = app.checkpointer  # same as __init__

        agent.go("task")

        self.assertIsNone(app.checkpointer)

    def test_explicit_thread_id_attaches_checkpointer(self):
        """With an explicit thread_id, the checkpointer is attached for multi-turn memory."""
        agent = self._make_agent()
        app = self._make_app()
        agent.app = app
        agent.checkpointer = app.checkpointer

        agent.go("first", thread_id="my-fixed-thread")

        tid = app.stream.last_config["configurable"]["thread_id"]
        self.assertEqual(tid, "my-fixed-thread")
        self.assertIsNotNone(app.checkpointer)

    def test_thread_id_explicit_is_used(self):
        agent = self._make_agent()
        app = self._make_app()
        agent.app = app

        agent.go("first", thread_id="my-fixed-thread")
        tid = app.stream.last_config["configurable"]["thread_id"]

        self.assertEqual(tid, "my-fixed-thread")

    def test_recursion_limit_passed_to_config(self):
        agent = self._make_agent(recursion_limit=1234)
        app = self._make_app()
        agent.app = app

        agent.go("task")

        self.assertEqual(app.stream.last_config["recursion_limit"], 1234)

    def test_summary_trims_long_last_step(self):
        agent = self._make_agent()
        agent.user_task = "long task"
        # Simulate a long execution log
        agent.log = ["x" * 1000]
        summary = agent._summarize_interrupted_execution()
        self.assertIn("...", summary)
        self.assertIn("long task", summary)

    def test_gradio_stream_wrapper_returns_summary_on_recursion(self):
        """The gradio path wraps app.stream; a recursion hit must yield a summary, not crash."""
        agent = self._make_agent()
        agent.user_task = "Analyze this dataset"
        app = types.SimpleNamespace()
        app.stream = _RecursiveStream()
        app.checkpointer = None
        agent.app = app

        # Replicate the gradio _stream_with_summary wrapper
        def _stream_with_summary():
            try:
                for s in agent.app.stream(
                    {"messages": [HumanMessage(content="Analyze this dataset")], "next_step": None},
                    stream_mode="values",
                    config={"recursion_limit": agent.recursion_limit, "configurable": {"thread_id": "x"}},
                ):
                    agent.log.append(pretty_print(s["messages"][-1], printout=False))
                    yield s
            except GraphRecursionError:
                summary = agent._summarize_interrupted_execution()
                yield {"messages": [AIMessage(content=summary)]}

        states = list(_stream_with_summary())

        # Must yield exactly one state (the summary), and it must mention incompletion
        self.assertEqual(len(states), 1)
        self.assertIn("not completed", states[0]["messages"][-1].content)

    @pytest.mark.integration
    def test_real_graph_recursion_returns_summary(self):
        """Integration: build the real A1 graph via configure() and verify that
        a recursion-limit hit returns a summary instead of raising.

        The graph is the real compiled workflow (real generate/execute/routing
        nodes, real tool schemas from read_module2api, real system prompt).
        Only the LLM is a stub: it always returns a <think> tag, which drives
        the real generate node into the generate->generate loop that hits the
        recursion limit (one of the real-world causes of #237).
        """
        import tempfile

        from biomni.env_desc import data_lake_dict, library_content_dict
        from biomni.utils import read_module2api

        class FakeLLM:
            model_name = "fake-llm"

            def invoke(self, messages):
                return AIMessage(content="<think>I am thinking about the problem but never acting...</think>")

        agent = A1.__new__(A1)
        agent.path = tempfile.mkdtemp(prefix="biomni_test_")
        agent.llm = FakeLLM()
        agent.module2api = read_module2api()
        agent.timeout_seconds = 600
        agent.recursion_limit = 5
        agent.use_tool_retriever = False
        agent.data_lake_dict = data_lake_dict
        agent.library_content_dict = library_content_dict

        agent.configure()
        self.assertIsNotNone(agent.app)
        self.assertGreater(len(agent.system_prompt), 1000)

        log, content = agent.go("Please solve this complex biology task")

        self.assertIn("not completed", content)
        self.assertIn("Steps executed", content)
        self.assertIn("Please solve this complex biology task", content)


if __name__ == "__main__":
    unittest.main()
