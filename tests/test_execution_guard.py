import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from biomni.agent import A1
from biomni.agent.execution_guard import (
    blocked_execution_message,
    is_execution_failure,
    record_execution_result,
    repeated_failure_guidance,
    should_block_execution,
)
from langchain_core.messages import AIMessage, HumanMessage

CODE = 'raise RuntimeError("boom")'
ERROR = "Error: boom"


class ExecutionGuardTests(unittest.TestCase):
    def test_identical_failure_is_allowed_once_then_blocked(self):
        state = record_execution_result(None, CODE, ERROR)

        self.assertIsNotNone(state)
        self.assertEqual(state["count"], 1)
        self.assertFalse(should_block_execution(state, CODE))

        state = record_execution_result(state, CODE, ERROR)

        self.assertIsNotNone(state)
        self.assertEqual(state["count"], 2)
        self.assertTrue(should_block_execution(state, CODE))

    def test_success_clears_previous_failure(self):
        state = record_execution_result(None, CODE, ERROR)

        self.assertIsNone(record_execution_result(state, CODE, "completed"))

    def test_changed_code_resets_failure_count(self):
        state = record_execution_result(None, CODE, ERROR)
        state = record_execution_result(state, 'raise RuntimeError("different")', ERROR)

        self.assertIsNotNone(state)
        self.assertEqual(state["count"], 1)
        self.assertFalse(should_block_execution(state, CODE))

    def test_changed_failure_resets_failure_count(self):
        state = record_execution_result(None, CODE, ERROR)
        state = record_execution_result(state, CODE, "Error: a different failure")

        self.assertIsNotNone(state)
        self.assertEqual(state["count"], 1)
        self.assertFalse(should_block_execution(state, CODE))

    def test_line_ending_and_trailing_whitespace_differences_are_ignored(self):
        state = record_execution_result(None, "line_one()  \r\nline_two()", "Error: boom\r\n")
        state = record_execution_result(state, "line_one()\nline_two()  ", "Error: boom\n")

        self.assertIsNotNone(state)
        self.assertEqual(state["count"], 2)
        self.assertTrue(should_block_execution(state, "line_one()\nline_two()"))

    def test_non_error_text_does_not_count_as_failure(self):
        self.assertFalse(is_execution_failure("Analysis completed with no error"))
        self.assertIsNone(record_execution_result(None, CODE, "Analysis completed with no error"))

    def test_standard_executor_failures_are_detected(self):
        self.assertTrue(is_execution_failure("Error: invalid syntax"))
        self.assertTrue(is_execution_failure("ERROR: Code execution timed out"))
        self.assertTrue(is_execution_failure("Traceback (most recent call last):"))

    def test_recovery_messages_tell_the_model_to_change_strategy(self):
        self.assertIn("Do not repeat the same code", repeated_failure_guidance(2))
        self.assertIn("Execution skipped", blocked_execution_message())
        self.assertIn("different strategy", blocked_execution_message())

    def test_a1_skips_third_identical_failed_execution(self):
        class StubLLM:
            def __init__(self):
                self.responses = iter(
                    [
                        AIMessage(content=f"<execute>{CODE}</execute>"),
                        AIMessage(content=f"<execute>{CODE}</execute>"),
                        AIMessage(content=f"<execute>{CODE}</execute>"),
                        AIMessage(content="<solution>Recovered safely.</solution>"),
                    ]
                )

            def invoke(self, _messages):
                return next(self.responses)

        with TemporaryDirectory() as temp_dir:
            agent = A1.__new__(A1)
            agent.path = str(Path(temp_dir))
            agent.data_lake_dict = {}
            agent.library_content_dict = {}
            agent.module2api = {}
            agent.know_how_loader = SimpleNamespace(documents={})
            agent.llm = StubLLM()
            agent.timeout_seconds = 1
            agent._generate_system_prompt = lambda **_kwargs: "Test system prompt"
            agent._clear_execution_plots = lambda: None
            agent._inject_custom_functions_to_repl = lambda: None
            agent.configure()

            inputs = {
                "messages": [HumanMessage(content="Run failing code")],
                "next_step": None,
                "execution_failure": None,
            }
            config = {"recursion_limit": 20, "configurable": {"thread_id": "loop-guard-test"}}

            with patch("biomni.agent.a1.run_with_timeout", return_value=ERROR) as run_code:
                states = list(agent.app.stream(inputs, stream_mode="values", config=config))

        self.assertEqual(run_code.call_count, 2)
        final_messages = states[-1]["messages"]
        self.assertTrue(any("Execution skipped" in str(message.content) for message in final_messages))
        self.assertIn("<solution>Recovered safely.</solution>", str(final_messages[-1].content))


if __name__ == "__main__":
    unittest.main()
