"""Offline regression tests for A1 action-tag conflict handling (#350)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from biomni.agent import A1
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FixedResponseModel(BaseChatModel):
    """Deterministic chat model that replays scripted responses in order."""

    responses: list[Any]
    cursor: int = 0
    seen_messages: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fixed-response-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        response = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])


class NodeTrace(BaseCallbackHandler):
    """Records visited A1 LangGraph node names."""

    def __init__(self):
        self.nodes: list[str] = []

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        name = kwargs.get("name")
        if name in ("generate", "execute", "self_critic"):
            self.nodes.append(name)


def _build_agent(directory: str, responses: list[Any], *, self_critic: bool = False) -> tuple[A1, FixedResponseModel]:
    model = FixedResponseModel(responses=responses)
    agent = A1.__new__(A1)
    agent.path = directory
    agent.module2api = {}
    agent.data_lake_dict = {}
    agent.library_content_dict = {}
    agent.timeout_seconds = 5
    agent.llm = model
    agent.configure(self_critic=self_critic)
    return agent, model


def _invoke_agent(agent: A1, prompt: str = "Write a small text file.", recursion_limit: int = 15):
    trace = NodeTrace()
    final_state = agent.app.invoke(
        {"messages": [HumanMessage(content=prompt)], "next_step": None},
        config={"callbacks": [trace], "recursion_limit": recursion_limit, "configurable": {"thread_id": 42}},
    )
    return final_state, trace


def _write_file_execute_tag(artifact: Path, content: str = "ok") -> str:
    return f"<execute>from pathlib import Path\nPath({str(artifact)!r}).write_text({content!r})</execute>"


class A1ActionTagConflictTest(unittest.TestCase):
    def test_execute_and_solution_in_single_string_requests_correction_and_recovers(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>The file has been written.</solution>"
            agent, model = _build_agent(directory, [f"{execute}\n{solution}", execute, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(), "ok")
            retry_messages = model.seen_messages[1]
            self.assertIsInstance(retry_messages[-1], HumanMessage)
            self.assertIn("Protocol conflict", retry_messages[-1].content)

    def test_execute_and_solution_in_separate_response_blocks_requests_correction(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>The file has been written.</solution>"
            blocks = [
                {"type": "text", "text": execute},
                {"type": "text", "text": solution},
            ]
            agent, model = _build_agent(directory, [blocks, execute, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(), "ok")

    def test_solution_before_execute_requests_correction(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>The file has been written.</solution>"
            agent, model = _build_agent(directory, [f"{solution}\n{execute}", execute, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(), "ok")

    def test_multiple_execute_blocks_do_not_partially_execute(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            first_file = Path(directory) / "first.txt"
            second_file = Path(directory) / "second.txt"
            conflict = f"{_write_file_execute_tag(first_file, '1')}\n{_write_file_execute_tag(second_file, '2')}"
            fixed = _write_file_execute_tag(second_file, "combined")
            solution = "<solution>Done.</solution>"
            agent, model = _build_agent(directory, [conflict, fixed, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertFalse(first_file.exists())
            self.assertTrue(second_file.exists())
            self.assertEqual(second_file.read_text(), "combined")

    def test_multiple_solution_blocks_request_correction(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            conflict = "<solution>Draft</solution>\n<solution>Final</solution>"
            fixed = "<solution>Final</solution>"
            agent, model = _build_agent(directory, [conflict, fixed])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")
            self.assertEqual(final_state["messages"][-1].content, fixed)

    def test_case_insensitive_conflicting_tags_request_correction(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            conflict = (
                f"<EXECUTE>from pathlib import Path\nPath({str(artifact)!r}).write_text('ok')</EXECUTE>\n"
                "<SOLUTION>Done.</SOLUTION>"
            )
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>Done.</solution>"
            agent, model = _build_agent(directory, [conflict, execute, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())

    def test_unclosed_tag_in_dual_tag_conflict_requests_correction(self):
        cases = [
            (
                "empty_execute_with_solution",
                lambda _p: "<execute></execute>\n<solution>The file has been written.</solution>",
            ),
            (
                "unclosed_solution",
                lambda p: f"{_write_file_execute_tag(p, 'ok')}\n<solution>The file has been written.",
            ),
            (
                "unclosed_execute",
                lambda p: (
                    f"<execute>from pathlib import Path\nPath({str(p)!r}).write_text('ok')\n"
                    "<solution>The file has been written.</solution>"
                ),
            ),
            (
                "both_unclosed",
                lambda p: (
                    f"<execute>from pathlib import Path\nPath({str(p)!r}).write_text('ok')\n"
                    "<solution>The file has been written."
                ),
            ),
        ]
        for label, build_conflict in cases:
            with self.subTest(case=label), TemporaryDirectory(prefix="biomni-test-") as directory:
                artifact = Path(directory) / "result.txt"
                execute = _write_file_execute_tag(artifact, "ok")
                solution = "<solution>The file has been written.</solution>"
                agent, model = _build_agent(directory, [build_conflict(artifact), execute, solution])

                final_state, trace = _invoke_agent(agent)

                self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate"])
                self.assertEqual(model.cursor, 3)
                self.assertEqual(final_state["next_step"], "end")
                self.assertTrue(artifact.exists())
                self.assertEqual(artifact.read_text(), "ok")

    def test_two_conflicts_followed_by_valid_response_recover(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>Done.</solution>"
            conflict = f"{execute}\n{solution}"
            agent, model = _build_agent(directory, [conflict, conflict, execute, solution])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "generate", "execute", "generate"])
            self.assertEqual(model.cursor, 4)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(), "ok")

    def test_three_conflicting_responses_raise_value_error_without_executing(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            conflict = f"{_write_file_execute_tag(artifact, 'ok')}\n<solution>Done.</solution>"
            agent, model = _build_agent(directory, [conflict, conflict, conflict])

            trace = NodeTrace()
            with self.assertRaisesRegex(ValueError, "Repeated conflicting action blocks"):
                agent.app.invoke(
                    {"messages": [HumanMessage(content="Write a small text file.")], "next_step": None},
                    config={"callbacks": [trace], "recursion_limit": 10, "configurable": {"thread_id": 42}},
                )

            self.assertEqual(trace.nodes, ["generate", "generate", "generate"])
            self.assertEqual(model.cursor, 3)
            self.assertFalse(artifact.exists())

    def test_conflict_budget_is_cumulative_across_conversation(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            conflict = f"{execute}\n<solution>Done.</solution>"
            agent, model = _build_agent(directory, [conflict, execute, conflict, execute, conflict])

            with self.assertRaisesRegex(ValueError, "Repeated conflicting action blocks"):
                _invoke_agent(agent, recursion_limit=20)

            self.assertEqual(model.cursor, 5)

    def test_self_critic_not_invoked_on_conflicting_response(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>Done.</solution>"
            agent, model = _build_agent(directory, [f"{execute}\n{solution}", execute, solution], self_critic=True)
            agent.critic_count = 0
            agent.user_task = "Write a small text file."

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate", "generate", "execute", "generate", "self_critic"])
            self.assertEqual(model.cursor, 3)
            self.assertEqual(final_state["next_step"], "end")
            self.assertTrue(artifact.exists())

    def test_go_recovers_from_conflicting_response_and_returns_final_solution(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            execute = _write_file_execute_tag(artifact, "ok")
            solution = "<solution>The file has been written.</solution>"
            agent, model = _build_agent(directory, [f"{execute}\n{solution}", execute, solution])
            agent.use_tool_retriever = False

            log, final_content = agent.go("Write a small text file.")

            self.assertEqual(model.cursor, 3)
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(), "ok")
            self.assertEqual(final_content, solution)
            self.assertTrue(any("Protocol conflict" in entry for entry in log))

    def test_solution_mentioning_literal_execute_tag_completes_normally(self):
        cases = [
            ("prose_mention", "<solution>Wrap Python code in the <execute> tag to run it.</solution>"),
            ("nested_example", "<solution>Example: <execute>print(1)</execute></solution>"),
            ("markdown_code_block", "<solution>Use:\n```python\nprint(1)\n```</solution>"),
        ]
        for label, response in cases:
            with self.subTest(case=label), TemporaryDirectory(prefix="biomni-test-") as directory:
                agent, model = _build_agent(directory, [response])

                final_state, trace = _invoke_agent(agent)

                self.assertEqual(trace.nodes, ["generate"])
                self.assertEqual(model.cursor, 1)
                self.assertEqual(final_state["next_step"], "end")
                self.assertEqual(final_state["messages"][-1].content, response)

    def test_execute_containing_literal_solution_string_executes_normally(self):
        cases = [
            ("closed_tag_literal", "<solution>ok</solution>"),
            ("unclosed_tag_literal", "<solution> partial"),
        ]
        for label, payload in cases:
            with self.subTest(case=label), TemporaryDirectory(prefix="biomni-test-") as directory:
                artifact = Path(directory) / "result.txt"
                execute = _write_file_execute_tag(artifact, payload)
                solution = "<solution>Done.</solution>"
                agent, model = _build_agent(directory, [execute, solution])

                final_state, trace = _invoke_agent(agent)

                self.assertEqual(trace.nodes, ["generate", "execute", "generate"])
                self.assertEqual(model.cursor, 2)
                self.assertEqual(final_state["next_step"], "end")
                self.assertTrue(artifact.exists())
                self.assertEqual(artifact.read_text(), payload)

    def test_think_block_mentioning_tags_does_not_conflict_with_single_action(self):
        with TemporaryDirectory(prefix="biomni-test-") as directory:
            response = "<think>I do not need <execute> here; I will return <solution>.</think>\n<solution>42</solution>"
            agent, model = _build_agent(directory, [response])

            final_state, trace = _invoke_agent(agent)

            self.assertEqual(trace.nodes, ["generate"])
            self.assertEqual(model.cursor, 1)
            self.assertEqual(final_state["next_step"], "end")

    def test_baseline_single_action_and_error_paths_preserved(self):
        with self.subTest(case="execute_then_solution"), TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            agent, model = _build_agent(directory, [_write_file_execute_tag(artifact, "ok"), "<solution>ok</solution>"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate", "execute", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")
            self.assertEqual(artifact.read_text(), "ok")

        with self.subTest(case="solution_only"), TemporaryDirectory(prefix="biomni-test-") as directory:
            agent, model = _build_agent(directory, ["<solution>42</solution>"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate"])
            self.assertEqual(model.cursor, 1)
            self.assertEqual(final_state["next_step"], "end")

        with self.subTest(case="unclosed_single_execute"), TemporaryDirectory(prefix="biomni-test-") as directory:
            artifact = Path(directory) / "result.txt"
            unclosed = f"<execute>from pathlib import Path\nPath({str(artifact)!r}).write_text('ok')"
            agent, model = _build_agent(directory, [unclosed, "<solution>ok</solution>"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate", "execute", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")
            self.assertEqual(artifact.read_text(), "ok")

        with self.subTest(case="unclosed_single_solution"), TemporaryDirectory(prefix="biomni-test-") as directory:
            agent, model = _build_agent(directory, ["<solution>42"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate"])
            self.assertEqual(model.cursor, 1)
            self.assertEqual(final_state["next_step"], "end")
            self.assertEqual(final_state["messages"][-1].content, "<solution>42</solution>")

        with self.subTest(case="execution_error_observation"), TemporaryDirectory(prefix="biomni-test-") as directory:
            agent, model = _build_agent(
                directory,
                ["<execute>raise RuntimeError('boom')</execute>", "<solution>Handled error.</solution>"],
            )
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate", "execute", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")
            observations = [m.content for m in final_state["messages"] if "<observation>" in m.content]
            self.assertEqual(len(observations), 1)
            self.assertIn("boom", observations[0])

        with self.subTest(case="think_only_loops_to_generate"), TemporaryDirectory(prefix="biomni-test-") as directory:
            agent, model = _build_agent(directory, ["<think>Planning step</think>", "<solution>Done.</solution>"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")

        with (
            self.subTest(case="missing_tags_keeps_existing_recovery"),
            TemporaryDirectory(prefix="biomni-test-") as directory,
        ):
            agent, model = _build_agent(directory, ["Plain response with no tags.", "<solution>Done.</solution>"])
            final_state, trace = _invoke_agent(agent)
            self.assertEqual(trace.nodes, ["generate", "generate"])
            self.assertEqual(model.cursor, 2)
            self.assertEqual(final_state["next_step"], "end")
            self.assertIn("no tags", model.seen_messages[1][-1].content)


if __name__ == "__main__":
    unittest.main()
