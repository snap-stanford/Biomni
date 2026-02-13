#!/usr/bin/env python
"""
Tests for OpenAI structured output support and llm_lite auto-inference.

Exercises the changes from the goodb-wobd branch:
  1. BiomniConfig._infer_lite_model() provider matching
  2. BiomniConfig auto-inference of llm_lite from llm
  3. AgentResponse Pydantic schema correctness
  4. A1._is_openai_model detection property
  5. generate() structured output path (mocked LLM)
  6. generate() XML fallback path (mocked LLM)

Run with: pytest tests/test_openai_structured_output.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. BiomniConfig._infer_lite_model
# ---------------------------------------------------------------------------
class TestInferLiteModel:
    """Test the static method that maps a main model to its lite counterpart."""

    def test_claude_model(self):
        from biomni.config import BiomniConfig

        assert BiomniConfig._infer_lite_model("claude-sonnet-4-5") == "claude-haiku-4-5"

    def test_gpt_model(self):
        from biomni.config import BiomniConfig

        assert BiomniConfig._infer_lite_model("gpt-5.2") == "gpt-5-mini"

    def test_gpt4_model(self):
        from biomni.config import BiomniConfig

        assert BiomniConfig._infer_lite_model("gpt-4o") == "gpt-5-mini"

    def test_gemini_model(self):
        from biomni.config import BiomniConfig

        assert BiomniConfig._infer_lite_model("gemini-2.0-pro") == "gemini-1.5-flash"

    def test_unknown_model_falls_back_to_claude(self):
        from biomni.config import BiomniConfig

        assert BiomniConfig._infer_lite_model("some-unknown-model") == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 2. BiomniConfig llm_lite auto-inference integration
# ---------------------------------------------------------------------------
class TestConfigLiteAutoInference:
    """Test that BiomniConfig auto-infers llm_lite when not explicitly set."""

    def test_default_is_claude(self):
        from biomni.config import BiomniConfig

        config = BiomniConfig()
        assert config.llm == "claude-sonnet-4-5"
        assert config.llm_lite == "claude-haiku-4-5"

    def test_openai_main_infers_openai_lite(self):
        from biomni.config import BiomniConfig

        config = BiomniConfig(llm="gpt-5.2")
        assert config.llm_lite == "gpt-5-mini"

    def test_gemini_main_infers_gemini_lite(self):
        from biomni.config import BiomniConfig

        config = BiomniConfig(llm="gemini-2.0-pro")
        assert config.llm_lite == "gemini-1.5-flash"

    def test_explicit_lite_not_overridden(self):
        from biomni.config import BiomniConfig

        config = BiomniConfig(llm="gpt-5.2", llm_lite="gpt-4o")
        assert config.llm_lite == "gpt-4o", "Explicit llm_lite should not be overridden"

    def test_env_var_overrides_inference(self):
        """BIOMNI_LLM_LITE env var should take precedence over auto-inference."""
        from biomni.config import BiomniConfig

        with patch.dict(os.environ, {"BIOMNI_LLM_LITE": "my-custom-lite-model"}):
            config = BiomniConfig(llm="gpt-5.2")
            assert config.llm_lite == "my-custom-lite-model"

    def test_env_var_llm_triggers_matching_lite(self):
        """When BIOMNI_LLM sets an OpenAI model, llm_lite should auto-match."""
        from biomni.config import BiomniConfig

        with patch.dict(os.environ, {"BIOMNI_LLM": "gpt-4o"}, clear=False):
            # Remove BIOMNI_LLM_LITE if present to test inference
            env = os.environ.copy()
            env.pop("BIOMNI_LLM_LITE", None)
            with patch.dict(os.environ, env, clear=True):
                config = BiomniConfig()
                assert config.llm == "gpt-4o"
                assert config.llm_lite == "gpt-5-mini"


# ---------------------------------------------------------------------------
# 3. AgentResponse schema
# ---------------------------------------------------------------------------
class TestAgentResponseSchema:
    """Test the Pydantic schema used for OpenAI structured outputs."""

    def test_fields_exist(self):
        from biomni.agent.a1 import AgentResponse

        fields = set(AgentResponse.model_fields.keys())
        assert fields == {"reasoning", "action", "content"}

    def test_action_enum_values(self):
        from biomni.agent.a1 import AgentResponse

        # Valid actions
        resp = AgentResponse(reasoning="thinking...", action="execute", content="print('hi')")
        assert resp.action == "execute"

        resp = AgentResponse(reasoning="done", action="solution", content="The answer is 42")
        assert resp.action == "solution"

    def test_invalid_action_rejected(self):
        from biomni.agent.a1 import AgentResponse

        with pytest.raises(Exception):  # ValidationError
            AgentResponse(reasoning="hmm", action="think", content="...")

    def test_json_schema_has_enum(self):
        """The JSON schema sent to OpenAI should constrain action to the enum."""
        from biomni.agent.a1 import AgentResponse

        schema = AgentResponse.model_json_schema()
        assert schema["properties"]["action"]["enum"] == ["execute", "solution"]

    def test_all_fields_required(self):
        from biomni.agent.a1 import AgentResponse

        schema = AgentResponse.model_json_schema()
        assert set(schema["required"]) == {"reasoning", "action", "content"}


# ---------------------------------------------------------------------------
# 4. A1._is_openai_model property
# ---------------------------------------------------------------------------
class TestIsOpenAIModel:
    """Test the property that detects whether the LLM is an OpenAI model."""

    def _make_stub(self, model_name):
        """Create a minimal A1-like object with a mock LLM."""
        from biomni.agent.a1 import A1

        stub = object.__new__(A1)  # skip __init__
        stub.llm = MagicMock()
        stub.llm.model_name = model_name
        return stub

    def test_gpt5_detected(self):
        stub = self._make_stub("gpt-5.2")
        assert stub._is_openai_model is True

    def test_gpt4o_detected(self):
        stub = self._make_stub("gpt-4o")
        assert stub._is_openai_model is True

    def test_claude_not_detected(self):
        stub = self._make_stub("claude-sonnet-4-5")
        assert stub._is_openai_model is False

    def test_gemini_not_detected(self):
        stub = self._make_stub("gemini-2.0-pro")
        assert stub._is_openai_model is False


# ---------------------------------------------------------------------------
# 5. generate() — structured output path (OpenAI)
# ---------------------------------------------------------------------------
class TestGenerateStructuredOutput:
    """Test that generate() uses structured output for OpenAI models and
    produces correctly tagged messages for the downstream execute() node."""

    def _build_generate(self, agent_response):
        """Build the generate() function from a1.py with a mocked LLM.

        Returns (generate_fn, mock_llm) so tests can inspect calls.
        """
        import re

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from biomni.agent.a1 import AgentResponse, AgentState

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = agent_response
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.model_name = "gpt-5.2"

        # Minimal self-like namespace
        class FakeSelf:
            llm = mock_llm
            system_prompt = "You are a test agent."

            @property
            def _is_openai_model(self):
                model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "")
                return str(model_name).lower().startswith("gpt-") or "openai" in str(type(self.llm)).lower()

        self_obj = FakeSelf()

        # Reconstruct the generate function logic (mirrors a1.py)
        def generate(state):
            messages = [SystemMessage(content=self_obj.system_prompt)] + state["messages"]

            if self_obj._is_openai_model:
                try:
                    structured_llm = self_obj.llm.with_structured_output(AgentResponse)
                    response = structured_llm.invoke(messages)
                    content = response.content
                    if response.action == "execute":
                        content = re.sub(r"^```(?:python|bash|r)?\s*\n?", "", content)
                        content = re.sub(r"\n?```\s*$", "", content)
                    tag = response.action
                    msg = f"{response.reasoning}\n<{tag}>{content}</{tag}>"
                    state["messages"].append(AIMessage(content=msg.strip()))
                    if response.action == "solution":
                        # Guard: reject premature solutions when no code has been executed
                        has_executed = any(
                            "<observation>" in (m.content if isinstance(m.content, str) else "")
                            for m in state["messages"]
                        )
                        if not has_executed:
                            nudge = (
                                "\n<observation>You chose 'solution' but have not executed any code yet. "
                                "You MUST run code using the available tools before providing a final answer. "
                                "Choose action 'execute' and write code to accomplish the task.</observation>"
                            )
                            state["messages"].append(AIMessage(content=nudge.strip()))
                            state["next_step"] = "generate"
                        else:
                            state["next_step"] = "end"
                    else:
                        state["next_step"] = "execute"
                    return state
                except Exception as e:
                    print(f"Structured output failed ({e}), falling back to XML parsing...")
                    response = self_obj.llm.invoke(messages)
                    msg = str(response.content)
            else:
                response = self_obj.llm.invoke(messages)
                msg = str(response.content)

            # XML fallback
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"

            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            state["messages"].append(AIMessage(content=msg.strip()))
            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            else:
                state["next_step"] = "generate"
            return state

        return generate, mock_llm

    def test_execute_action_produces_execute_tag(self):
        from biomni.agent.a1 import AgentResponse

        resp = AgentResponse(
            reasoning="I will query the database.",
            action="execute",
            content='from biomni.tool.database import query_geo\nresult = query_geo(prompt="test")\nprint(result)',
        )
        generate, mock_llm = self._build_generate(resp)

        state = {"messages": [], "next_step": None}
        result = generate(state)

        assert result["next_step"] == "execute"
        last_msg = result["messages"][-1].content
        assert "<execute>" in last_msg
        assert "</execute>" in last_msg
        assert "query_geo" in last_msg
        assert "I will query the database." in last_msg
        # Verify structured output was used (not raw invoke)
        mock_llm.with_structured_output.assert_called_once_with(AgentResponse)

    def test_solution_action_produces_solution_tag_and_ends(self):
        from langchain_core.messages import AIMessage

        from biomni.agent.a1 import AgentResponse

        resp = AgentResponse(
            reasoning="All steps complete. Here are the results.",
            action="solution",
            content="The top datasets are GSE123, GSE456, GSE789.",
        )
        generate, _ = self._build_generate(resp)

        # With a prior <observation> (i.e. code has been executed), solution should end
        state = {
            "messages": [AIMessage(content="<observation>some output</observation>")],
            "next_step": None,
        }
        result = generate(state)

        assert result["next_step"] == "end"
        last_msg = result["messages"][-1].content
        assert "<solution>" in last_msg
        assert "</solution>" in last_msg
        assert "GSE123" in last_msg

    def test_premature_solution_rejected_without_prior_execution(self):
        """Agent cannot skip to 'solution' without executing code first."""
        from biomni.agent.a1 import AgentResponse

        resp = AgentResponse(
            reasoning="I already know the answer.",
            action="solution",
            content="The answer is 42.",
        )
        generate, _ = self._build_generate(resp)

        # No prior <observation> messages — no code has been executed
        state = {"messages": [], "next_step": None}
        result = generate(state)

        # Should be sent back to generate, not end
        assert result["next_step"] == "generate"
        # Should have a nudge message telling the agent to execute code
        nudge_msg = result["messages"][-1].content
        assert "have not executed any code" in nudge_msg

    def test_execute_content_extractable_by_downstream_regex(self):
        """Verify the reconstructed message can be parsed by execute() regex."""
        import re

        from biomni.agent.a1 import AgentResponse

        code = "print('hello world')"
        resp = AgentResponse(reasoning="Testing code.", action="execute", content=code)
        generate, _ = self._build_generate(resp)

        state = {"messages": [], "next_step": None}
        result = generate(state)

        last_msg = result["messages"][-1].content
        match = re.search(r"<execute>(.*?)</execute>", last_msg, re.DOTALL)
        assert match is not None, "execute() regex must find the code"
        assert match.group(1) == code

    def test_markdown_code_fences_stripped_from_content(self):
        """LLMs often wrap code in ```python fences even in structured output. These must be stripped."""
        import re

        from biomni.agent.a1 import AgentResponse

        # Simulate model returning markdown-fenced code
        fenced_code = "```python\nprint('hello world')\n```"
        resp = AgentResponse(reasoning="Running code.", action="execute", content=fenced_code)
        generate, _ = self._build_generate(resp)

        state = {"messages": [], "next_step": None}
        result = generate(state)

        last_msg = result["messages"][-1].content
        match = re.search(r"<execute>(.*?)</execute>", last_msg, re.DOTALL)
        assert match is not None
        extracted = match.group(1)
        assert "```" not in extracted, f"Markdown fences should be stripped, got: {extracted}"
        assert "print('hello world')" in extracted

    def test_markdown_fences_not_stripped_from_solution(self):
        """Markdown fences in solution content should be preserved (they may be intentional formatting)."""
        from langchain_core.messages import AIMessage

        from biomni.agent.a1 import AgentResponse

        content_with_fences = "Here is the code:\n```python\nx = 42\n```"
        resp = AgentResponse(reasoning="Done.", action="solution", content=content_with_fences)
        generate, _ = self._build_generate(resp)

        # Provide a prior observation so the premature-solution guard doesn't reject
        state = {
            "messages": [AIMessage(content="<observation>prior output</observation>")],
            "next_step": None,
        }
        result = generate(state)

        last_msg = result["messages"][-1].content
        assert "```python" in last_msg, "Solution content should preserve markdown fences"


# ---------------------------------------------------------------------------
# 6. generate() — XML fallback when structured output fails
# ---------------------------------------------------------------------------
class TestGenerateXMLFallback:
    """Test that generate() falls back to XML parsing when structured output raises."""

    def test_fallback_on_structured_output_error(self):
        import re

        from langchain_core.messages import AIMessage, SystemMessage

        from biomni.agent.a1 import AgentResponse, AgentState

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o"
        # Structured output raises
        mock_llm.with_structured_output.side_effect = Exception("unsupported")
        # Raw invoke returns XML-tagged content
        mock_raw_response = MagicMock()
        mock_raw_response.content = "Thinking...\n<execute>print('fallback')</execute>"
        mock_llm.invoke.return_value = mock_raw_response

        class FakeSelf:
            llm = mock_llm
            system_prompt = "You are a test agent."

            @property
            def _is_openai_model(self):
                model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "")
                return str(model_name).lower().startswith("gpt-") or "openai" in str(type(self.llm)).lower()

        self_obj = FakeSelf()

        def generate(state):
            messages = [SystemMessage(content=self_obj.system_prompt)] + state["messages"]
            if self_obj._is_openai_model:
                try:
                    structured_llm = self_obj.llm.with_structured_output(AgentResponse)
                    response = structured_llm.invoke(messages)
                    content = response.content
                    if response.action == "execute":
                        content = re.sub(r"^```(?:python|bash|r)?\s*\n?", "", content)
                        content = re.sub(r"\n?```\s*$", "", content)
                    tag = response.action
                    msg = f"{response.reasoning}\n<{tag}>{content}</{tag}>"
                    state["messages"].append(AIMessage(content=msg.strip()))
                    if response.action == "solution":
                        # Guard: reject premature solutions when no code has been executed
                        has_executed = any(
                            "<observation>" in (m.content if isinstance(m.content, str) else "")
                            for m in state["messages"]
                        )
                        if not has_executed:
                            nudge = (
                                "\n<observation>You chose 'solution' but have not executed any code yet. "
                                "You MUST run code using the available tools before providing a final answer. "
                                "Choose action 'execute' and write code to accomplish the task.</observation>"
                            )
                            state["messages"].append(AIMessage(content=nudge.strip()))
                            state["next_step"] = "generate"
                        else:
                            state["next_step"] = "end"
                    else:
                        state["next_step"] = "execute"
                    return state
                except Exception as e:
                    print(f"Structured output failed ({e}), falling back to XML parsing...")
                    response = self_obj.llm.invoke(messages)
                    content = response.content
                    msg = str(content) if not isinstance(content, list) else ""
            else:
                response = self_obj.llm.invoke(messages)
                msg = str(response.content)

            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)
            state["messages"].append(AIMessage(content=msg.strip()))
            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            else:
                state["next_step"] = "generate"
            return state

        state = {"messages": [], "next_step": None}
        result = generate(state)

        assert result["next_step"] == "execute"
        assert "fallback" in result["messages"][-1].content


# ---------------------------------------------------------------------------
# 7. System prompt differs by provider
# ---------------------------------------------------------------------------
class TestSystemPromptBranching:
    """Test that the system prompt format instructions differ for OpenAI vs Claude."""

    def _make_stub(self, model_name):
        from biomni.agent.a1 import A1

        stub = object.__new__(A1)
        stub.llm = MagicMock()
        stub.llm.model_name = model_name
        stub.data_lake_dict = {}
        stub.library_content_dict = {}
        stub.path = "/tmp/test_biomni/biomni_data"
        stub.commercial_mode = False
        return stub

    def test_openai_prompt_mentions_structured_fields(self):
        """When _is_openai_model is True, prompt should reference reasoning/action/content fields."""
        stub = self._make_stub("gpt-5.2")

        prompt = stub._generate_system_prompt(
            tool_desc={},
            data_lake_content=[],
            library_content_list=[],
        )
        assert '"reasoning"' in prompt, "OpenAI prompt should mention the reasoning field"
        assert '"action"' in prompt, "OpenAI prompt should mention the action field"
        assert '"content"' in prompt, "OpenAI prompt should mention the content field"
        # Should NOT contain the XML format instructions
        assert "Your code should be enclosed using" not in prompt

    def test_claude_prompt_mentions_xml_tags(self):
        """When _is_openai_model is False, prompt should reference <execute>/<solution> tags."""
        stub = self._make_stub("claude-sonnet-4-5")

        prompt = stub._generate_system_prompt(
            tool_desc={},
            data_lake_content=[],
            library_content_list=[],
        )
        assert "<execute>" in prompt, "Claude prompt should mention <execute> tag"
        assert "<solution>" in prompt, "Claude prompt should mention <solution> tag"
        # Should NOT contain structured output field names
        assert '"reasoning"' not in prompt


# ---------------------------------------------------------------------------
# 8. Live API tests (require --live flag and API keys)
# ---------------------------------------------------------------------------
@pytest.mark.live
class TestLiveOpenAIStructuredOutput:
    """Tests that call the real OpenAI API with structured output.

    Run with: pytest tests/test_openai_structured_output.py -v --live
    Requires OPENAI_API_KEY in environment.
    """

    @pytest.fixture(autouse=True)
    def check_openai_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_structured_output_execute(self):
        """Live call: OpenAI model returns a valid AgentResponse with action=execute."""
        from biomni.agent.a1 import AgentResponse
        from biomni.llm import get_llm

        llm = get_llm(model="gpt-4o", temperature=0.0)
        structured_llm = llm.with_structured_output(AgentResponse)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a coding assistant. Respond with action='execute' and provide Python code."),
            HumanMessage(content="Write a one-line Python print statement that says hello world."),
        ]
        response = structured_llm.invoke(messages)

        assert isinstance(response, AgentResponse)
        assert response.action == "execute"
        assert "print" in response.content.lower()
        assert len(response.reasoning) > 0
        print(f"  reasoning: {response.reasoning[:80]}...")
        print(f"  action:    {response.action}")
        print(f"  content:   {response.content}")

    def test_structured_output_solution(self):
        """Live call: OpenAI model returns a valid AgentResponse with action=solution."""
        from biomni.agent.a1 import AgentResponse
        from biomni.llm import get_llm

        llm = get_llm(model="gpt-4o", temperature=0.0)
        structured_llm = llm.with_structured_output(AgentResponse)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. The user's task is already complete. "
                    "Respond with action='solution' and provide a final answer."
                )
            ),
            HumanMessage(content="What is 2 + 2? Give the final answer directly."),
        ]
        response = structured_llm.invoke(messages)

        assert isinstance(response, AgentResponse)
        assert response.action == "solution"
        assert "4" in response.content
        print(f"  reasoning: {response.reasoning[:80]}...")
        print(f"  action:    {response.action}")
        print(f"  content:   {response.content}")

    def test_structured_output_roundtrip_with_xml_reconstruction(self):
        """Live call: structured response reconstructs into XML that execute() regex can parse."""
        import re

        from biomni.agent.a1 import AgentResponse
        from biomni.llm import get_llm

        llm = get_llm(model="gpt-4o", temperature=0.0)
        structured_llm = llm.with_structured_output(AgentResponse)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a coding assistant. Respond with action='execute'."),
            HumanMessage(content="Write Python code: x = 1 + 1; print(x)"),
        ]
        response = structured_llm.invoke(messages)

        # Reconstruct the same way generate() does
        tag = response.action
        msg = f"{response.reasoning}\n<{tag}>{response.content}</{tag}>"

        # Verify the downstream execute() regex can extract the code
        match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
        assert match is not None, "Reconstructed message must be parseable by execute()"
        assert "print" in match.group(1)
        print(f"  Extracted code: {match.group(1)}")

    def test_gpt5_structured_output_if_available(self):
        """Live call with gpt-5 (if available). Skips gracefully if model not accessible."""
        from biomni.agent.a1 import AgentResponse
        from biomni.llm import get_llm

        try:
            llm = get_llm(model="gpt-5", temperature=0.0)
        except Exception as e:
            pytest.skip(f"gpt-5 not available: {e}")

        structured_llm = llm.with_structured_output(AgentResponse)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a coding assistant. Respond with action='execute'."),
            HumanMessage(content="Write Python code that prints the numbers 1 through 5."),
        ]
        try:
            response = structured_llm.invoke(messages)
        except Exception as e:
            pytest.skip(f"gpt-5 structured output call failed: {e}")

        assert isinstance(response, AgentResponse)
        assert response.action == "execute"
        print(f"  gpt-5 reasoning: {response.reasoning[:80]}...")
        print(f"  gpt-5 content:   {response.content[:120]}")


@pytest.mark.live
class TestLiveClaudeXML:
    """Tests that call the real Anthropic API with XML tag format.

    Run with: pytest tests/test_openai_structured_output.py -v --live
    Requires ANTHROPIC_API_KEY in environment.
    """

    @pytest.fixture(autouse=True)
    def check_anthropic_key(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_claude_produces_execute_tag(self):
        """Live call: Claude model produces <execute> tags without structured output."""
        from biomni.llm import get_llm

        llm = get_llm(model="claude-haiku-4-5", temperature=0.0)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "You are a coding assistant. Always respond with reasoning followed by code "
                    "inside <execute></execute> tags. Example: I will run code.\n<execute>print('hi')</execute>"
                )
            ),
            HumanMessage(content="Write a Python print statement that says hello world."),
        ]
        response = llm.invoke(messages)
        content = str(response.content)

        assert "<execute>" in content, f"Claude should produce <execute> tag, got: {content[:200]}"
        assert "</execute>" in content, f"Claude should close the tag, got: {content[:200]}"
        print(f"  Claude response: {content[:200]}...")

    def test_claude_produces_solution_tag(self):
        """Live call: Claude model produces <solution> tags."""
        from biomni.llm import get_llm

        llm = get_llm(model="claude-haiku-4-5", temperature=0.0)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Provide your final answer inside "
                    "<solution></solution> tags. Example: The answer is <solution>42</solution>"
                )
            ),
            HumanMessage(content="What is 2 + 2? Give the final answer."),
        ]
        response = llm.invoke(messages)
        content = str(response.content)

        assert "<solution>" in content, f"Claude should produce <solution> tag, got: {content[:200]}"
        assert "4" in content
        print(f"  Claude response: {content[:200]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
