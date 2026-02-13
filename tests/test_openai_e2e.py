#!/usr/bin/env python
"""
End-to-end integration test for the Biomni agent with OpenAI models.

This test creates a real A1 agent with an OpenAI model, sends it a simple task,
and verifies the full ReAct loop completes: retrieval -> generate -> execute -> result.

Run with: pytest tests/test_openai_e2e.py -v --live -s
Requires OPENAI_API_KEY in environment (loaded from .env).
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.live
class TestOpenAIEndToEnd:
    """Full agent loop with a real OpenAI model and real code execution.

    These tests instantiate A1 with gpt-5-mini (cheap, fast) and run
    tasks that force the agent through generate -> execute -> solution.
    """

    @pytest.fixture(autouse=True)
    def check_openai_key(self):
        # Load .env if present (same as the agent does)
        from dotenv import load_dotenv

        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    @pytest.fixture
    def agent_4o_mini(self, tmp_path):
        """Create a minimal A1 agent with gpt-5-mini, no datalake download."""
        from biomni.agent import A1

        agent = A1(
            path=str(tmp_path / "data"),
            llm="gpt-5-mini",
            use_tool_retriever=False,  # skip retrieval to keep it fast
            expected_data_lake_files=[],  # skip datalake download
            timeout_seconds=120,
        )
        return agent

    @pytest.fixture
    def agent_gpt5(self, tmp_path):
        """Create a minimal A1 agent with gpt-5, no datalake download."""
        from biomni.agent import A1

        try:
            agent = A1(
                path=str(tmp_path / "data"),
                llm="gpt-5",
                use_tool_retriever=False,
                expected_data_lake_files=[],
                timeout_seconds=120,
            )
        except Exception as e:
            pytest.skip(f"gpt-5 not available: {e}")
        return agent

    def test_simple_math_e2e(self, agent_4o_mini):
        """Agent should execute Python code and return a correct math result."""
        log, final = agent_4o_mini.go(
            "Use Python to compute 7 * 13 and tell me the result. "
            "You must execute the code, not just tell me the answer."
        )

        # The log should contain multiple entries (at least generate + execute + solution)
        assert len(log) >= 3, f"Expected at least 3 log entries, got {len(log)}"

        # Verify code was actually executed (an <execute> tag should appear somewhere)
        all_log_text = "\n".join(str(entry) for entry in log)
        assert "<execute>" in all_log_text, "Agent should have produced an <execute> tag"

        # Verify the correct answer appears in the final output
        assert "91" in final, f"Expected 91 in final answer, got: {final[:300]}"

        print(f"\n  Log entries: {len(log)}")
        print(f"  Final answer (first 200 chars): {final[:200]}")

    def test_no_syntax_error_loop(self, agent_4o_mini):
        """Regression test: agent should NOT loop on 'invalid syntax' from markdown fences."""
        log, final = agent_4o_mini.go(
            "Write and run a Python script that creates a list of the first 10 square numbers "
            "and prints them. Execute the code."
        )

        all_log_text = "\n".join(str(entry) for entry in log)

        # Count syntax errors - should be 0 or at most 1 (transient)
        syntax_errors = all_log_text.count("invalid syntax")
        assert syntax_errors <= 1, (
            f"Agent looped on syntax errors ({syntax_errors} occurrences). "
            f"Markdown fence stripping may not be working."
        )

        # Should have produced a solution
        assert "<solution>" in all_log_text or "solution" in all_log_text.lower(), (
            "Agent should have reached a solution"
        )

        # The squares should appear somewhere
        assert "1" in final and "4" in final and "9" in final, (
            f"Expected square numbers in output, got: {final[:300]}"
        )

        print(f"\n  Log entries: {len(log)}")
        print(f"  Syntax errors seen: {syntax_errors}")
        print(f"  Final answer (first 200 chars): {final[:200]}")

    def test_execute_tag_contains_clean_code(self, agent_4o_mini):
        """Verify that code inside <execute> tags has no markdown fences."""
        log, final = agent_4o_mini.go(
            "Execute Python code that prints 'biomni_test_marker_12345'. Just run it."
        )

        all_log_text = "\n".join(str(entry) for entry in log)

        # Find all <execute> blocks
        execute_blocks = re.findall(r"<execute>(.*?)</execute>", all_log_text, re.DOTALL)
        assert len(execute_blocks) >= 1, "Should have at least one execute block"

        for i, block in enumerate(execute_blocks):
            assert not block.strip().startswith("```"), (
                f"Execute block {i} starts with markdown fence: {block[:100]}"
            )
            assert not block.strip().endswith("```"), (
                f"Execute block {i} ends with markdown fence: {block[-100:]}"
            )

        # The marker should have been printed
        assert "biomni_test_marker_12345" in all_log_text, (
            "The test marker should appear in execution output"
        )

        print(f"\n  Execute blocks found: {len(execute_blocks)}")
        print(f"  First block (first 120 chars): {execute_blocks[0][:120]}")

    def test_gpt5_e2e_if_available(self, agent_gpt5):
        """Same test with gpt-5 to verify the Responses API path works end-to-end."""
        log, final = agent_gpt5.go(
            "Use Python to compute 2**10 and tell me the result. Execute the code."
        )

        assert len(log) >= 3, f"Expected at least 3 log entries, got {len(log)}"

        all_log_text = "\n".join(str(entry) for entry in log)
        assert "<execute>" in all_log_text, "Agent should have produced an <execute> tag"
        assert "1024" in final, f"Expected 1024 in final answer, got: {final[:300]}"

        # Verify no syntax error loop
        syntax_errors = all_log_text.count("invalid syntax")
        assert syntax_errors <= 1, f"Syntax error loop detected ({syntax_errors} occurrences)"

        print(f"\n  Log entries: {len(log)}")
        print(f"  Final answer (first 200 chars): {final[:200]}")


@pytest.mark.live
class TestOpenAIWithRetriever:
    """Test with tool retriever enabled — closer to real usage."""

    @pytest.fixture(autouse=True)
    def check_keys(self):
        from dotenv import load_dotenv

        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        # Retriever uses the lite model which may need an Anthropic key
        # depending on config, so check both
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("Need at least one API key for retriever")

    def test_retriever_plus_execute(self, tmp_path):
        """Full loop with retriever: retrieve tools -> generate -> execute -> solution."""
        from biomni.agent import A1

        agent = A1(
            path=str(tmp_path / "data"),
            llm="gpt-5-mini",
            use_tool_retriever=True,
            expected_data_lake_files=[],
            timeout_seconds=180,
        )

        log, final = agent.go(
            "Using Python, calculate the factorial of 10 and tell me the result."
        )

        assert len(log) >= 3, f"Expected at least 3 log entries, got {len(log)}"

        all_log_text = "\n".join(str(entry) for entry in log)
        assert "<execute>" in all_log_text, "Agent should have executed code"
        assert "3628800" in final, f"Expected 3628800 (10!) in final answer, got: {final[:300]}"

        # No syntax error looping
        syntax_errors = all_log_text.count("invalid syntax")
        assert syntax_errors <= 1, f"Syntax error loop detected ({syntax_errors} occurrences)"

        print(f"\n  Log entries: {len(log)}")
        print(f"  Final answer (first 200 chars): {final[:200]}")


@pytest.mark.live
class TestOpenAIWithMCP:
    """Test with OKN-WOBD MCP server — full agent loop including external tool discovery."""

    @pytest.fixture(autouse=True)
    def check_keys(self):
        from dotenv import load_dotenv

        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    @pytest.fixture(autouse=True)
    def check_mcp_available(self):
        """Skip if the OKN-WOBD MCP server source isn't available locally."""
        okn_src = "/Users/bgood/Documents/GitHub/OKN-WOBD/src"
        if not os.path.isdir(okn_src):
            pytest.skip("OKN-WOBD source not found at /Users/bgood/Documents/GitHub/OKN-WOBD/src")
        mcp_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_config.yaml")
        if not os.path.isfile(mcp_config):
            pytest.skip("mcp_config.yaml not found in project root")

    def test_mcp_osteoarthritis_geo_dataset(self, tmp_path):
        """Agent uses OKN-WOBD MCP tools to find a GEO dataset about osteoarthritis.

        This exercises:
          1. MCP server startup and tool discovery
          2. Tool retriever selecting MCP tools
          3. Agent generating code that calls an MCP tool
          4. Code execution returning real results
          5. Agent summarizing findings in a solution
        """
        from biomni.agent import A1

        mcp_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_config.yaml")

        agent = A1(
            path=str(tmp_path / "data"),
            llm="gpt-5",
            use_tool_retriever=True,
            expected_data_lake_files=[],
            timeout_seconds=300,
        )
        agent.add_mcp(config_path=mcp_config)

        log, final = agent.go(
            "Using okn-wobd mcp, find me a geo dataset about osteoarthritis."
        )

        all_log_text = "\n".join(str(entry) for entry in log)

        # Should have executed code (not just answered from knowledge)
        assert "<execute>" in all_log_text, "Agent should have produced an <execute> tag"

        # Should have reached a solution
        assert "<solution>" in all_log_text, "Agent should have reached a solution"

        # The answer should reference GEO datasets (GSE IDs) or osteoarthritis
        final_lower = final.lower()
        has_geo = "gse" in final_lower or "geo" in final_lower
        has_oa = "osteoarthritis" in final_lower or "arthritis" in final_lower
        assert has_geo or has_oa, (
            f"Expected GEO dataset IDs or osteoarthritis mention in answer, got: {final[:500]}"
        )

        # No syntax error loop
        syntax_errors = all_log_text.count("invalid syntax")
        assert syntax_errors <= 1, f"Syntax error loop detected ({syntax_errors} occurrences)"

        print(f"\n  Log entries: {len(log)}")
        print(f"  Final answer (first 500 chars): {final[:500]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--live", "-s"])
