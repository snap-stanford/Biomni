#!/usr/bin/env python3
"""Comprehensive test script for POPPERAgent with verbose output"""

import os
import sys

sys.path.append("/dfs/user/kexinh/biomni")

from biomni.agent.popper_agent import create_popper_agent


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def test_popper():
    """Test the POPPERAgent with verbose output"""

    print_section("POPPER AGENT TEST")

    # Create the agent
    print("\n1. Creating POPPERAgent...")
    agent = create_popper_agent(
        llm="claude-3-5-sonnet-20241022",
        data_path=None,  # Will use biomni data lake
        use_only_database_tools=True,
        max_num_of_tests=2,  # Limit to 2 tests for demo
        time_limit=5,  # 5 minutes per test
        max_retry=3,  # Limit retries for demo
        domain="biology",
    )
    print("✓ Agent created successfully!")

    # Show agent components
    print_section("AGENT COMPONENTS")

    # Show available tools
    print_section("AVAILABLE TOOLS")
    print(f"Total tools loaded: {len(agent.tools)}")
    print("\nDatabase query tools available:")
    for i, tool in enumerate(agent.tools[:10]):  # Show first 10
        print(f"  {i + 1}. {tool.name}: {tool.description[:80]}...")
    if len(agent.tools) > 10:
        print(f"  ... and {len(agent.tools) - 10} more tools")

    # Show data loader info
    print_section("DATA LOADER INFORMATION")
    if agent.data_loader:
        print("Data loader configured: Yes")
        try:
            data_desc = agent.data_loader.get_data_description()
            print(f"Data description length: {len(data_desc)} characters")
            print(f"First 200 chars of data description:\n{data_desc[:200]}...")
        except Exception as e:
            print(f"Could not get data description: {e}")
    else:
        print("Data loader configured: No")

    # Show test proposal info
    print_section("TEST PROPOSAL AGENT")
    if hasattr(agent, "test_proposal_agent"):
        print("Test proposal agent: Configured")
        if hasattr(agent.test_proposal_agent, "generate_prompt"):
            try:
                sample_prompt = agent.test_proposal_agent.generate_prompt("Test hypothesis")
                print(f"Proposal prompt template length: {len(sample_prompt)} characters")
            except Exception as e:
                print(f"Could not get proposal prompt: {e}")

    # Show A1 agent configuration
    print_section("A1 AGENT CONFIGURATION")
    if hasattr(agent, "test_coding_agent") and hasattr(agent.test_coding_agent, "a1_agent"):
        a1_agent = agent.test_coding_agent.a1_agent
        print(f"A1 LLM: {agent.a1_llm}")
        print(f"A1 timeout: {a1_agent.timeout_seconds} seconds")
        print(f"A1 use_tool_retriever: {a1_agent.use_tool_retriever}")

        # Show A1 tools
        if hasattr(a1_agent, "module2api"):
            print("\nA1 available modules:")
            for module, tools in a1_agent.module2api.items():
                print(f"  {module}: {len(tools)} tools")
                for tool in tools[:3]:  # Show first 3 tools
                    print(f"    - {tool.get('name', 'unnamed')}")
                if len(tools) > 3:
                    print(f"    ... and {len(tools) - 3} more")

    # Test with a hypothesis
    print_section("HYPOTHESIS TESTING")

    # Simple hypothesis that should work with database queries
    hypothesis = "The TP53 gene is associated with multiple cancer types"

    print(f"Hypothesis: {hypothesis}")
    print("\nRunning POPPER sequential falsification testing...")
    print("(This may take a few minutes)\n")

    # Capture more detailed logs
    agent.test_coding_agent.verbose if hasattr(agent, "test_coding_agent") else True

    try:
        # Run the test
        log, summary, result = agent.go(hypothesis)

        # Display results
        print_section("TEST RESULTS")

        # Summary
        print("Summary:")
        print("-" * 40)
        print(summary[:500] + "..." if len(summary) > 500 else summary)

        # Result
        print("\nFinal Result:")
        print("-" * 40)
        print(f"Conclusion: {result.get('conclusion', 'Unknown')}")
        print(f"Action to user: {result.get('action_to_user', 'None specified')}")

        # Detailed logs
        print_section("DETAILED LOGS")

        # Designer logs
        if "designer" in log and log["designer"]:
            print(f"\nDesigner logs ({len(log['designer'])} entries):")
            for i, entry in enumerate(log["designer"][:3]):  # Show first 3
                print(f"\n  Entry {i + 1}:")
                print(f"    {str(entry)[:200]}...")

        # Executor logs
        if "executor" in log and log["executor"]:
            print(f"\nExecutor logs ({len(log['executor'])} entries):")
            for i, entry in enumerate(log["executor"][:3]):  # Show first 3
                print(f"\n  Entry {i + 1}:")
                print(f"    {str(entry)[:200]}...")

        # Test Critic logs
        if "test_critic" in log and log["test_critic"]:
            print(f"\nTest Critic logs ({len(log['test_critic'])} entries):")
            for i, entry in enumerate(log["test_critic"][:3]):  # Show first 3
                print(f"\n  Entry {i + 1}:")
                print(f"    {str(entry)[:200]}...")

        # Tracked tests
        if hasattr(agent, "tracked_tests"):
            print_section("TRACKED TESTS")
            print(f"Total tests tracked: {len(agent.tracked_tests)}")
            for i, test in enumerate(agent.tracked_tests):
                print(f"\nTest {i + 1}:")
                if isinstance(test, dict):
                    print(f"  Name: {test.get('test_name', 'Unknown')}")
                    print(f"  P-value: {test.get('p_value', 'N/A')}")
                    print(f"  Decision: {test.get('decision', 'N/A')}")
                    print(f"  Summary: {test.get('summary', 'N/A')[:100]}...")
                else:
                    # Handle string format
                    print(f"  Description: {str(test)[:200]}...")
                    if hasattr(agent, "tracked_stat") and i < len(agent.tracked_stat):
                        print(f"  P-value: {agent.tracked_stat[i]}")
                    else:
                        print("  P-value: N/A")

    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()

    print_section("TEST COMPLETE")


def test_hypothesis_prompt_generation():
    """Test the hypothesis testing prompt generation"""
    print_section("HYPOTHESIS TESTING PROMPT TEST")

    # Create a minimal version to test prompt generation
    from biomni.agent.popper_agent import A1TestCodingAgent

    class MockA1Agent:
        def go(self, prompt):
            return [], "Mock output"

    class MockDataLoader:
        def get_data_description(self):
            return "Mock data: dataset1, dataset2, dataset3"

    class MockLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            class MockResult:
                def dict(self):
                    return {"check_output_error": "no", "p_val": "0.05"}

            return MockResult()

    # Create test instance
    test_agent = A1TestCodingAgent(data_loader=MockDataLoader(), llm=MockLLM(), a1_agent=MockA1Agent(), verbose=False)

    # Generate a test prompt
    test_question = """Main hypothesis: Gene X affects Disease Y
Falsification Test name: Expression correlation test
Falsification Test description: Test if Gene X expression correlates with Disease Y severity
Falsification Test Null sub-hypothesis: No correlation between Gene X and Disease Y
Falsification Test Alternate sub-hypothesis: Significant correlation exists"""

    prompt = test_agent._create_hypothesis_testing_prompt(test_question)

    print("Generated Hypothesis Testing Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


if __name__ == "__main__":
    # Run the main test
    test_popper()

    # Also test prompt generation
    test_hypothesis_prompt_generation()
