#!/usr/bin/env python
"""Test Biomni agent with GEO query functionality.

This script tests the full agent with a simple GEO query task.
Run with: python tests/test_agent_geo.py
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from biomni.config import default_config

print("=" * 60)
print("BIOMNI AGENT GEO QUERY TEST")
print("=" * 60)
print(f"LLM (reasoning): {default_config.llm}")
print(f"LLM Lite (simple): {default_config.llm_lite}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")
print()

# Test: Run the agent with a GEO query
print("TEST: Agent query for diabetic nephropathy datasets")
print("-" * 40)

try:
    from biomni.agent import A1

    # Create agent with config from .env
    print("Creating agent...")
    agent = A1(
        path="./data",
        llm=default_config.llm,  # Use config from .env
        expected_data_lake_files=[],
        use_tool_retriever=False,  # Disable retriever for simpler testing
    )

    # Configure the agent
    print("Configuring agent...")
    agent.configure()

    # Run a simple query
    print("Running query...")
    prompt = "Use the query_geo tool to find RNA-seq datasets related to diabetic nephropathy in humans. Return the top 3 results with their GEO accession numbers and titles."

    result = agent.go(prompt)

    print("\n" + "=" * 40)
    print("AGENT RESULT:")
    print("=" * 40)

    if result:
        messages, final_answer = result
        print(f"Final Answer:\n{final_answer}")
    else:
        print("No result returned")

except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
