#!/usr/bin/env python
"""Test script for Biomni GEO query functionality.

This script tests the query_geo function directly without the full agent.
Run with: python tests/test_geo_query.py
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
print("BIOMNI GEO QUERY TEST")
print("=" * 60)
print(f"LLM (reasoning): {default_config.llm}")
print(f"LLM Lite (simple): {default_config.llm_lite}")
print()

# Test 1: Direct search term (bypass LLM)
print("TEST 1: Direct search term (bypassing LLM)")
print("-" * 40)

try:
    from biomni.tool.database import query_geo

    result = query_geo(
        search_term="diabetic nephropathy[Title] AND Homo sapiens[Organism] AND gse[ETYP]", max_results=3
    )

    if isinstance(result, dict):
        if "total_results" in result:
            print(f"SUCCESS! Found {result.get('total_results')} datasets")
        elif result.get("error"):
            print(f"ERROR: {result.get('error')}")
        else:
            print(f"Result keys: {result.keys()}")
    else:
        print(f"Result: {result}")

except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 2: Natural language query (uses LLM)
print("TEST 2: Natural language query (uses LLM lite)")
print("-" * 40)

try:
    from biomni.tool.database import query_geo

    result = query_geo(prompt="Find RNA-seq datasets for diabetic nephropathy in humans", max_results=3)

    if isinstance(result, dict):
        if "total_results" in result:
            print(f"SUCCESS! Found {result.get('total_results')} datasets")
            print(f"Query interpretation: {result.get('query_interpretation', 'N/A')}")
        elif result.get("error"):
            print(f"ERROR: {result.get('error')}")
            if result.get("raw_response"):
                print(f"Raw response: {result.get('raw_response')[:200]}")
        else:
            print(f"Result keys: {result.keys()}")
    else:
        print(f"Result: {result}")

except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 60)
print("TESTS COMPLETE")
print("=" * 60)
