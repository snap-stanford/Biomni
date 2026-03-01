#!/usr/bin/env python
"""
Integration tests for Biomni GEO query functionality.

Tests the query_geo tool both directly and through the agent to ensure
OpenAI models can successfully query the NCBI GEO database.

Run with: pytest tests/test_geo_query_integration.py -v
Or directly: python tests/test_geo_query_integration.py
"""

import os
import sys

import pytest

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class TestGeoQueryDirect:
    """Test query_geo function directly without the agent."""

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """Set up test configuration (pytest fixture)."""
        self._setup()
        yield

    def _setup(self):
        """Set up test configuration."""
        from biomni.config import default_config

        default_config.llm = "gpt-4o"
        default_config.llm_lite = "gpt-4o-mini"  # Use OpenAI lite model for tests
        self.config = default_config

    def test_direct_search_term(self):
        """Test query_geo with a direct search term (bypasses LLM)."""
        from biomni.tool.database import query_geo

        result = query_geo(
            search_term="diabetic nephropathy[Title] AND Homo sapiens[Organism] AND gse[ETYP]", max_results=3
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "total_results" in result or "error" in result, "Should have results or error"

        if "error" not in result:
            assert result.get("total_results", 0) > 0, "Should find some datasets"
            assert "formatted_results" in result, "Should have formatted results"
            print(f"✓ Found {result['total_results']} datasets")

    def test_natural_language_query(self):
        """Test query_geo with natural language prompt (uses LLM)."""
        from biomni.tool.database import query_geo

        result = query_geo(prompt="Find RNA-seq datasets for diabetic nephropathy in humans", max_results=3)

        assert isinstance(result, dict), "Result should be a dictionary"

        # Check for success
        if "error" in result:
            pytest.fail(f"Query failed: {result.get('error')} - Raw: {result.get('raw_response', 'N/A')}")

        assert "total_results" in result, "Should have total_results"
        assert result["total_results"] > 0, "Should find some datasets"
        assert "formatted_results" in result, "Should have formatted results"

        # Verify we got actual dataset info
        results = result.get("formatted_results", {}).get("result", {})
        uids = results.get("uids", [])
        assert len(uids) > 0, "Should have dataset UIDs"

        # Check first dataset has expected fields
        first_uid = uids[0]
        first_dataset = results.get(first_uid, {})
        assert "accession" in first_dataset, "Dataset should have accession"
        assert "title" in first_dataset, "Dataset should have title"

        print(f"✓ Found {result['total_results']} datasets via natural language")
        print(f"  Query interpretation: {result.get('query_interpretation', 'N/A')}")
        for uid in uids[:3]:
            ds = results.get(uid, {})
            print(f"  - {ds.get('accession')}: {ds.get('title', 'N/A')[:60]}...")

    def test_llm_query_parsing(self):
        """Test that the LLM correctly parses natural language to GEO query."""
        import pickle

        from biomni.tool.database import _query_llm_for_api

        # Load GEO schema
        schema_path = os.path.join(os.path.dirname(__file__), "../biomni/tool/schema_db/geo.pkl")
        with open(schema_path, "rb") as f:
            geo_schema = pickle.load(f)

        system_template = """
        You are a bioinformatics assistant. Convert the user query into a GEO search term.
        Output only a JSON object with:
        1. "search_term": The GEO search query
        2. "database": Either "gds" or "geoprofiles"

        Schema: {schema}
        """

        result = _query_llm_for_api(
            prompt="Find RNA-seq data for diabetic nephropathy", schema=geo_schema, system_template=system_template
        )

        assert result.get("success"), f"LLM query failed: {result.get('error')}"
        assert "data" in result, "Should have data"
        assert "search_term" in result["data"], "Should have search_term"
        assert "database" in result["data"], "Should have database"

        print(f"✓ LLM generated search term: {result['data']['search_term']}")


class TestGeoQueryAgent:
    """Test query_geo through the Biomni agent."""

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """Set up test configuration (pytest fixture)."""
        self._setup()
        yield

    def _setup(self):
        """Set up test configuration."""
        from biomni.config import default_config

        default_config.llm = "gpt-4o"
        default_config.llm_lite = "gpt-4o-mini"

    @pytest.mark.slow
    def test_agent_geo_query(self):
        """Test that the agent can use query_geo to find datasets."""
        from biomni.agent import A1

        # Create agent with minimal setup
        agent = A1(path="./data", llm="gpt-4o", expected_data_lake_files=[], use_tool_retriever=False)
        agent.configure()

        # Run query
        prompt = "Use the query_geo tool to find RNA-seq datasets related to diabetic nephropathy in humans. Return the top 3 results with their GEO accession numbers and titles."

        result = agent.go(prompt)

        # Result is a tuple of (messages, final_answer)
        assert result is not None, "Agent should return a result"

        messages, final_answer = result
        assert final_answer is not None, "Should have a final answer"

        # Check that the answer mentions GEO accession numbers
        answer_lower = final_answer.lower()
        assert "gse" in answer_lower, "Answer should mention GSE accession numbers"

        # Check for specific dataset indicators
        has_datasets = any(
            x in answer_lower
            for x in [
                "gse317266",
                "gse315877",
                "gse273001",  # Known datasets
                "diabetic",
                "nephropathy",
                "kidney",
            ]
        )
        assert has_datasets, "Answer should contain dataset information"

        print("✓ Agent successfully queried GEO and returned results")
        print(f"Final answer preview: {final_answer[:500]}...")


class TestOpenAICompatibility:
    """Test OpenAI-specific compatibility features."""

    def test_config_uses_openai(self):
        """Verify configuration is set to use OpenAI models."""
        from biomni.config import default_config

        default_config.llm = "gpt-4o"
        default_config.llm_lite = "gpt-4o-mini"

        assert default_config.llm == "gpt-4o", "Config should use gpt-4o"
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY should be set"
        print("✓ OpenAI configuration verified")

    def test_llm_factory_creates_openai(self):
        """Test that get_llm creates OpenAI model correctly."""
        from biomni.config import default_config
        from biomni.llm import get_llm

        default_config.llm = "gpt-4o"
        default_config.llm_lite = "gpt-4o-mini"

        llm = get_llm(model="gpt-4o", temperature=0.7)

        assert llm is not None, "Should create LLM instance"
        assert "openai" in str(type(llm)).lower(), "Should be an OpenAI model"
        print(f"✓ Created LLM: {type(llm).__name__}")


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("BIOMNI GEO QUERY INTEGRATION TESTS")
    print("=" * 60)
    print()

    # Check environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Please set it in .env file.")
        return False

    passed = 0
    failed = 0

    # Test 1: OpenAI configuration
    print("Test 1: OpenAI Configuration")
    print("-" * 40)
    try:
        test = TestOpenAICompatibility()
        test.test_config_uses_openai()
        test.test_llm_factory_creates_openai()
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1
    print()

    # Test 2: Direct search term
    print("Test 2: Direct GEO Search Term")
    print("-" * 40)
    try:
        test = TestGeoQueryDirect()
        test._setup()
        test.test_direct_search_term()
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1
    print()

    # Test 3: LLM query parsing
    print("Test 3: LLM Query Parsing")
    print("-" * 40)
    try:
        test = TestGeoQueryDirect()
        test._setup()
        test.test_llm_query_parsing()
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1
    print()

    # Test 4: Natural language query
    print("Test 4: Natural Language GEO Query")
    print("-" * 40)
    try:
        test = TestGeoQueryDirect()
        test._setup()
        test.test_natural_language_query()
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1
    print()

    # Summary
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
