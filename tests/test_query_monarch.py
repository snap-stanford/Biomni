import pytest
from biomni.tool import database


def test_query_monarch_prompt():
    # This test assumes the Monarch API and Claude are accessible
    result = database.query_monarch(prompt="Find phenotypes associated with BRCA1", max_results=2, verbose=False)
    assert isinstance(result, dict)
    assert "error" in result or "result" in result or "success" in result


def test_query_monarch_endpoint():
    # Direct endpoint test (no Claude required)
    result = database.query_monarch(endpoint="/bioentity/gene/NCBIGene:672/phenotypes", max_results=2, verbose=False)
    assert isinstance(result, dict)
    assert "error" in result or "result" in result or "success" in result


def test_query_monarch_error():
    # Should error if neither prompt nor endpoint is provided
    result = database.query_monarch()
    assert "error" in result
