import pytest
from biomni.tool.database import query_openfda


def test_openfda_direct_endpoint():
    # Test direct endpoint for drug event
    endpoint = "https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:lipitor&limit=1"
    result = query_openfda(endpoint=endpoint, verbose=False)
    assert isinstance(result, dict)
    assert result.get("success", True)  # May not have 'success' if no error
    assert "results" in str(result).lower() or "error" not in result


def test_openfda_prompt_drug_event():
    # Test prompt for adverse events
    prompt = "Find adverse events for Lipitor, limit 1"
    result = query_openfda(prompt=prompt, max_results=1, verbose=False, model="gemini-2.5-flash")
    assert isinstance(result, dict)
    assert result.get("success", True)
    assert "results" in str(result).lower() or "error" not in result


def test_openfda_prompt_drug_label():
    # Test prompt for drug label
    prompt = "Get the drug label for Lipitor, limit 1"
    result = query_openfda(prompt=prompt, max_results=1, verbose=False, model="gemini-2.5-flash")
    assert isinstance(result, dict)
    assert result.get("success", True)
    assert "results" in str(result).lower() or "error" not in result


def test_openfda_invalid():
    # Test invalid prompt
    prompt = "This is not a real drug or endpoint"
    result = query_openfda(prompt=prompt, max_results=1, verbose=False)
    assert isinstance(result, dict)
    assert "error" in result or result.get("success", True)
