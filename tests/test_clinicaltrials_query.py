"""Tests for query_clinicaltrials parameter handling (issue #215).

ClinicalTrials.gov API v2 does not support ``filter.phase`` or
``filter.intervention``. This test suite locks in the correct parameter
usage (``query.intr`` and ``filter.advanced=AREA[Phase]...``), verifies
that the schema examples and system prompt teach valid parameters, and
checks that the 400-error fallback converts old-style ``filter.phase``
into the supported ``filter.advanced`` expression.

Unit tests mock the network layer (``_query_rest_api``) so they are
deterministic. Integration tests hit the real API and are marked with
``@pytest.mark.integration``.
"""

import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SCHEMA_PATH = REPO_ROOT / "biomni" / "tool" / "schema_db" / "clinicaltrials.pkl"

sys.path.insert(0, str(REPO_ROOT))

from biomni.tool.database import query_clinicaltrials

OFFICIAL_PHASES = {"NA", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeRestAPI:
    """Records calls and returns preset responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, endpoint, method="GET", description="", params=None):
        self.calls.append({"endpoint": endpoint, "method": method, "description": description})
        if self.responses:
            return self.responses.pop(0)
        return {"success": True, "result": {}}


@pytest.fixture
def fake_rest_api(monkeypatch):
    """Install a FakeRestAPI as _query_rest_api and return it."""
    fake = FakeRestAPI([])
    monkeypatch.setattr("biomni.tool.database._query_rest_api", fake)
    return fake


# ---------------------------------------------------------------------------
# Unit tests: endpoint construction
# ---------------------------------------------------------------------------


def test_query_intr_endpoint_constructed(fake_rest_api):
    """A direct endpoint using query.intr is passed through with pageSize added."""
    fake_rest_api.responses = [{"success": True, "result": {"studies": []}}]

    result = query_clinicaltrials(endpoint="/studies?query.intr=aspirin", max_results=5)

    assert result["success"] is True
    assert len(fake_rest_api.calls) == 1
    url = fake_rest_api.calls[0]["endpoint"]
    assert "https://clinicaltrials.gov/api/v2" in url
    assert "query.intr=aspirin" in url
    assert "pageSize=5" in url


def test_phase_advanced_endpoint_passed_through(fake_rest_api):
    """A direct endpoint using filter.advanced is passed through untouched."""
    fake_rest_api.responses = [{"success": True, "result": {"studies": []}}]

    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.advanced=AREA[Phase]PHASE3")

    assert result["success"] is True
    url = fake_rest_api.calls[0]["endpoint"]
    assert "filter.advanced=AREA[Phase]PHASE3" in url


def test_no_prompt_no_endpoint_returns_error():
    """Calling with neither prompt nor endpoint returns a clear error."""
    result = query_clinicaltrials()
    assert result["error"] == "Either a prompt or an endpoint must be provided"


# ---------------------------------------------------------------------------
# Unit tests: system prompt content
# ---------------------------------------------------------------------------


def test_system_prompt_has_no_filter_phase():
    """The generated prompt must not teach the unsupported filter.phase param.

    The template must:
    - NOT contain "Use filter.phase" as an instruction
    - NOT contain "filter.phase=PHASE1, PHASE2" style teaching examples
    - DO contain the correct filter.advanced=AREA[Phase] syntax
    - DO mention that filter.intervention does not exist
    """
    import inspect
    import re as _re

    src = inspect.getsource(query_clinicaltrials)
    # Extract the system_template string literal from the source
    # (the template is a triple-quoted string in the function body).
    template_match = _re.search(r'system_template = """(.*?)"""', src, _re.DOTALL)
    assert template_match, "system_template not found in source"
    template = template_match.group(1)

    # Must not teach the unsupported parameter as a usage instruction
    assert "Use filter.phase" not in template
    assert "filter.phase=PHASE1, PHASE2" not in template

    # Must teach the correct syntax
    assert "filter.advanced=AREA[Phase]" in template
    assert "query.intr" in template
    assert "NOT filter.intervention" in template


# ---------------------------------------------------------------------------
# Unit tests: schema (clinicaltrials.pkl)
# ---------------------------------------------------------------------------


def test_schema_examples_use_valid_params():
    """Schema examples must not contain filter.phase or filter.intervention."""
    with open(SCHEMA_PATH, "rb") as f:
        schema = pickle.load(f)

    serialized = str(schema)
    assert "filter.phase" not in serialized, "schema examples still use filter.phase"
    assert "filter.intervention" not in serialized, "schema still uses filter.intervention"

    recruiting = schema.get("examples", {}).get("recruiting_phase3", {})
    params = recruiting.get("parameters", {})
    assert "filter.advanced" in params
    assert "AREA[Phase]" in params.get("filter.advanced", "")


def test_schema_study_phases_match_official():
    """study_phases must equal the six values supported by API v2."""
    with open(SCHEMA_PATH, "rb") as f:
        schema = pickle.load(f)

    phases = set(schema.get("study_phases", []))
    assert phases == OFFICIAL_PHASES


# ---------------------------------------------------------------------------
# Unit tests: 400 fallback
# ---------------------------------------------------------------------------


def test_400_fallback_converts_filter_phase(fake_rest_api):
    """A 400 with old-style filter.phase retries as filter.advanced."""
    fake_rest_api.responses = [
        {"success": False, "error": "HTTP 400: `filter.phase` is unknown parameter"},
        {"success": True, "result": {"studies": [{"nctId": "NCT1"}]}},
    ]

    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.phase=PHASE3")

    assert result["success"] is True
    assert len(fake_rest_api.calls) == 2
    retry_url = fake_rest_api.calls[1]["endpoint"]
    assert "filter.phase" not in retry_url
    # Generated by the fallback, so brackets/spaces are URL-encoded
    assert "filter.advanced=AREA%5BPhase%5DPHASE3" in retry_url
    assert result.get("note") == "Converted unsupported filter.phase to filter.advanced"


def test_400_fallback_converts_multiple_phases(fake_rest_api):
    """Multiple phases in filter.phase become OR-combined, URL-encoded AREA expressions."""
    fake_rest_api.responses = [
        {"success": False, "error": "HTTP 400: `filter.phase` is unknown parameter"},
        {"success": True, "result": {"studies": []}},
    ]

    query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.phase=PHASE1,PHASE2")

    retry_url = fake_rest_api.calls[1]["endpoint"]
    # Space between OR terms must be URL-encoded (%20) for requests.get to work
    assert "filter.advanced=AREA%5BPhase%5DPHASE1%20OR%20AREA%5BPhase%5DPHASE2" in retry_url


def test_400_without_filter_phase_no_retry(fake_rest_api):
    """A 400 that is not phase-related must not trigger a conversion retry."""
    fake_rest_api.responses = [
        {"success": False, "error": "HTTP 400: some other parameter error"},
    ]

    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer")

    assert result["success"] is False
    assert len(fake_rest_api.calls) == 1


# ---------------------------------------------------------------------------
# Integration tests (real API, marked)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_query_intr_returns_studies():
    """Real API: query.intr is a valid parameter and returns studies."""
    result = query_clinicaltrials(endpoint="/studies?query.intr=aspirin&pageSize=2")

    assert result["success"] is True
    assert len(result.get("result", {}).get("studies", [])) > 0


@pytest.mark.integration
def test_real_phase_advanced_returns_phase3():
    """Real API: filter.advanced=AREA[Phase]PHASE3 returns Phase 3 studies."""
    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.advanced=AREA[Phase]PHASE3&pageSize=2")

    assert result["success"] is True
    studies = result.get("result", {}).get("studies", [])
    assert len(studies) > 0
    for study in studies:
        phases = study.get("protocolSection", {}).get("designModule", {}).get("phases", [])
        assert "PHASE3" in phases


@pytest.mark.integration
def test_real_bad_param_auto_converts():
    """Real API: old filter.phase is auto-converted to filter.advanced and succeeds.

    This verifies the fallback path against the live API: the first request
    with the unsupported ``filter.phase`` returns 400, and the function retries
    with ``filter.advanced=AREA[Phase]PHASE3``, which succeeds.
    """
    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.phase=PHASE3")

    # The fallback converts filter.phase -> filter.advanced, so it succeeds
    assert result["success"] is True
    assert result.get("note") == "Converted unsupported filter.phase to filter.advanced"
    studies = result.get("result", {}).get("studies", [])
    assert len(studies) > 0
    for study in studies:
        phases = study.get("protocolSection", {}).get("designModule", {}).get("phases", [])
        assert "PHASE3" in phases


@pytest.mark.integration
def test_real_multi_phase_auto_converts():
    """Real API: comma-separated filter.phase converts to URL-encoded OR expression."""
    result = query_clinicaltrials(endpoint="/studies?query.cond=cancer&filter.phase=PHASE1,PHASE2&pageSize=3")

    assert result["success"] is True
    assert result.get("note") == "Converted unsupported filter.phase to filter.advanced"
    studies = result.get("result", {}).get("studies", [])
    assert len(studies) > 0
    for study in studies:
        phases = study.get("protocolSection", {}).get("designModule", {}).get("phases", [])
        assert any(p in phases for p in ("PHASE1", "PHASE2"))
