import json
from pathlib import Path
from unittest.mock import Mock

import requests
from biomni.tool import database
from biomni.tool.tool_description import database as database_description

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class Response:
    def __init__(self, payload=None, *, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def fixture_payload():
    return json.loads((FIXTURES_DIR / "hca_brain_project.json").read_text(encoding="utf-8"))["response"]


def fixture_project():
    return fixture_payload()["hits"][0]


def test_extract_hca_projects_returns_hits():
    projects = database._extract_hca_projects(fixture_payload())
    assert len(projects) == 1
    assert projects[0]["entryId"] == "74b6d569-3b11-42ef-b6b1-a0454522b4a0"


def test_get_hca_values_deduplicates_values():
    values = database._get_hca_values(
        [{"organ": ["brain", "brain"]}, {"organ": ["lung"]}, {"organ": []}],
        "organ",
    )
    assert values == ["brain", "lung"]


def test_hca_project_matches_metadata_terms():
    project = fixture_project()
    assert database._hca_project_matches(project, ["brain", "mice"])
    assert database._hca_project_matches(project, ["mus", "musculus"])
    assert database._hca_project_matches(project, ["cortex"])
    assert not database._hca_project_matches(project, ["retina"])


def test_format_hca_project_returns_metadata_summary():
    output = database._format_hca_project(fixture_project())
    assert "Project: 1.3 Million Brain Cells from E18 Mice" in output
    assert "Project ID: 74b6d569-3b11-42ef-b6b1-a0454522b4a0" in output
    assert "Short Name: 1M Neurons" in output
    assert "Data Use Restriction: NRES" in output
    assert "Organs: brain" in output
    assert "Organ Parts: cortex" in output
    assert "Species: Mus musculus" in output
    assert "Diseases: normal" in output
    assert "Library Methods: 10x 3' v2" in output
    assert "Total Cells: 1330000" in output
    assert "File Types: fastq (16377), h5 (1)" in output


def test_search_hca_projects_calls_public_endpoint():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    projects = database._search_hca_projects("brain mice", max_results=1, session=session)
    assert len(projects) == 1
    session.get.assert_called_once_with(
        "https://service.azul.data.humancellatlas.org/index/projects",
        params={"size": 10},
        timeout=30,
    )


def test_query_human_cell_atlas_projects_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(database, "_search_hca_projects", lambda query, max_results=5: [fixture_project()])
    result = database.query_human_cell_atlas_projects("brain mice", max_results=1)
    assert "Project: 1.3 Million Brain Cells from E18 Mice" in result
    assert "Total Cells: 1330000" in result


def test_query_human_cell_atlas_projects_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(database, "_search_hca_projects", lambda query, max_results=5: [])
    result = database.query_human_cell_atlas_projects("missing")
    assert result == "No Human Cell Atlas projects found."


def test_query_human_cell_atlas_projects_rejects_empty_query():
    result = database.query_human_cell_atlas_projects("   ")
    assert result == "Error querying Human Cell Atlas: Query must not be empty."


def test_query_human_cell_atlas_projects_returns_error_string_on_http_error(monkeypatch):
    def raise_error(query, max_results=5):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(database, "_search_hca_projects", raise_error)
    result = database.query_human_cell_atlas_projects("brain")
    assert result == "Error querying Human Cell Atlas: 500 Server Error"


def test_human_cell_atlas_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_human_cell_atlas_projects" in names
