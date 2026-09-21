import json
from pathlib import Path
from unittest.mock import Mock

from biomni.tool import database
from biomni.tool.tool_description import database as database_description

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture_payload():
    return json.loads((FIXTURES_DIR / "organoid_cell_atlas_projects.json").read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_organoid_cell_atlas_project_matches_terms():
    hit = fixture_payload()["hits"][0]
    assert database._organoid_cell_atlas_project_matches(hit, "lung organoid")
    assert not database._organoid_cell_atlas_project_matches(hit, "kidney")


def test_format_organoid_cell_atlas_project_returns_metadata_summary():
    output = database._format_organoid_cell_atlas_project(fixture_payload()["hits"][0])
    assert "Project: A human fetal lung cell atlas" in output
    assert "Project ID: 2fe3c60b-ac1a-4c61-9b59-f6556c0fce63" in output
    assert (
        "HCA Data Portal URL: https://data.humancellatlas.org/explore/projects/2fe3c60b-ac1a-4c61-9b59-f6556c0fce63"
        in output
    )
    assert "Bionetwork: Development, Organoid" in output
    assert "Tissue Atlas: Development (v1.0), Organoid-Endoderm (v1.0)" in output
    assert "Organoid Model Organs: lung" in output
    assert "Species: Homo sapiens" in output
    assert "File Types: h5ad (15), fastq.gz (402)" in output


def test_search_organoid_cell_atlas_projects_uses_hca_organoid_filter():
    session = Mock()
    session.get.return_value = FakeResponse(fixture_payload())

    results = database._search_organoid_cell_atlas_projects("lung", max_results=1, session=session)

    assert len(results) == 1
    assert results[0]["entryId"] == "2fe3c60b-ac1a-4c61-9b59-f6556c0fce63"
    session.get.assert_called_once()
    kwargs = session.get.call_args.kwargs
    assert kwargs["params"]["filters"] == json.dumps({"sampleEntityType": {"is": ["organoids"]}})
    assert kwargs["params"]["size"] == 1


def test_search_organoid_cell_atlas_projects_follows_pagination():
    first_page = {
        "hits": [],
        "pagination": {"next": "https://service.azul.data.humancellatlas.org/index/projects?next"},
    }
    second_page = fixture_payload()
    session = Mock()
    session.get.side_effect = [FakeResponse(first_page), FakeResponse(second_page)]

    results = database._search_organoid_cell_atlas_projects("intestine", max_results=1, session=session)

    assert len(results) == 1
    assert results[0]["entryId"] == "8ab8726d-81b9-4bd2-acc2-4d50bee786b4"
    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["params"] is None


def test_query_organoid_cell_atlas_projects_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(
        database, "_search_organoid_cell_atlas_projects", lambda query, max_results: fixture_payload()["hits"][:1]
    )

    result = database.query_organoid_cell_atlas_projects("lung", max_results=1)

    assert "Project: A human fetal lung cell atlas" in result
    assert "Organoid Model Organs: lung" in result


def test_query_organoid_cell_atlas_projects_returns_error_string_on_failure(monkeypatch):
    def raise_error(query, max_results):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(database, "_search_organoid_cell_atlas_projects", raise_error)

    result = database.query_organoid_cell_atlas_projects("lung")

    assert result == "Error querying HCA Organoid Cell Atlas projects: API unavailable"


def test_organoid_cell_atlas_projects_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_organoid_cell_atlas_projects" in names
