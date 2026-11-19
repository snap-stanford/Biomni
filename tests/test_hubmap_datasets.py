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
    return json.loads((FIXTURES_DIR / "hubmap_ovary_dataset.json").read_text(encoding="utf-8"))["response"]


def fixture_dataset():
    return fixture_payload()["hits"]["hits"][0]["_source"]


def test_extract_hubmap_dataset_hits_returns_sources():
    datasets = database._extract_hubmap_dataset_hits(fixture_payload())
    assert len(datasets) == 1
    assert datasets[0]["hubmap_id"] == "HBM522.GTLH.372"


def test_get_hubmap_values_handles_lists_and_scalars():
    assert database._get_hubmap_values(["ovary", None, "kidney"]) == ["ovary", "kidney"]
    assert database._get_hubmap_values("public") == ["public"]
    assert database._get_hubmap_values(None) == []


def test_hubmap_dataset_matches_metadata_terms():
    dataset = fixture_dataset()
    assert database._hubmap_dataset_matches(dataset, ["ovary"])
    assert database._hubmap_dataset_matches(dataset, ["10x", "multiome"])
    assert database._hubmap_dataset_matches(dataset, ["pennsylvania"])
    assert not database._hubmap_dataset_matches(dataset, ["kidney"])


def test_format_hubmap_dataset_returns_metadata_summary():
    output = database._format_hubmap_dataset(fixture_dataset())
    assert "Dataset: 10X Multiome [Salmon + ArchR + Muon] data from the ovary" in output
    assert "HuBMAP ID: HBM522.GTLH.372" in output
    assert "UUID: ba4753cba9dcb01d54185d737df9022c" in output
    assert "Dataset Type: 10X Multiome [Salmon + ArchR + Muon]" in output
    assert "Organ: Ovary (Left)" in output
    assert "Group: TMC - University of Pennsylvania" in output
    assert "Data Access Level: public" in output
    assert "Spatial: Yes" in output
    assert "File Count: 2" in output


def test_search_hubmap_datasets_calls_public_endpoint():
    session = Mock()
    session.post.return_value = Response(fixture_payload())
    datasets = database._search_hubmap_datasets("ovary multiome", max_results=1, session=session)
    assert len(datasets) == 1
    session.post.assert_called_once_with(
        "https://search.api.hubmapconsortium.org/v3/portal/search",
        json={"query": {"bool": {"filter": [{"term": {"entity_type.keyword": "Dataset"}}]}}, "size": 10},
        timeout=30,
    )


def test_query_hubmap_datasets_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(database, "_search_hubmap_datasets", lambda query, max_results=5: [fixture_dataset()])
    result = database.query_hubmap_datasets("ovary multiome", max_results=1)
    assert "HuBMAP ID: HBM522.GTLH.372" in result
    assert "Organ: Ovary (Left)" in result


def test_query_hubmap_datasets_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(database, "_search_hubmap_datasets", lambda query, max_results=5: [])
    result = database.query_hubmap_datasets("missing")
    assert result == "No HuBMAP datasets found."


def test_query_hubmap_datasets_rejects_empty_query():
    result = database.query_hubmap_datasets("   ")
    assert result == "Error querying HuBMAP: Query must not be empty."


def test_query_hubmap_datasets_returns_error_string_on_http_error(monkeypatch):
    def raise_error(query, max_results=5):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(database, "_search_hubmap_datasets", raise_error)
    result = database.query_hubmap_datasets("ovary")
    assert result == "Error querying HuBMAP: 500 Server Error"


def test_hubmap_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_hubmap_datasets" in names
