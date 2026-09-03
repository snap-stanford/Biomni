import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from biomni.tool import literature

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


@pytest.fixture
def fixture_payload() -> dict:
    return json.loads((FIXTURES_DIR / "europe_pmc_2020.json").read_text(encoding="utf-8"))["response"]


def test_extract_records_ignores_non_lists():
    assert literature._extract_europe_pmc_records({}) == []
    assert literature._extract_europe_pmc_records({"resultList": {"result": {"id": "1"}}}) == []


def test_format_record_uses_abstract_and_journal():
    record = {
        "title": "Europe PMC in 2020.",
        "abstractText": "Europe PMC is a database of life science literature.",
        "journalTitle": "Nucleic acids research",
    }
    output = literature._format_europe_pmc_record(record)
    assert "Title: Europe PMC in 2020." in output
    assert "Abstract: Europe PMC is a database of life science literature." in output
    assert "Journal: Nucleic acids research" in output


def test_search_europe_pmc_returns_records(fixture_payload, monkeypatch):
    session = Mock()
    session.get.return_value = Response(fixture_payload)
    records = literature._search_europe_pmc('TITLE:"Europe PMC"', page_size=2, session=session)
    assert records[0]["pmid"] == "33180112"
    session.get.assert_called_once()


def test_query_europe_pmc_returns_formatted_results(fixture_payload, monkeypatch):
    monkeypatch.setattr(
        literature,
        "_search_europe_pmc",
        lambda query, page_size=25: literature._extract_europe_pmc_records(fixture_payload),
    )
    result = literature.query_europe_pmc('ABSTRACT:"Europe PMC"', max_papers=1, max_retries=0)
    assert "Title: Europe PMC in 2020." in result
    assert "Abstract: Europe PMC (https://europepmc.org) is a database" in result
    assert "Journal: Nucleic acids research" in result


def test_query_europe_pmc_retries_with_simplified_query(monkeypatch):
    queries = []

    def fake_search(query, page_size=25):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [{"title": "Paper", "abstractText": "Abstract", "journalTitle": "Journal"}]

    monkeypatch.setattr(literature, "_search_europe_pmc", fake_search)
    monkeypatch.setattr(literature.time, "sleep", lambda seconds: None)
    result = literature.query_europe_pmc("single cell atlases", max_papers=1, max_retries=1)
    assert queries == ["single cell atlases", "single cell"]
    assert "Title: Paper" in result


def test_query_europe_pmc_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_europe_pmc", lambda query, page_size=25: [])
    result = literature.query_europe_pmc("missing", max_papers=1, max_retries=1)
    assert result == "No papers found on Europe PMC after multiple query attempts."


def test_query_europe_pmc_returns_error_string_on_http_error(monkeypatch):
    def fake_search(query, page_size=25):
        raise requests.HTTPError("429 Client Error")

    monkeypatch.setattr(literature, "_search_europe_pmc", fake_search)
    result = literature.query_europe_pmc("malaria")
    assert result == "Error querying Europe PMC: 429 Client Error"


def test_query_europe_pmc_passes_max_papers_through(monkeypatch):
    page_sizes = []

    def fake_search(query, page_size=25):
        del query
        page_sizes.append(page_size)
        return []

    monkeypatch.setattr(literature, "_search_europe_pmc", fake_search)
    literature.query_europe_pmc("malaria", max_papers=100, max_retries=0)
    assert page_sizes == [100]


def test_query_europe_pmc_rejects_empty_query():
    result = literature.query_europe_pmc("   ")
    assert result == "Error querying Europe PMC: Query must not be empty."


def test_literature_tool_description_exposes_only_query_tool():
    from biomni.tool.tool_description.literature import description

    names = [tool["name"] for tool in description]
    assert "query_europe_pmc" in names
    assert "query_pubmed" in names
