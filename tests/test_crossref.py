import json
from pathlib import Path
from unittest.mock import Mock

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


def fixture_payload() -> dict:
    return json.loads((FIXTURES_DIR / "crossref_crispr.json").read_text(encoding="utf-8"))["response"]


def test_extract_records_ignores_non_lists():
    assert literature._extract_crossref_records({}) == []
    assert literature._extract_crossref_records({"message": {"items": {"DOI": "1"}}}) == []


def test_format_record_uses_title_abstract_and_journal():
    record = {
        "title": ["Example paper"],
        "abstract": "<jats:p>Structured abstract text.</jats:p>",
        "container-title": ["Example Journal"],
    }
    output = literature._format_crossref_record(record)
    assert "Title: Example paper" in output
    assert "Abstract: Structured abstract text." in output
    assert "Journal: Example Journal" in output


def test_search_crossref_returns_records(monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    records = literature._search_crossref("CRISPR", page_size=1, session=session)
    assert records[0]["DOI"] == "10.1089/crispr.2020.29108.kda"
    session.get.assert_called_once()


def test_search_crossref_uses_mailto_when_configured(monkeypatch):
    monkeypatch.setenv("CROSSREF_MAILTO", "test@example.com")
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    literature._search_crossref("CRISPR", page_size=1, session=session)
    params = session.get.call_args.kwargs["params"]
    assert params["mailto"] == "test@example.com"


def test_query_crossref_returns_formatted_results(monkeypatch):
    monkeypatch.setattr(
        literature,
        "_search_crossref",
        lambda query, page_size=25: literature._extract_crossref_records(fixture_payload()),
    )
    result = literature.query_crossref("CRISPR", max_papers=1, max_retries=0)
    assert "Title: My CRISPR Book" in result
    assert "Abstract: No abstract available." in result
    assert "Journal: The CRISPR Journal" in result


def test_query_crossref_retries_with_simplified_query(monkeypatch):
    queries = []

    def fake_search(query, page_size=25):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [{"title": ["Paper"], "abstract": "Abstract", "container-title": ["Journal"]}]

    monkeypatch.setattr(literature, "_search_crossref", fake_search)
    monkeypatch.setattr(literature.time, "sleep", lambda seconds: None)
    result = literature.query_crossref("single cell atlases", max_papers=1, max_retries=1)
    assert queries == ["single cell atlases", "single cell"]
    assert "Title: Paper" in result


def test_query_crossref_passes_max_papers_through(monkeypatch):
    page_sizes = []

    def fake_search(query, page_size=25):
        page_sizes.append(page_size)
        return [{"title": ["Paper"], "abstract": "Abstract", "container-title": ["Journal"]}]

    monkeypatch.setattr(literature, "_search_crossref", fake_search)
    literature.query_crossref("malaria", max_papers=100, max_retries=0)
    assert page_sizes == [100]


def test_query_crossref_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_crossref", lambda query, page_size=25: [])
    result = literature.query_crossref("missing", max_papers=1, max_retries=1)
    assert result == "No papers found on Crossref after multiple query attempts."


def test_query_crossref_returns_error_string_on_http_error(monkeypatch):
    def fake_search(query, page_size=25):
        raise requests.HTTPError("429 Client Error")

    monkeypatch.setattr(literature, "_search_crossref", fake_search)
    result = literature.query_crossref("malaria")
    assert result == "Error querying Crossref: 429 Client Error"


def test_query_crossref_rejects_empty_query():
    result = literature.query_crossref("   ")
    assert result == "Error querying Crossref: Query must not be empty."


def test_literature_tool_description_exposes_crossref():
    from biomni.tool.tool_description.literature import description

    names = [tool["name"] for tool in description]
    assert "query_crossref" in names
    assert "query_pubmed" in names
