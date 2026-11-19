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
    return json.loads((FIXTURES_DIR / "openalex_crispr.json").read_text(encoding="utf-8"))["response"]


def test_extract_records_ignores_non_lists():
    assert literature._extract_openalex_records({}) == []
    assert literature._extract_openalex_records({"results": {"id": "1"}}) == []


def test_extract_openalex_abstract_reconstructs_text():
    record = {"abstract_inverted_index": {"Genome": [0], "editing": [1], "works.": [2]}}
    assert literature._extract_openalex_abstract(record) == "Genome editing works."


def test_format_record_uses_abstract_and_journal():
    record = {
        "title": "Multiplex Genome Engineering Using CRISPR/Cas Systems",
        "abstract_inverted_index": {"Functional": [0], "genome": [1], "editing.": [2]},
        "primary_location": {"source": {"display_name": "Science"}},
    }
    output = literature._format_openalex_record(record)
    assert "Title: Multiplex Genome Engineering Using CRISPR/Cas Systems" in output
    assert "Abstract: Functional genome editing." in output
    assert "Journal: Science" in output


def test_search_openalex_returns_records():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    records = literature._search_openalex("crispr", page_size=1, session=session)
    assert records[0]["id"] == "https://openalex.org/W2064815984"
    session.get.assert_called_once()


def test_query_openalex_returns_formatted_results(monkeypatch):
    monkeypatch.setattr(
        literature,
        "_search_openalex",
        lambda query, page_size=25: literature._extract_openalex_records(fixture_payload()),
    )
    result = literature.query_openalex("crispr", max_papers=1, max_retries=0)
    assert "Title: Multiplex Genome Engineering Using CRISPR/Cas Systems" in result
    assert "Abstract: Functional elucidation of causal genetic variants" in result
    assert "Journal: Science" in result


def test_query_openalex_retries_with_simplified_query(monkeypatch):
    queries = []

    def fake_search(query, page_size=25):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [
            {
                "title": "Paper",
                "abstract_inverted_index": {"Abstract": [0]},
                "primary_location": {"source": {"display_name": "Journal"}},
            }
        ]

    monkeypatch.setattr(literature, "_search_openalex", fake_search)
    monkeypatch.setattr(literature.time, "sleep", lambda seconds: None)
    result = literature.query_openalex("single cell atlases", max_papers=1, max_retries=1)
    assert queries == ["single cell atlases", "single cell"]
    assert "Title: Paper" in result


def test_query_openalex_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_openalex", lambda query, page_size=25: [])
    result = literature.query_openalex("missing", max_papers=1, max_retries=1)
    assert result == "No papers found on OpenAlex after multiple query attempts."


def test_query_openalex_returns_error_string_on_http_error(monkeypatch):
    def fake_search(query, page_size=25):
        raise requests.HTTPError("429 Client Error")

    monkeypatch.setattr(literature, "_search_openalex", fake_search)
    result = literature.query_openalex("malaria")
    assert result == "Error querying OpenAlex: 429 Client Error"


def test_query_openalex_rejects_empty_query():
    result = literature.query_openalex("   ")
    assert result == "Error querying OpenAlex: Query must not be empty."


def test_literature_tool_description_exposes_openalex():
    from biomni.tool.tool_description.literature import description

    names = [tool["name"] for tool in description]
    assert "query_openalex" in names
    assert "query_pubmed" in names
