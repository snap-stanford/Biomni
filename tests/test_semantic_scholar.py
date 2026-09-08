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
    return json.loads((FIXTURES_DIR / "semantic_scholar_crispr.json").read_text(encoding="utf-8"))["response"]


def test_extract_records_ignores_non_lists():
    assert literature._extract_semantic_scholar_records({}) == []
    assert literature._extract_semantic_scholar_records({"data": {"paperId": "1"}}) == []


def test_format_record_uses_abstract_and_journal():
    record = {
        "title": "Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9",
        "abstract": "CRISPR-Cas9 screens are a powerful tool.",
        "venue": "Nature Biotechnology",
    }
    output = literature._format_semantic_scholar_record(record)
    assert "Title: Optimized sgRNA design" in output
    assert "Abstract: CRISPR-Cas9 screens are a powerful tool." in output
    assert "Journal: Nature Biotechnology" in output


def test_search_semantic_scholar_returns_records():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    records = literature._search_semantic_scholar("CRISPR", page_size=1, session=session)
    assert records[0]["paperId"] == "b9e5fa707e804d6008e5011b058244437c656a93"
    session.get.assert_called_once()


def test_query_semantic_scholar_returns_formatted_results(monkeypatch):
    monkeypatch.setattr(
        literature,
        "_search_semantic_scholar",
        lambda query, page_size=25: literature._extract_semantic_scholar_records(fixture_payload()),
    )
    result = literature.query_semantic_scholar("CRISPR", max_papers=1, max_retries=0)
    assert "Title: Optimized sgRNA design to maximize activity" in result
    assert "Abstract: CRISPR-Cas9" in result
    assert "Journal: Nature Biotechnology" in result


def test_query_semantic_scholar_retries_with_simplified_query(monkeypatch):
    queries = []

    def fake_search(query, page_size=25):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [{"title": "Paper", "abstract": "Abstract", "venue": "Journal"}]

    monkeypatch.setattr(literature, "_search_semantic_scholar", fake_search)
    monkeypatch.setattr(literature.time, "sleep", lambda seconds: None)
    result = literature.query_semantic_scholar("single cell atlases", max_papers=1, max_retries=1)
    assert queries == ["single cell atlases", "single cell"]
    assert "Title: Paper" in result


def test_query_semantic_scholar_passes_max_papers_through(monkeypatch):
    page_sizes = []

    def fake_search(query, page_size=25):
        page_sizes.append(page_size)
        return [{"title": "Paper", "abstract": "Abstract", "venue": "Journal"}]

    monkeypatch.setattr(literature, "_search_semantic_scholar", fake_search)
    literature.query_semantic_scholar("malaria", max_papers=100, max_retries=0)
    assert page_sizes == [100]


def test_query_semantic_scholar_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_semantic_scholar", lambda query, page_size=25: [])
    result = literature.query_semantic_scholar("missing", max_papers=1, max_retries=1)
    assert result == "No papers found on Semantic Scholar after multiple query attempts."


def test_query_semantic_scholar_returns_error_string_on_http_error(monkeypatch):
    def fake_search(query, page_size=25):
        raise requests.HTTPError("429 Client Error")

    monkeypatch.setattr(literature, "_search_semantic_scholar", fake_search)
    result = literature.query_semantic_scholar("malaria")
    assert result == "Error querying Semantic Scholar: 429 Client Error"


def test_query_semantic_scholar_rejects_empty_query():
    result = literature.query_semantic_scholar("   ")
    assert result == "Error querying Semantic Scholar: Query must not be empty."


def test_literature_tool_description_exposes_semantic_scholar():
    from biomni.tool.tool_description.literature import description

    names = [tool["name"] for tool in description]
    assert "query_semantic_scholar" in names
    assert "query_pubmed" in names
