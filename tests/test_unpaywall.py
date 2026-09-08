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
    return json.loads((FIXTURES_DIR / "unpaywall_bioconductor.json").read_text(encoding="utf-8"))["response"]


def test_format_record_uses_open_access_fields():
    record = {
        "doi": "10.1186/gb-2004-5-10-r80",
        "title": "Bioconductor: open software development for computational biology and bioinformatics",
        "journal_name": "Genome Biology",
        "is_oa": True,
        "oa_status": "gold",
        "best_oa_location": {
            "url_for_pdf": "https://genomebiology.biomedcentral.com/counter/pdf/10.1186/gb-2004-5-10-r80",
            "license": "cc-by",
        },
    }
    output = literature._format_unpaywall_record(record)
    assert "Title: Bioconductor: open software development" in output
    assert "DOI: 10.1186/gb-2004-5-10-r80" in output
    assert "Journal: Genome Biology" in output
    assert "Open Access: Yes" in output
    assert "OA Status: gold" in output
    assert "Best OA URL: https://genomebiology.biomedcentral.com/counter/pdf/10.1186/gb-2004-5-10-r80" in output
    assert "License: cc-by" in output


def test_search_unpaywall_returns_record():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    record = literature._search_unpaywall("10.1186/gb-2004-5-10-r80", "test@example.org", session=session)
    assert record["doi"] == "10.1186/gb-2004-5-10-r80"
    session.get.assert_called_once()


def test_query_unpaywall_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(literature, "_search_unpaywall", lambda doi, email: fixture_payload())
    result = literature.query_unpaywall("10.1186/gb-2004-5-10-r80", email="test@example.org")
    assert "Title: Bioconductor: open software development" in result
    assert "Open Access: Yes" in result
    assert "OA Status: gold" in result
    assert "License: cc-by" in result


def test_query_unpaywall_uses_env_email(monkeypatch):
    calls = []

    def fake_search(doi, email):
        calls.append((doi, email))
        return fixture_payload()

    monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.org")
    monkeypatch.setattr(literature, "_search_unpaywall", fake_search)
    literature.query_unpaywall("10.1186/gb-2004-5-10-r80")
    assert calls == [("10.1186/gb-2004-5-10-r80", "test@example.org")]


def test_query_unpaywall_requires_email(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    result = literature.query_unpaywall("10.1186/gb-2004-5-10-r80")
    assert result == "Error querying Unpaywall: Set UNPAYWALL_EMAIL or pass email."


def test_query_unpaywall_rejects_empty_doi():
    result = literature.query_unpaywall("   ", email="test@example.org")
    assert result == "Error querying Unpaywall: DOI must not be empty."


def test_query_unpaywall_returns_no_record_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_unpaywall", lambda doi, email: {})
    result = literature.query_unpaywall("10.1234/missing", email="test@example.org")
    assert result == "No record found on Unpaywall."


def test_query_unpaywall_returns_error_string_on_http_error(monkeypatch):
    def fake_search(doi, email):
        raise requests.HTTPError("404 Client Error")

    monkeypatch.setattr(literature, "_search_unpaywall", fake_search)
    result = literature.query_unpaywall("10.1234/missing", email="test@example.org")
    assert result == "Error querying Unpaywall: 404 Client Error"


def test_literature_tool_description_exposes_unpaywall():
    from biomni.tool.tool_description.literature import description

    names = [tool["name"] for tool in description]
    assert "query_unpaywall" in names
    assert "query_pubmed" in names
