import json
from pathlib import Path
from unittest.mock import Mock

import requests
from biomni.tool import literature
from biomni.tool.tool_description import literature as literature_description

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
    return json.loads((FIXTURES_DIR / "biorxiv_zebrafish.json").read_text(encoding="utf-8"))["response"]


def fixture_record() -> dict:
    return fixture_payload()["collection"][0]


def test_extract_records_returns_collection_records():
    records = literature._extract_biorxiv_medrxiv_records(fixture_payload())
    assert len(records) == 1
    assert records[0]["doi"] == "10.64898/2026.08.28.747819"


def test_get_total_returns_provider_total():
    assert literature._get_biorxiv_medrxiv_total(fixture_payload()) == 1478
    assert literature._get_biorxiv_medrxiv_total({"messages": [{"total": "not-a-number"}]}) is None


def test_format_record_uses_preprint_fields():
    output = literature._format_biorxiv_medrxiv_record(fixture_record())
    assert "Title: Zebrafish larval nitrogen excretion" in output
    assert "Journal: bioRxiv" in output
    assert "DOI: 10.64898/2026.08.28.747819" in output
    assert "Posted: 2026-09-01" in output
    assert "Category: physiology" in output
    assert "URL: https://www.biorxiv.org/content/early/2026/09/01/2026.08.28.747819.source.xml" in output


def test_record_matches_query_terms_and_category():
    record = fixture_record()
    assert literature._biorxiv_medrxiv_record_matches(record, ["zebrafish", "rhesus"], "physiology")
    assert literature._biorxiv_medrxiv_record_matches(record, ["zebrafish"], "all")
    assert not literature._biorxiv_medrxiv_record_matches(record, ["malaria"], "all")
    assert not literature._biorxiv_medrxiv_record_matches(record, ["zebrafish"], "genomics")


def test_search_biorxiv_medrxiv_filters_records():
    session = Mock()
    session.get.side_effect = [Response(fixture_payload()), Response({"collection": []})]
    records = literature._search_biorxiv_medrxiv(
        "zebrafish rhesus",
        server="biorxiv",
        max_papers=1,
        days_back=7,
        session=session,
    )
    assert records[0]["doi"] == "10.64898/2026.08.28.747819"
    session.get.assert_called_once()


def test_search_biorxiv_medrxiv_paginates_until_match():
    session = Mock()
    miss = fixture_payload()
    miss["collection"] = [{**fixture_record(), "title": "Unrelated preprint", "abstract": "No matching terms"}]
    session.get.side_effect = [Response(miss), Response(fixture_payload())]
    records = literature._search_biorxiv_medrxiv(
        "zebrafish rhesus",
        server="biorxiv",
        max_papers=1,
        days_back=7,
        session=session,
    )
    assert records[0]["title"].startswith("Zebrafish")
    assert session.get.call_count == 2


def test_search_biorxiv_medrxiv_stops_at_provider_total():
    session = Mock()
    miss = fixture_payload()
    miss["messages"][0]["total"] = "1"
    miss["collection"] = [{**fixture_record(), "title": "Unrelated preprint", "abstract": "No matching terms"}]
    session.get.return_value = Response(miss)
    records = literature._search_biorxiv_medrxiv(
        "zebrafish rhesus",
        server="biorxiv",
        max_papers=1,
        days_back=7,
        session=session,
    )
    assert records == []
    session.get.assert_called_once()


def test_query_biorxiv_medrxiv_returns_formatted_results(monkeypatch):
    monkeypatch.setattr(literature, "_search_biorxiv_medrxiv", lambda *args, **kwargs: [fixture_record()])
    result = literature.query_biorxiv_medrxiv("zebrafish rhesus", max_papers=1, days_back=7)
    assert "Title: Zebrafish larval nitrogen excretion" in result
    assert "Journal: bioRxiv" in result
    assert "Category: physiology" in result


def test_query_biorxiv_medrxiv_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_biorxiv_medrxiv", lambda *args, **kwargs: [])
    result = literature.query_biorxiv_medrxiv("missing", max_papers=1)
    assert result == "No preprints found on bioRxiv/medRxiv."


def test_query_biorxiv_medrxiv_rejects_empty_query():
    result = literature.query_biorxiv_medrxiv("   ")
    assert result == "Error querying bioRxiv/medRxiv: Query must not be empty."


def test_query_biorxiv_medrxiv_rejects_unknown_server():
    result = literature.query_biorxiv_medrxiv("zebrafish", server="arxiv")
    assert result == 'Error querying bioRxiv/medRxiv: Server must be "biorxiv" or "medrxiv".'


def test_query_biorxiv_medrxiv_returns_error_string_on_http_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(literature, "_search_biorxiv_medrxiv", raise_error)
    result = literature.query_biorxiv_medrxiv("zebrafish")
    assert result == "Error querying bioRxiv/medRxiv: 500 Server Error"


def test_biorxiv_medrxiv_tool_description_is_registered():
    names = {entry["name"] for entry in literature_description.description}
    assert "query_biorxiv_medrxiv" in names
