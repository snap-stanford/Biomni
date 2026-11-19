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
    return json.loads((FIXTURES_DIR / "pubtator_asthma.json").read_text(encoding="utf-8"))["response"]


def fixture_document() -> dict:
    return fixture_payload()["PubTator3"][0]


def test_extract_documents_returns_pubtator_documents():
    documents = literature._extract_pubtator_documents(fixture_payload())
    assert len(documents) == 1
    assert documents[0]["id"] == "28483577"


def test_get_passage_text_returns_requested_passage():
    document = fixture_document()
    assert literature._get_pubtator_passage_text(document, "title").startswith("Formoterol")
    assert literature._get_pubtator_passage_text(document, "missing") == ""


def test_extract_annotations_adds_passage_type():
    annotations = literature._extract_pubtator_annotations(fixture_document())
    assert len(annotations) == 3
    assert annotations[0]["passage_type"] == "title"
    assert annotations[-1]["passage_type"] == "abstract"


def test_format_annotation_uses_normalized_fields():
    annotation = literature._extract_pubtator_annotations(fixture_document())[0]
    output = literature._format_pubtator_annotation(annotation)
    assert "Formoterol (Chemical)" in output
    assert "Normalized: Formoterol Fumarate" in output
    assert "ID: MESH:D000068759" in output
    assert "Database: ncbi_mesh" in output
    assert "Passage: title" in output


def test_format_document_limits_annotations():
    output = literature._format_pubtator_document(fixture_document(), max_annotations=2)
    assert "PMID: 28483577" in output
    assert "PMCID: PMC5424182" in output
    assert "Title: Formoterol and fluticasone" in output
    assert "Journal: Biochim Biophys Acta Mol Basis Dis" in output
    assert "Year: 2017" in output
    assert "Annotations Returned: 2 of 3" in output
    assert "Formoterol (Chemical)" in output
    assert "fluticasone propionate (Chemical)" in output
    assert "asthma (Disease)" not in output


def test_search_pubtator_calls_export_endpoint():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    documents = literature._search_pubtator("28483577", session=session)
    assert documents[0]["id"] == "28483577"
    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["params"] == {"pmids": "28483577"}


def test_query_pubtator_returns_formatted_annotations(monkeypatch):
    monkeypatch.setattr(literature, "_search_pubtator", lambda pmid: [fixture_document()])
    result = literature.query_pubtator("28483577", max_annotations=2)
    assert "Title: Formoterol and fluticasone" in result
    assert "Annotations Returned: 2 of 3" in result
    assert "Formoterol (Chemical)" in result


def test_query_pubtator_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(literature, "_search_pubtator", lambda pmid: [])
    result = literature.query_pubtator("12345678")
    assert result == "No PubTator annotations found."


def test_query_pubtator_rejects_empty_pmid():
    result = literature.query_pubtator("   ")
    assert result == "Error querying PubTator: PMID must not be empty."


def test_query_pubtator_rejects_non_numeric_pmid():
    result = literature.query_pubtator("PMC5424182")
    assert result == "Error querying PubTator: PMID must contain only digits."


def test_query_pubtator_returns_error_string_on_http_error(monkeypatch):
    def raise_error(pmid):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(literature, "_search_pubtator", raise_error)
    result = literature.query_pubtator("28483577")
    assert result == "Error querying PubTator: 500 Server Error"


def test_pubtator_tool_description_is_registered():
    names = {entry["name"] for entry in literature_description.description}
    assert "query_pubtator" in names
