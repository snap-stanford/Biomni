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
    return json.loads((FIXTURES_DIR / "alphafold_p69905_prediction.json").read_text(encoding="utf-8"))["response"]


def fixture_record():
    return fixture_payload()[0]


def test_normalize_alphafold_uniprot_ids_accepts_string_and_list():
    assert database._normalize_alphafold_uniprot_ids("P69905, P68871\nP05067") == ["P69905", "P68871", "P05067"]
    assert database._normalize_alphafold_uniprot_ids(["P69905", " ", "P05067"]) == ["P69905", "P05067"]


def test_extract_alphafold_prediction_record_returns_first_list_record():
    assert database._extract_alphafold_prediction_record(fixture_payload())["uniprotAccession"] == "P69905"
    assert database._extract_alphafold_prediction_record(fixture_record())["uniprotAccession"] == "P69905"
    assert database._extract_alphafold_prediction_record([]) == {}


def test_format_alphafold_bulk_record_returns_metadata_summary():
    output = database._format_alphafold_bulk_record("P69905", fixture_record())
    assert "UniProt ID: P69905" in output
    assert "Entry ID: AF-P69905-F1" in output
    assert "Gene: HBA1" in output
    assert "Description: Hemoglobin subunit alpha" in output
    assert "Organism: Homo sapiens" in output
    assert "Sequence Range: 1-142" in output
    assert "Latest Version: 6" in output
    assert "Model Created: 2025-08-01T00:00:00Z" in output
    assert "Sequence Checksum: 6077c452d1dc6151040b2b179e2294c7" in output
    assert "Mean pLDDT: 98.06" in output
    assert "PDB URL: https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.pdb" in output


def test_format_alphafold_bulk_record_handles_missing_record():
    output = database._format_alphafold_bulk_record("MISSING", {})
    assert output == "UniProt ID: MISSING\nStatus: No AlphaFold prediction found."


def test_search_alphafold_bulk_metadata_calls_prediction_endpoint():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    records = database._search_alphafold_bulk_metadata(["P69905"], session=session)
    assert records[0][0] == "P69905"
    assert records[0][1]["entryId"] == "AF-P69905-F1"
    session.get.assert_called_once_with("https://alphafold.ebi.ac.uk/api/prediction/P69905", timeout=30)


def test_search_alphafold_bulk_metadata_records_404_as_missing():
    session = Mock()
    session.get.return_value = Response({"error": "not found"}, status_code=404)
    records = database._search_alphafold_bulk_metadata(["MISSING"], session=session)
    assert records == [("MISSING", {})]


def test_query_alphafold_bulk_metadata_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(database, "_search_alphafold_bulk_metadata", lambda uniprot_ids: [("P69905", fixture_record())])
    result = database.query_alphafold_bulk_metadata("P69905", max_results=1)
    assert "UniProt ID: P69905" in result
    assert "Sequence Checksum: 6077c452d1dc6151040b2b179e2294c7" in result


def test_query_alphafold_bulk_metadata_passes_limited_ids(monkeypatch):
    captured = {}

    def fake_search(uniprot_ids):
        captured["uniprot_ids"] = uniprot_ids
        return [("P69905", fixture_record())]

    monkeypatch.setattr(database, "_search_alphafold_bulk_metadata", fake_search)
    database.query_alphafold_bulk_metadata("P69905,P68871,P05067", max_results=2)
    assert captured["uniprot_ids"] == ["P69905", "P68871"]


def test_query_alphafold_bulk_metadata_rejects_empty_ids():
    result = database.query_alphafold_bulk_metadata(" , ")
    assert result == "Error querying AlphaFold bulk metadata: UniProt IDs must not be empty."


def test_query_alphafold_bulk_metadata_returns_error_string_on_http_error(monkeypatch):
    def raise_error(uniprot_ids):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(database, "_search_alphafold_bulk_metadata", raise_error)
    result = database.query_alphafold_bulk_metadata("P69905")
    assert result == "Error querying AlphaFold bulk metadata: 500 Server Error"


def test_alphafold_bulk_metadata_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_alphafold_bulk_metadata" in names
