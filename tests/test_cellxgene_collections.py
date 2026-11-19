import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import requests

bio_module = types.ModuleType("Bio")
blast_module = types.ModuleType("Bio.Blast")
blast_module.NCBIWWW = object()
blast_module.NCBIXML = object()
seq_module = types.ModuleType("Bio.Seq")
seq_module.Seq = str
sys.modules.setdefault("Bio", bio_module)
sys.modules.setdefault("Bio.Blast", blast_module)
sys.modules.setdefault("Bio.Seq", seq_module)

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
    return json.loads((FIXTURES_DIR / "cellxgene_retina_collection.json").read_text(encoding="utf-8"))["response"]


def fixture_collection():
    return fixture_payload()[0]


def test_extract_cellxgene_collections_returns_list_payload():
    collections = database._extract_cellxgene_collections(fixture_payload())
    assert len(collections) == 1
    assert collections[0]["collection_id"] == "af893e86-8e9f-41f1-a474-ef05359b1fb7"


def test_extract_cellxgene_collections_returns_dict_payload():
    collections = database._extract_cellxgene_collections({"collections": fixture_payload()})
    assert len(collections) == 1
    assert collections[0]["name"].startswith("Single-cell transcriptomic atlas")


def test_get_cellxgene_labels_handles_ontology_terms_and_strings():
    values = [{"label": "Homo sapiens", "ontology_term_id": "NCBITaxon:9606"}, "nucleus"]
    assert database._get_cellxgene_labels(values) == ["Homo sapiens", "nucleus"]


def test_get_cellxgene_dataset_labels_deduplicates_values():
    labels = database._get_cellxgene_dataset_labels(fixture_collection(), "organism")
    assert labels == ["Homo sapiens"]


def test_cellxgene_collection_matches_query_terms():
    collection = fixture_collection()
    assert database._cellxgene_collection_matches(collection, ["human", "retina"])
    assert database._cellxgene_collection_matches(collection, ["homo", "sapiens"])
    assert database._cellxgene_collection_matches(collection, ["fovea"])
    assert not database._cellxgene_collection_matches(collection, ["zebrafish"])


def test_format_cellxgene_collection_returns_metadata_summary():
    output = database._format_cellxgene_collection(fixture_collection())
    assert "Collection: Single-cell transcriptomic atlas for adult human retina" in output
    assert "Collection ID: af893e86-8e9f-41f1-a474-ef05359b1fb7" in output
    assert "DOI: 10.1016/j.xgen.2023.100343" in output
    assert "Dataset Count: 1" in output
    assert "Organisms: Homo sapiens" in output
    assert "Tissues: fovea centralis, macula lutea proper, peripheral region of retina" in output
    assert "Assays: 10x 3' v3" in output
    assert "Diseases: normal" in output


def test_search_cellxgene_collections_calls_public_endpoint():
    session = Mock()
    session.get.return_value = Response(fixture_payload())
    collections = database._search_cellxgene_collections("human retina", session=session)
    assert len(collections) == 1
    session.get.assert_called_once_with(
        "https://api.cellxgene.cziscience.com/curation/v1/collections",
        timeout=30,
    )


def test_query_cellxgene_collections_returns_formatted_result(monkeypatch):
    monkeypatch.setattr(database, "_search_cellxgene_collections", lambda query: [fixture_collection()])
    result = database.query_cellxgene_collections("human retina", max_results=1)
    assert "Collection: Single-cell transcriptomic atlas for adult human retina" in result
    assert "Dataset Count: 1" in result


def test_query_cellxgene_collections_returns_no_results_message(monkeypatch):
    monkeypatch.setattr(database, "_search_cellxgene_collections", lambda query: [])
    result = database.query_cellxgene_collections("missing")
    assert result == "No CELLxGENE collections found."


def test_query_cellxgene_collections_rejects_empty_query():
    result = database.query_cellxgene_collections("   ")
    assert result == "Error querying CELLxGENE: Query must not be empty."


def test_query_cellxgene_collections_returns_error_string_on_http_error(monkeypatch):
    def raise_error(query):
        raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(database, "_search_cellxgene_collections", raise_error)
    result = database.query_cellxgene_collections("retina")
    assert result == "Error querying CELLxGENE: 500 Server Error"


def test_cellxgene_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_cellxgene_collections" in names
