import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[1]


@pytest.fixture
def genomics_module(monkeypatch):
    """Load genomics.py without requiring Biomni's full scientific environment."""
    for module_name in ("esm", "gget", "gseapy", "numpy", "pandas", "requests", "scanpy", "torch"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    pybiomart = ModuleType("pybiomart")
    pybiomart.Dataset = object
    monkeypatch.setitem(sys.modules, "pybiomart", pybiomart)

    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **_kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    biomni_module = ModuleType("biomni")
    biomni_module.__path__ = []
    biomni_llm = ModuleType("biomni.llm")
    biomni_llm.get_llm = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "biomni", biomni_module)
    monkeypatch.setitem(sys.modules, "biomni.llm", biomni_llm)

    spec = importlib.util.spec_from_file_location("genomics_under_test", REPO_ROOT / "biomni/tool/genomics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_conversion_preserves_order_duplicates_all_ids_and_unresolved(genomics_module, monkeypatch):
    payload = [
        {
            "query": "RNH1",
            "symbol": "RNH1",
            "taxid": 9606,
            "ensembl": [{"gene": "ENSG00000023191"}, {"gene": "ENSG00000276230"}],
        },
        {"query": "BRCA1", "symbol": "BRCA1", "taxid": 9606, "ensembl": {"gene": "ENSG00000012048"}},
        {"query": "NOT_A_GENE", "notfound": True},
    ]
    calls = []

    def fake_post(url, data, timeout):
        calls.append((url, data, timeout))
        return FakeResponse(payload)

    monkeypatch.setattr(genomics_module.requests, "post", fake_post, raising=False)

    result = genomics_module.convert_gene_symbols_to_ensembl_ids([" BRCA1 ", "RNH1", "NOT_A_GENE", "BRCA1"], " human ")

    assert result == {
        "species": "human",
        "resolved_count": 3,
        "unresolved_count": 1,
        "results": [
            {
                "query": "BRCA1",
                "matched_symbols": ["BRCA1"],
                "ensembl_ids": ["ENSG00000012048"],
                "taxids": [9606],
                "resolved": True,
            },
            {
                "query": "RNH1",
                "matched_symbols": ["RNH1"],
                "ensembl_ids": ["ENSG00000023191", "ENSG00000276230"],
                "taxids": [9606],
                "resolved": True,
            },
            {
                "query": "NOT_A_GENE",
                "matched_symbols": [],
                "ensembl_ids": [],
                "taxids": [],
                "resolved": False,
            },
            {
                "query": "BRCA1",
                "matched_symbols": ["BRCA1"],
                "ensembl_ids": ["ENSG00000012048"],
                "taxids": [9606],
                "resolved": True,
            },
        ],
    }
    assert calls == [
        (
            "https://mygene.info/v3/query",
            {
                "q": "BRCA1,RNH1,NOT_A_GENE",
                "scopes": "symbol",
                "fields": "ensembl.gene,symbol,taxid",
                "species": "human",
            },
            30,
        )
    ]


def test_conversion_batches_more_than_1000_unique_symbols(genomics_module, monkeypatch):
    batch_sizes = []

    def fake_post(_url, data, timeout):
        assert timeout == 30
        symbols = data["q"].split(",")
        batch_sizes.append(len(symbols))
        return FakeResponse(
            [
                {
                    "query": symbol,
                    "symbol": symbol,
                    "taxid": 9606,
                    "ensembl": {"gene": f"ENSG{index:011d}"},
                }
                for index, symbol in enumerate(symbols)
            ]
        )

    monkeypatch.setattr(genomics_module.requests, "post", fake_post, raising=False)
    symbols = [f"GENE{index}" for index in range(1001)]

    result = genomics_module.convert_gene_symbols_to_ensembl_ids(symbols, "9606")

    assert batch_sizes == [1000, 1]
    assert len(result["results"]) == 1001
    assert result["resolved_count"] == 1001


@pytest.mark.parametrize(
    ("gene_symbols", "species", "error"),
    [
        ([], "human", "non-empty list"),
        ("BRCA1", "human", "non-empty list"),
        (["BRCA1", ""], "human", r"gene_symbols\[1\]"),
        (["BRCA1", 53], "human", r"gene_symbols\[1\]"),
        (["BRCA1,TP53"], "human", "must not contain a comma"),
        (["BRCA1"], "", "species must be a non-empty string"),
    ],
)
def test_conversion_validates_inputs(genomics_module, gene_symbols, species, error):
    with pytest.raises(ValueError, match=error):
        genomics_module.convert_gene_symbols_to_ensembl_ids(gene_symbols, species)


def test_conversion_rejects_unexpected_api_response(genomics_module, monkeypatch):
    monkeypatch.setattr(
        genomics_module.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse({"hits": []}),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="expected a list"):
        genomics_module.convert_gene_symbols_to_ensembl_ids(["BRCA1"], "human")


def test_tool_description_registers_function_and_required_parameters():
    namespace = runpy.run_path(REPO_ROOT / "biomni/tool/tool_description/genomics.py")
    schema = next(item for item in namespace["description"] if item["name"] == "convert_gene_symbols_to_ensembl_ids")

    assert [parameter["name"] for parameter in schema["required_parameters"]] == ["gene_symbols", "species"]
    assert schema["optional_parameters"] == []
