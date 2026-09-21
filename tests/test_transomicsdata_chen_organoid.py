from pathlib import Path
from unittest.mock import Mock

from biomni.tool import database
from biomni.tool.tool_description import database as database_description

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def fixture_text(name):
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_parse_transomics_table_reads_manifest():
    rows = database._parse_transomics_table(fixture_text("transomicsdata_manifest.txt"), delimiter="\t")

    assert rows[0]["Title"] == "chen-organoid"
    assert rows[0]["Species"] == "human"
    assert rows[0]["RDataPath"] == "TransOmicsData/0.99.0/chen-organoid"


def test_fetch_transomicsdata_chen_organoid_metadata_uses_package_source_urls():
    session = Mock()
    session.get.side_effect = [
        FakeResponse(fixture_text("transomicsdata_manifest.txt")),
        FakeResponse(fixture_text("transomicsdata_chen_organoid_metadata.csv")),
    ]

    dataset, assays = database._fetch_transomicsdata_chen_organoid_metadata(session=session)

    assert dataset["Title"] == "chen-organoid"
    assert len(assays) == 4
    assert assays[0]["Title"] == "Chen organoid phosphoproteome"
    assert session.get.call_args_list[0].args[0] == database.TRANSOMICS_CHEN_MANIFEST_URL
    assert session.get.call_args_list[1].args[0] == database.TRANSOMICS_CHEN_METADATA_URL


def test_filter_transomics_chen_assays_accepts_aliases():
    assays = database._parse_transomics_table(fixture_text("transomicsdata_chen_organoid_metadata.csv"))

    assert len(database._filter_transomics_chen_assays(assays, "all")) == 4
    assert database._filter_transomics_chen_assays(assays, "scrna")[0]["RDataPath"].endswith("sctranscriptome.rds")
    assert database._filter_transomics_chen_assays(assays, "protein")[0]["RDataPath"].endswith("proteome.rds")


def test_format_transomicsdata_chen_organoid_includes_experimenthub_paths():
    dataset = database._parse_transomics_table(fixture_text("transomicsdata_manifest.txt"), delimiter="\t")[0]
    assays = database._parse_transomics_table(fixture_text("transomicsdata_chen_organoid_metadata.csv"))[:1]

    output = database._format_transomicsdata_chen_organoid(dataset, assays)

    assert "Dataset: chen-organoid" in output
    assert "Omics: phosphoproteome, proteome, transcriptome, single-cell transcriptome" in output
    assert "Title: Chen organoid phosphoproteome" in output
    assert "RDataPath: TransOmicsData/0.99.0/chen-organoid/phosphoproteome.rds" in output
    assert (
        'matches <- records[records$rdatapath == "TransOmicsData/0.99.0/chen-organoid/phosphoproteome.rds"]' in output
    )


def test_query_transomicsdata_chen_organoid_returns_filtered_result(monkeypatch):
    dataset = database._parse_transomics_table(fixture_text("transomicsdata_manifest.txt"), delimiter="\t")[0]
    assays = database._parse_transomics_table(fixture_text("transomicsdata_chen_organoid_metadata.csv"))
    monkeypatch.setattr(database, "_fetch_transomicsdata_chen_organoid_metadata", lambda: (dataset, assays))

    result = database.query_transomicsdata_chen_organoid(assay="transcriptome", include_load_code=False)

    assert "Title: Chen organoid transcriptome" in result
    assert "Title: Chen organoid phosphoproteome" not in result
    assert "ExperimentHub load code" not in result


def test_query_transomicsdata_chen_organoid_returns_error_string_on_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(database, "_fetch_transomicsdata_chen_organoid_metadata", raise_error)

    result = database.query_transomicsdata_chen_organoid()

    assert result == "Error querying TransOmicsData chen-organoid metadata: metadata unavailable"


def test_transomicsdata_chen_organoid_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_transomicsdata_chen_organoid" in names
