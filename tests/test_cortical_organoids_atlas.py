from pathlib import Path
from unittest.mock import Mock

from biomni.tool import database
from biomni.tool.tool_description import database as database_description

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self.payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def fixture_html():
    return (FIXTURES_DIR / "cortical_organoids_atlas.html").read_text(encoding="utf-8")


def fixture_clusters():
    return [
        "23days scRNA-seq",
        "1month scRNA-seq",
        "3month scATAC-seq",
        "SHARE-seq RNA",
        "SHARE-seq ATAC",
        "Human Fetal Tissue",
    ]


def test_extract_cortical_organoids_study_metadata():
    metadata = database._extract_cortical_organoids_study_metadata(fixture_html())

    assert metadata["accession"] == "SCP1756"
    assert metadata["title"] == "Cortical Organoids Atlas"
    assert metadata["cell_count"] == "777092 total cells"
    assert metadata["gene_count"] == "34211 genes"
    assert metadata["pmid"] == "36179669"
    assert "longitudinal molecular programs" in metadata["summary"]


def test_fetch_cortical_organoids_atlas_metadata_uses_public_page_and_cluster_api():
    session = Mock()
    session.get.side_effect = [
        FakeResponse(text=fixture_html()),
        FakeResponse(payload=fixture_clusters()),
    ]

    metadata, clusters = database._fetch_cortical_organoids_atlas_metadata(session=session)

    assert metadata["title"] == "Cortical Organoids Atlas"
    assert clusters == fixture_clusters()
    assert session.get.call_args_list[0].args[0] == database.CORTICAL_ORGANOIDS_ATLAS_URL
    assert session.get.call_args_list[1].args[0] == database.CORTICAL_ORGANOIDS_CLUSTERS_URL


def test_filter_cortical_organoids_clusters_matches_all_terms():
    clusters = fixture_clusters()

    assert database._filter_cortical_organoids_clusters(clusters, "scRNA") == ["23days scRNA-seq", "1month scRNA-seq"]
    assert database._filter_cortical_organoids_clusters(clusters, "SHARE ATAC") == ["SHARE-seq ATAC"]
    assert database._filter_cortical_organoids_clusters(clusters, "") == clusters


def test_summarize_cortical_organoids_cluster_annotations_counts_top_labels():
    session = Mock()
    session.get.return_value = FakeResponse(payload={"data": {"annotations": ["aRG", "aRG", "Neuron"]}})

    annotations, top_annotations = database._summarize_cortical_organoids_cluster_annotations(
        "23days scRNA-seq", session=session
    )

    assert annotations == ["aRG", "aRG", "Neuron"]
    assert top_annotations == [("aRG", 2), ("Neuron", 1)]


def test_format_cortical_organoids_atlas_returns_summary():
    metadata = database._extract_cortical_organoids_study_metadata(fixture_html())
    output = database._format_cortical_organoids_atlas(metadata, fixture_clusters(), ["SHARE-seq ATAC"])

    assert "Study: Cortical Organoids Atlas" in output
    assert "Accession: SCP1756" in output
    assert "Cells: 777092 total cells" in output
    assert "Public Cluster Count: 6" in output
    assert "- SHARE-seq ATAC" in output


def test_query_cortical_organoids_atlas_returns_filtered_result(monkeypatch):
    metadata = database._extract_cortical_organoids_study_metadata(fixture_html())
    monkeypatch.setattr(database, "_fetch_cortical_organoids_atlas_metadata", lambda: (metadata, fixture_clusters()))

    result = database.query_cortical_organoids_atlas("fetal")

    assert "Study: Cortical Organoids Atlas" in result
    assert "- Human Fetal Tissue" in result
    assert "- SHARE-seq ATAC" not in result


def test_query_cortical_organoids_atlas_returns_error_string_on_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("portal unavailable")

    monkeypatch.setattr(database, "_fetch_cortical_organoids_atlas_metadata", raise_error)

    result = database.query_cortical_organoids_atlas()

    assert result == "Error querying Cortical Organoids Atlas: portal unavailable"


def test_cortical_organoids_atlas_tool_description_is_registered():
    names = {entry["name"] for entry in database_description.description}
    assert "query_cortical_organoids_atlas" in names
