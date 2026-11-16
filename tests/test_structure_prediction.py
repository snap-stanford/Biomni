"""Tests for the structure prediction module."""

import os
import tempfile
from unittest import mock

from biomni.tool.structure_prediction import (
    _generate_job_id,
    _validate_nucleic_acid,
    _validate_sequence,
    analyze_structure_confidence,
    batch_predict_structures,
    download_structure,
    get_job_status,
    predict_protein_complex,
    predict_protein_ligand_complex,
    predict_protein_nucleic_acid_complex,
    predict_protein_structure,
)


class TestSequenceValidation:
    """Tests for sequence validation functions."""

    def test_validate_valid_protein_sequence(self):
        """Valid protein sequences should pass validation."""
        seq = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        is_valid, result = _validate_sequence(seq)
        assert is_valid is True
        assert result == seq.upper()

    def test_validate_sequence_with_whitespace(self):
        """Sequences with whitespace should be cleaned."""
        seq = "MKFL ILLF\nNILC LFPV"
        is_valid, result = _validate_sequence(seq)
        assert is_valid is True
        assert " " not in result
        assert "\n" not in result

    def test_validate_sequence_lowercase(self):
        """Lowercase sequences should be converted to uppercase."""
        seq = "mkflillfnilclfpvlaadnh"
        is_valid, result = _validate_sequence(seq)
        assert is_valid is True
        assert result == seq.upper()

    def test_validate_sequence_invalid_characters(self):
        """Sequences with invalid characters should fail."""
        seq = "MKFLILLFNILC123LFPV"
        is_valid, result = _validate_sequence(seq)
        assert is_valid is False
        assert "Invalid characters" in result

    def test_validate_sequence_too_short(self):
        """Sequences shorter than 10 residues should fail."""
        seq = "MKFLI"
        is_valid, result = _validate_sequence(seq)
        assert is_valid is False
        assert "too short" in result

    def test_validate_sequence_too_long(self):
        """Sequences longer than 2000 residues should fail."""
        seq = "M" * 2001
        is_valid, result = _validate_sequence(seq)
        assert is_valid is False
        assert "too long" in result

    def test_validate_dna_sequence(self):
        """Valid DNA sequences should pass validation."""
        seq = "ATCGATCGATCG"
        is_valid, result = _validate_nucleic_acid(seq, "DNA")
        assert is_valid is True
        assert result == seq

    def test_validate_rna_sequence(self):
        """Valid RNA sequences should pass validation."""
        seq = "AUCGAUCGAUCG"
        is_valid, result = _validate_nucleic_acid(seq, "RNA")
        assert is_valid is True
        assert result == seq

    def test_validate_dna_invalid_base(self):
        """DNA with U base should fail."""
        seq = "ATCGUATCG"
        is_valid, result = _validate_nucleic_acid(seq, "DNA")
        assert is_valid is False
        assert "Invalid characters" in result

    def test_validate_nucleic_acid_too_short(self):
        """Nucleic acid sequences shorter than 5 bases should fail."""
        seq = "ATCG"
        is_valid, result = _validate_nucleic_acid(seq, "DNA")
        assert is_valid is False
        assert "too short" in result


class TestJobIdGeneration:
    """Tests for job ID generation."""

    def test_generate_job_id_deterministic(self):
        """Same sequences should produce same job ID."""
        seqs = ["MKFLILLFNILCLFPV", "MALTEVNPKKYIPGTK"]
        id1 = _generate_job_id(seqs)
        id2 = _generate_job_id(seqs)
        assert id1 == id2

    def test_generate_job_id_different_for_different_seqs(self):
        """Different sequences should produce different job IDs."""
        id1 = _generate_job_id(["MKFLILLFNILCLFPV"])
        id2 = _generate_job_id(["MALTEVNPKKYIPGTK"])
        assert id1 != id2


class TestPredictProteinStructure:
    """Tests for single protein structure prediction."""

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_protein_success(self, mock_request):
        """Successful prediction should return structure data."""
        mock_request.side_effect = [
            {"success": True, "result": {"job_id": "test123"}},
            {
                "success": True,
                "result": {"status": "completed", "pdb_content": "ATOM...", "plddt_mean": 85.5, "ptm": 0.9},
            },
        ]

        result = predict_protein_structure("MKFLILLFNILCLFPVLAADNH")

        assert result["success"] is True
        assert result["job_id"] == "test123"
        assert "pdb_content" in result
        assert result["confidence"]["plddt_mean"] == 85.5

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_protein_saves_file(self, mock_request):
        """Prediction should save PDB file when path provided."""
        mock_request.side_effect = [
            {"success": True, "result": {"job_id": "test123"}},
            {"success": True, "result": {"status": "completed", "pdb_content": "ATOM 1 CA ALA", "plddt_mean": 85.5}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test.pdb")
            result = predict_protein_structure("MKFLILLFNILCLFPVLAADNH", save_path=save_path)

            assert result["success"] is True
            assert os.path.exists(save_path)
            with open(save_path) as f:
                assert f.read() == "ATOM 1 CA ALA"

    def test_predict_protein_invalid_sequence(self):
        """Invalid sequence should return error without API call."""
        result = predict_protein_structure("INVALID123")
        assert result["success"] is False
        assert "Invalid characters" in result["error"]

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_protein_no_wait(self, mock_request):
        """No-wait mode should return job ID immediately."""
        mock_request.return_value = {"success": True, "result": {"job_id": "test123"}}

        result = predict_protein_structure("MKFLILLFNILCLFPVLAADNH", wait_for_result=False)

        assert result["success"] is True
        assert result["status"] == "submitted"
        assert result["job_id"] == "test123"
        mock_request.assert_called_once()


class TestPredictProteinComplex:
    """Tests for protein complex prediction."""

    def test_predict_complex_single_sequence_fails(self):
        """Complex prediction with single sequence should fail."""
        result = predict_protein_complex(["MKFLILLFNILCLFPVLAADNH"])
        assert result["success"] is False
        assert "At least 2 sequences" in result["error"]

    def test_predict_complex_empty_list_fails(self):
        """Complex prediction with empty list should fail."""
        result = predict_protein_complex([])
        assert result["success"] is False

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_complex_success(self, mock_request):
        """Successful complex prediction should return interface data."""
        mock_request.side_effect = [
            {"success": True, "result": {"job_id": "complex123"}},
            {
                "success": True,
                "result": {
                    "status": "completed",
                    "pdb_content": "ATOM...",
                    "plddt_mean": 80.0,
                    "iptm": 0.85,
                    "interface_contacts": [{"chain_a": "A", "chain_b": "B", "residue_a": 10, "residue_b": 25}],
                },
            },
        ]

        result = predict_protein_complex(
            ["MKFLILLFNILCLFPVLAADNH", "MALTEVNPKKYIPGTKMIFAG"], chain_names=["Receptor", "Ligand"]
        )

        assert result["success"] is True
        assert result["confidence"]["iptm"] == 0.85
        assert len(result["interface_contacts"]) > 0


class TestPredictProteinNucleicAcid:
    """Tests for protein-nucleic acid complex prediction."""

    def test_predict_invalid_na_type(self):
        """Invalid nucleic acid type should fail."""
        result = predict_protein_nucleic_acid_complex(["MKFLILLFNILCLFPVLAADNH"], "ATCGATCG", "INVALID")
        assert result["success"] is False
        assert "DNA" in result["error"] or "RNA" in result["error"]

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_protein_dna_success(self, mock_request):
        """Successful protein-DNA prediction should work."""
        mock_request.side_effect = [
            {"success": True, "result": {"job_id": "dna123"}},
            {"success": True, "result": {"status": "completed", "pdb_content": "ATOM...", "plddt_mean": 75.0}},
        ]

        result = predict_protein_nucleic_acid_complex(["MKFLILLFNILCLFPVLAADNH"], "ATCGATCGATCGATCG", "DNA")

        assert result["success"] is True
        assert result["job_id"] == "dna123"


class TestPredictProteinLigand:
    """Tests for protein-ligand complex prediction."""

    def test_predict_empty_smiles_fails(self):
        """Empty SMILES string should fail."""
        result = predict_protein_ligand_complex("MKFLILLFNILCLFPVLAADNH", "")
        assert result["success"] is False
        assert "SMILES" in result["error"]

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_predict_protein_ligand_success(self, mock_request):
        """Successful protein-ligand prediction should return binding data."""
        mock_request.side_effect = [
            {"success": True, "result": {"job_id": "ligand123"}},
            {
                "success": True,
                "result": {
                    "status": "completed",
                    "pdb_content": "ATOM...",
                    "binding_residues": [45, 67, 89, 112],
                    "predicted_affinity": -8.5,
                },
            },
        ]

        result = predict_protein_ligand_complex("MKFLILLFNILCLFPVLAADNH", "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC")

        assert result["success"] is True
        assert len(result["binding_site"]) > 0
        assert result["binding_affinity"] == -8.5


class TestGetJobStatus:
    """Tests for job status checking."""

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_get_job_status_running(self, mock_request):
        """Running job should return pending status."""
        mock_request.return_value = {"success": True, "result": {"status": "running", "progress": 45}}

        result = get_job_status("test123")

        assert result["success"] is True
        assert result["status"] == "running"
        assert result["progress"] == 45

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_get_job_status_completed(self, mock_request):
        """Completed job should include result data."""
        mock_request.return_value = {"success": True, "result": {"status": "completed", "pdb_content": "ATOM..."}}

        result = get_job_status("test123")

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["result"] is not None


class TestDownloadStructure:
    """Tests for structure file download."""

    def test_download_invalid_format(self):
        """Invalid format should fail."""
        result = download_structure("test123", "output.xyz", "xyz")
        assert result["success"] is False
        assert "pdb" in result["error"] or "cif" in result["error"]

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_download_structure_success(self, mock_request):
        """Successful download should save file."""
        mock_request.return_value = {"success": True, "result": {"pdb_content": "ATOM 1 CA ALA A 1"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "structure.pdb")
            result = download_structure("test123", output_path, "pdb")

            assert result["success"] is True
            assert os.path.exists(output_path)
            assert result["format"] == "pdb"


class TestAnalyzeStructureConfidence:
    """Tests for structure confidence analysis."""

    def test_analyze_nonexistent_file(self):
        """Nonexistent file should fail."""
        result = analyze_structure_confidence("/nonexistent/file.pdb")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_analyze_valid_pdb(self):
        """Valid PDB file should produce confidence metrics."""
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  85.50           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  85.50           C
ATOM      3  N   GLY A   2       2.000   1.000   0.000  1.00  45.00           N
ATOM      4  CA  GLY A   2       3.458   1.000   0.000  1.00  45.00           C
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name

        try:
            result = analyze_structure_confidence(temp_path)

            assert result["success"] is True
            assert result["total_residues"] == 2
            assert 60 < result["mean_plddt"] < 70
            assert result["confident_residue_count"] == 1
            assert result["low_confidence_residue_count"] == 1
        finally:
            os.unlink(temp_path)


class TestBatchPredictStructures:
    """Tests for batch structure prediction."""

    def test_batch_empty_jobs(self):
        """Empty job list should fail."""
        result = batch_predict_structures([])
        assert result["success"] is False

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_batch_predict_multiple_jobs(self, mock_request):
        """Multiple jobs should be submitted."""
        mock_request.return_value = {"success": True, "result": {"job_id": "batch123"}}

        jobs = [
            {"type": "protein", "sequences": ["MKFLILLFNILCLFPVLAADNH"], "name": "protein1"},
            {"type": "protein", "sequences": ["MALTEVNPKKYIPGTKMIFAG"], "name": "protein2"},
        ]

        result = batch_predict_structures(jobs)

        assert result["success"] is True
        assert result["total_jobs"] == 2
        assert result["submitted"] == 2

    @mock.patch("biomni.tool.structure_prediction._make_api_request")
    def test_batch_predict_with_output_dir(self, mock_request):
        """Batch prediction should create output directory."""
        mock_request.return_value = {"success": True, "result": {"job_id": "batch123"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "structures")
            jobs = [{"type": "protein", "sequences": ["MKFLILLFNILCLFPVLAADNH"], "name": "test"}]

            result = batch_predict_structures(jobs, output_dir=output_dir)

            assert result["success"] is True
            assert os.path.isdir(output_dir)

    def test_batch_predict_invalid_job_type(self):
        """Invalid job type should be reported as failed."""
        jobs = [{"type": "invalid_type", "sequences": ["MKFL"], "name": "test"}]

        result = batch_predict_structures(jobs)

        assert result["failed"] == 1
        assert "Unknown job type" in result["jobs"][0]["error"]
