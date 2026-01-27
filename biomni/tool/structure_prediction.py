"""Structure prediction tools using AlphaFold3 for protein complex modeling.

This module provides functions for predicting protein structures and their
interactions with other proteins, nucleic acids, and small molecules using
the AlphaFold Server API.
"""

import hashlib
import json
import os
import time
from typing import Any

import requests

ALPHAFOLD_SERVER_URL = "https://alphafoldserver.com"
ALPHAFOLD_API_URL = f"{ALPHAFOLD_SERVER_URL}/api"


def _get_api_token() -> str | None:
    """Retrieve the AlphaFold Server API token from environment."""
    return os.environ.get("ALPHAFOLD_API_TOKEN")


def _make_api_request(
    endpoint: str,
    method: str = "GET",
    data: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make an authenticated request to the AlphaFold Server API."""
    token = _get_api_token()
    
    if headers is None:
        headers = {}
    
    headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"{ALPHAFOLD_API_URL}/{endpoint.lstrip('/')}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        
        try:
            result = response.json()
        except ValueError:
            result = {"raw_text": response.text}
        
        return {"success": True, "result": result}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", error_data.get("message", str(e)))
            except ValueError:
                error_msg = e.response.text[:500] if e.response.text else str(e)
        return {"success": False, "error": error_msg}


def _validate_sequence(sequence: str) -> tuple[bool, str]:
    """Validate a protein sequence contains only valid amino acid codes."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    sequence_upper = sequence.upper().replace(" ", "").replace("\n", "")
    
    invalid_chars = set(sequence_upper) - valid_aa
    if invalid_chars:
        return False, f"Invalid characters in sequence: {invalid_chars}"
    
    if len(sequence_upper) < 10:
        return False, "Sequence too short (minimum 10 residues)"
    
    if len(sequence_upper) > 2000:
        return False, "Sequence too long (maximum 2000 residues for API)"
    
    return True, sequence_upper


def _validate_nucleic_acid(sequence: str, na_type: str = "DNA") -> tuple[bool, str]:
    """Validate a nucleic acid sequence."""
    if na_type.upper() == "DNA":
        valid_bases = set("ATCGN")
    else:
        valid_bases = set("AUCGN")
    
    sequence_upper = sequence.upper().replace(" ", "").replace("\n", "")
    invalid_chars = set(sequence_upper) - valid_bases
    
    if invalid_chars:
        return False, f"Invalid characters for {na_type}: {invalid_chars}"
    
    if len(sequence_upper) < 5:
        return False, "Sequence too short (minimum 5 bases)"
    
    return True, sequence_upper


def _generate_job_id(sequences: list[str]) -> str:
    """Generate a unique job ID based on input sequences."""
    content = "".join(sequences)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _poll_job_status(
    job_id: str,
    max_wait_seconds: int = 600,
    poll_interval: int = 10,
) -> dict[str, Any]:
    """Poll for job completion status."""
    elapsed = 0
    
    while elapsed < max_wait_seconds:
        result = _make_api_request(f"jobs/{job_id}")
        
        if not result["success"]:
            return result
        
        status = result["result"].get("status", "unknown")
        
        if status == "completed":
            return {"success": True, "result": result["result"]}
        elif status == "failed":
            error = result["result"].get("error", "Job failed without error message")
            return {"success": False, "error": error}
        elif status in ("pending", "running"):
            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            return {"success": False, "error": f"Unknown job status: {status}"}
    
    return {"success": False, "error": f"Job timed out after {max_wait_seconds} seconds"}


def predict_protein_structure(
    sequence: str,
    save_path: str | None = None,
    wait_for_result: bool = True,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Predict the 3D structure of a single protein sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence in single-letter code (e.g., "MKFLILLFNILC...")
    save_path : str, optional
        Path to save the predicted structure file (PDB format)
    wait_for_result : bool
        If True, wait for prediction to complete. If False, return job ID immediately
    timeout_seconds : int
        Maximum time to wait for prediction in seconds
    
    Returns
    -------
    dict
        Dictionary containing:
        - success: bool indicating if prediction succeeded
        - structure_path: path to saved PDB file (if save_path provided)
        - pdb_content: PDB file content as string
        - confidence: overall confidence metrics
        - job_id: ID for tracking the prediction job
    
    Examples
    --------
    >>> result = predict_protein_structure("MKFLILLFNILCLFPVLAADNH...")
    >>> print(result["confidence"]["plddt_mean"])
    
    """
    is_valid, validated_seq = _validate_sequence(sequence)
    if not is_valid:
        return {"success": False, "error": validated_seq}
    
    job_data = {
        "sequences": [{"type": "protein", "sequence": validated_seq}],
        "model": "alphafold3",
    }
    
    submit_result = _make_api_request("predict", method="POST", data=job_data)
    
    if not submit_result["success"]:
        return submit_result
    
    job_id = submit_result["result"].get("job_id")
    
    if not job_id:
        return {"success": False, "error": "No job ID returned from server"}
    
    if not wait_for_result:
        return {"success": True, "job_id": job_id, "status": "submitted"}
    
    poll_result = _poll_job_status(job_id, max_wait_seconds=timeout_seconds)
    
    if not poll_result["success"]:
        return poll_result
    
    job_result = poll_result["result"]
    pdb_content = job_result.get("pdb_content", "")
    
    response = {
        "success": True,
        "job_id": job_id,
        "pdb_content": pdb_content,
        "confidence": {
            "plddt_mean": job_result.get("plddt_mean"),
            "ptm": job_result.get("ptm"),
        },
    }
    
    if save_path and pdb_content:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pdb_content)
        response["structure_path"] = os.path.abspath(save_path)
    
    return response


def predict_protein_complex(
    sequences: list[str],
    chain_names: list[str] | None = None,
    save_path: str | None = None,
    wait_for_result: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Predict the 3D structure of a protein complex from multiple sequences.
    
    Parameters
    ----------
    sequences : list[str]
        List of protein sequences, one per chain
    chain_names : list[str], optional
        Names for each chain (defaults to A, B, C, ...)
    save_path : str, optional
        Path to save the predicted complex structure (PDB format)
    wait_for_result : bool
        If True, wait for prediction to complete
    timeout_seconds : int
        Maximum time to wait for prediction
    
    Returns
    -------
    dict
        Dictionary containing:
        - success: bool
        - structure_path: path to saved PDB file
        - pdb_content: PDB content as string
        - confidence: per-chain and interface confidence metrics
        - interface_contacts: predicted inter-chain contacts
    
    Examples
    --------
    >>> seqs = ["MKFLILLFNILCLFPVLAADNH...", "MALTEVNPKKYIPGTKMIFAG..."]
    >>> result = predict_protein_complex(seqs, chain_names=["Receptor", "Ligand"])
    
    """
    if not sequences or len(sequences) < 2:
        return {"success": False, "error": "At least 2 sequences required for complex prediction"}
    
    validated_sequences = []
    for i, seq in enumerate(sequences):
        is_valid, validated_seq = _validate_sequence(seq)
        if not is_valid:
            return {"success": False, "error": f"Chain {i+1}: {validated_seq}"}
        validated_sequences.append(validated_seq)
    
    if chain_names is None:
        chain_names = [chr(ord("A") + i) for i in range(len(sequences))]
    
    sequence_data = [
        {"type": "protein", "sequence": seq, "chain_id": name}
        for seq, name in zip(validated_sequences, chain_names)
    ]
    
    job_data = {
        "sequences": sequence_data,
        "model": "alphafold3",
        "predict_interface": True,
    }
    
    submit_result = _make_api_request("predict", method="POST", data=job_data)
    
    if not submit_result["success"]:
        return submit_result
    
    job_id = submit_result["result"].get("job_id")
    
    if not wait_for_result:
        return {"success": True, "job_id": job_id, "status": "submitted"}
    
    poll_result = _poll_job_status(job_id, max_wait_seconds=timeout_seconds)
    
    if not poll_result["success"]:
        return poll_result
    
    job_result = poll_result["result"]
    pdb_content = job_result.get("pdb_content", "")
    
    response = {
        "success": True,
        "job_id": job_id,
        "pdb_content": pdb_content,
        "confidence": {
            "plddt_mean": job_result.get("plddt_mean"),
            "ptm": job_result.get("ptm"),
            "iptm": job_result.get("iptm"),
            "chain_confidences": job_result.get("chain_confidences", {}),
        },
        "interface_contacts": job_result.get("interface_contacts", []),
    }
    
    if save_path and pdb_content:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pdb_content)
        response["structure_path"] = os.path.abspath(save_path)
    
    return response


def predict_protein_nucleic_acid_complex(
    protein_sequences: list[str],
    nucleic_acid_sequence: str,
    nucleic_acid_type: str = "DNA",
    save_path: str | None = None,
    wait_for_result: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Predict structure of a protein-DNA or protein-RNA complex.
    
    Parameters
    ----------
    protein_sequences : list[str]
        One or more protein sequences
    nucleic_acid_sequence : str
        DNA or RNA sequence
    nucleic_acid_type : str
        Either "DNA" or "RNA"
    save_path : str, optional
        Path to save structure
    wait_for_result : bool
        Wait for completion
    timeout_seconds : int
        Maximum wait time
    
    Returns
    -------
    dict
        Prediction results including structure and confidence metrics
    
    Examples
    --------
    >>> protein = "MKFLILLFNILCLFPVLAADNH..."
    >>> dna = "ATCGATCGATCGATCG"
    >>> result = predict_protein_nucleic_acid_complex([protein], dna, "DNA")
    
    """
    if nucleic_acid_type.upper() not in ("DNA", "RNA"):
        return {"success": False, "error": "nucleic_acid_type must be 'DNA' or 'RNA'"}
    
    validated_proteins = []
    for i, seq in enumerate(protein_sequences):
        is_valid, validated_seq = _validate_sequence(seq)
        if not is_valid:
            return {"success": False, "error": f"Protein {i+1}: {validated_seq}"}
        validated_proteins.append(validated_seq)
    
    is_valid, validated_na = _validate_nucleic_acid(nucleic_acid_sequence, nucleic_acid_type)
    if not is_valid:
        return {"success": False, "error": validated_na}
    
    sequence_data = [
        {"type": "protein", "sequence": seq, "chain_id": chr(ord("A") + i)}
        for i, seq in enumerate(validated_proteins)
    ]
    
    na_chain_id = chr(ord("A") + len(validated_proteins))
    sequence_data.append({
        "type": nucleic_acid_type.lower(),
        "sequence": validated_na,
        "chain_id": na_chain_id,
    })
    
    job_data = {
        "sequences": sequence_data,
        "model": "alphafold3",
        "predict_interface": True,
    }
    
    submit_result = _make_api_request("predict", method="POST", data=job_data)
    
    if not submit_result["success"]:
        return submit_result
    
    job_id = submit_result["result"].get("job_id")
    
    if not wait_for_result:
        return {"success": True, "job_id": job_id, "status": "submitted"}
    
    poll_result = _poll_job_status(job_id, max_wait_seconds=timeout_seconds)
    
    if not poll_result["success"]:
        return poll_result
    
    job_result = poll_result["result"]
    pdb_content = job_result.get("pdb_content", "")
    
    response = {
        "success": True,
        "job_id": job_id,
        "pdb_content": pdb_content,
        "confidence": {
            "plddt_mean": job_result.get("plddt_mean"),
            "ptm": job_result.get("ptm"),
            "iptm": job_result.get("iptm"),
        },
        "protein_na_contacts": job_result.get("interface_contacts", []),
    }
    
    if save_path and pdb_content:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pdb_content)
        response["structure_path"] = os.path.abspath(save_path)
    
    return response


def predict_protein_ligand_complex(
    protein_sequence: str,
    ligand_smiles: str,
    save_path: str | None = None,
    wait_for_result: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Predict structure of a protein-small molecule complex.
    
    Parameters
    ----------
    protein_sequence : str
        Protein sequence in single-letter amino acid code
    ligand_smiles : str
        Small molecule in SMILES format
    save_path : str, optional
        Path to save structure
    wait_for_result : bool
        Wait for completion
    timeout_seconds : int
        Maximum wait time
    
    Returns
    -------
    dict
        Prediction results including:
        - structure_path: saved PDB file path
        - pdb_content: structure as string
        - binding_site: predicted binding residues
        - binding_affinity: predicted binding strength (if available)
    
    Examples
    --------
    >>> protein = "MKFLILLFNILCLFPVLAADNH..."
    >>> erlotinib = "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"
    >>> result = predict_protein_ligand_complex(protein, erlotinib)
    
    """
    is_valid, validated_seq = _validate_sequence(protein_sequence)
    if not is_valid:
        return {"success": False, "error": validated_seq}
    
    if not ligand_smiles or len(ligand_smiles) < 2:
        return {"success": False, "error": "Invalid SMILES string"}
    
    sequence_data = [
        {"type": "protein", "sequence": validated_seq, "chain_id": "A"},
        {"type": "ligand", "smiles": ligand_smiles, "chain_id": "L"},
    ]
    
    job_data = {
        "sequences": sequence_data,
        "model": "alphafold3",
        "predict_binding": True,
    }
    
    submit_result = _make_api_request("predict", method="POST", data=job_data)
    
    if not submit_result["success"]:
        return submit_result
    
    job_id = submit_result["result"].get("job_id")
    
    if not wait_for_result:
        return {"success": True, "job_id": job_id, "status": "submitted"}
    
    poll_result = _poll_job_status(job_id, max_wait_seconds=timeout_seconds)
    
    if not poll_result["success"]:
        return poll_result
    
    job_result = poll_result["result"]
    pdb_content = job_result.get("pdb_content", "")
    
    response = {
        "success": True,
        "job_id": job_id,
        "pdb_content": pdb_content,
        "confidence": {
            "plddt_mean": job_result.get("plddt_mean"),
            "ptm": job_result.get("ptm"),
        },
        "binding_site": job_result.get("binding_residues", []),
        "binding_affinity": job_result.get("predicted_affinity"),
    }
    
    if save_path and pdb_content:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pdb_content)
        response["structure_path"] = os.path.abspath(save_path)
    
    return response


def get_job_status(job_id: str) -> dict[str, Any]:
    """Check the status of a submitted prediction job.
    
    Parameters
    ----------
    job_id : str
        Job ID returned from a prediction function
    
    Returns
    -------
    dict
        Job status including:
        - status: "pending", "running", "completed", or "failed"
        - progress: percentage complete (if available)
        - result: prediction results (if completed)
    
    """
    result = _make_api_request(f"jobs/{job_id}")
    
    if not result["success"]:
        return result
    
    job_data = result["result"]
    
    return {
        "success": True,
        "job_id": job_id,
        "status": job_data.get("status", "unknown"),
        "progress": job_data.get("progress"),
        "created_at": job_data.get("created_at"),
        "result": job_data if job_data.get("status") == "completed" else None,
    }


def download_structure(
    job_id: str,
    output_path: str,
    file_format: str = "pdb",
) -> dict[str, Any]:
    """Download the structure file from a completed prediction job.
    
    Parameters
    ----------
    job_id : str
        Job ID of a completed prediction
    output_path : str
        Path to save the structure file
    file_format : str
        Output format: "pdb" or "cif"
    
    Returns
    -------
    dict
        Download result with file path
    
    """
    if file_format.lower() not in ("pdb", "cif"):
        return {"success": False, "error": "Format must be 'pdb' or 'cif'"}
    
    result = _make_api_request(f"jobs/{job_id}/structure", timeout=60)
    
    if not result["success"]:
        return result
    
    structure_content = result["result"].get(f"{file_format}_content", "")
    
    if not structure_content:
        return {"success": False, "error": f"No {file_format.upper()} content available"}
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(structure_content)
    
    return {
        "success": True,
        "file_path": os.path.abspath(output_path),
        "format": file_format,
        "size_bytes": len(structure_content),
    }


def analyze_structure_confidence(pdb_path: str) -> dict[str, Any]:
    """Analyze confidence scores from a predicted structure file.
    
    Parameters
    ----------
    pdb_path : str
        Path to a PDB file from AlphaFold prediction
    
    Returns
    -------
    dict
        Confidence analysis including:
        - residue_plddt: per-residue pLDDT scores
        - mean_plddt: average pLDDT
        - confident_regions: regions with pLDDT > 70
        - low_confidence_regions: regions with pLDDT < 50
    
    """
    if not os.path.exists(pdb_path):
        return {"success": False, "error": f"File not found: {pdb_path}"}
    
    residue_plddt = {}
    
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21].strip()
                    res_num = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                    
                    key = f"{chain}:{res_num}"
                    if key not in residue_plddt:
                        residue_plddt[key] = b_factor
    except Exception as e:
        return {"success": False, "error": f"Failed to parse PDB: {str(e)}"}
    
    if not residue_plddt:
        return {"success": False, "error": "No residues found in PDB file"}
    
    scores = list(residue_plddt.values())
    mean_plddt = sum(scores) / len(scores)
    
    confident_regions = [k for k, v in residue_plddt.items() if v > 70]
    low_confidence_regions = [k for k, v in residue_plddt.items() if v < 50]
    
    return {
        "success": True,
        "file": pdb_path,
        "total_residues": len(residue_plddt),
        "mean_plddt": round(mean_plddt, 2),
        "confident_residue_count": len(confident_regions),
        "low_confidence_residue_count": len(low_confidence_regions),
        "quality_assessment": (
            "High confidence" if mean_plddt > 70 else
            "Medium confidence" if mean_plddt > 50 else
            "Low confidence"
        ),
        "residue_scores": residue_plddt,
    }


def batch_predict_structures(
    jobs: list[dict[str, Any]],
    output_dir: str | None = None,
    parallel: bool = True,
    max_concurrent: int = 5,
) -> dict[str, Any]:
    """Submit multiple structure prediction jobs.
    
    Parameters
    ----------
    jobs : list[dict]
        List of job specifications, each containing:
        - type: "protein", "complex", "protein_dna", or "protein_ligand"
        - sequences: sequence data for the job
        - name: optional name for the job
    output_dir : str, optional
        Directory to save all output structures
    parallel : bool
        Whether to submit jobs in parallel
    max_concurrent : int
        Maximum concurrent jobs if parallel=True
    
    Returns
    -------
    dict
        Batch results including job IDs and status for each submission
    
    Examples
    --------
    >>> jobs = [
    ...     {"type": "protein", "sequences": ["MKFL..."], "name": "protein1"},
    ...     {"type": "complex", "sequences": ["MKFL...", "MALT..."], "name": "complex1"},
    ... ]
    >>> result = batch_predict_structures(jobs, output_dir="./structures")
    
    """
    if not jobs:
        return {"success": False, "error": "No jobs provided"}
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, job in enumerate(jobs):
        job_type = job.get("type", "protein")
        sequences = job.get("sequences", [])
        name = job.get("name", f"job_{i+1}")
        
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{name}.pdb")
        
        try:
            if job_type == "protein":
                if not sequences:
                    result = {"success": False, "error": "No sequence provided"}
                else:
                    result = predict_protein_structure(
                        sequences[0],
                        save_path=save_path,
                        wait_for_result=False,
                    )
            elif job_type == "complex":
                result = predict_protein_complex(
                    sequences,
                    save_path=save_path,
                    wait_for_result=False,
                )
            elif job_type == "protein_dna":
                proteins = job.get("proteins", sequences[:-1] if len(sequences) > 1 else [])
                na_seq = job.get("nucleic_acid", sequences[-1] if sequences else "")
                na_type = job.get("nucleic_acid_type", "DNA")
                result = predict_protein_nucleic_acid_complex(
                    proteins,
                    na_seq,
                    na_type,
                    save_path=save_path,
                    wait_for_result=False,
                )
            elif job_type == "protein_ligand":
                protein = sequences[0] if sequences else ""
                ligand = job.get("ligand_smiles", "")
                result = predict_protein_ligand_complex(
                    protein,
                    ligand,
                    save_path=save_path,
                    wait_for_result=False,
                )
            else:
                result = {"success": False, "error": f"Unknown job type: {job_type}"}
            
            result["name"] = name
            results.append(result)
            
        except Exception as e:
            results.append({
                "success": False,
                "name": name,
                "error": str(e),
            })
    
    successful = sum(1 for r in results if r.get("success", False))
    
    return {
        "success": successful > 0,
        "total_jobs": len(jobs),
        "submitted": successful,
        "failed": len(jobs) - successful,
        "jobs": results,
    }
