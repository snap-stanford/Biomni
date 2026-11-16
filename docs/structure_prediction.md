# AlphaFold3 Structure Prediction Tool

> Predict protein structures and molecular complexes using state-of-the-art AlphaFold3.

## Overview

This module provides a programmatic interface to AlphaFold3 for predicting:

- **Single protein structures** - 3D atomic coordinates from amino acid sequences
- **Protein-protein complexes** - Multi-chain assemblies with interface predictions
- **Protein-DNA/RNA complexes** - Nucleic acid binding and interaction modeling
- **Protein-ligand complexes** - Small molecule binding site and affinity prediction

## Installation

The module is included in Biomni. Set your API token:

```bash
export ALPHAFOLD_API_TOKEN="your_token_here"
```

## Quick Start

```python
from biomni.tool.structure_prediction import (
    predict_protein_structure,
    predict_protein_complex,
    predict_protein_nucleic_acid_complex,
    predict_protein_ligand_complex,
)

# Single protein
result = predict_protein_structure(
    sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
    save_path="./protein.pdb"
)

# Protein complex
result = predict_protein_complex(
    sequences=["MKFLILLFNILC...", "MALTEVNPKKY..."],
    chain_names=["ChainA", "ChainB"]
)

# Protein-DNA
result = predict_protein_nucleic_acid_complex(
    protein_sequences=["MKFLILLFNILC..."],
    nucleic_acid_sequence="ATCGATCGATCG",
    nucleic_acid_type="DNA"
)

# Protein-ligand
result = predict_protein_ligand_complex(
    protein_sequence="MKFLILLFNILC...",
    ligand_smiles="CCO"
)
```

## API Reference

### predict_protein_structure

Predict the 3D structure of a single protein.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | str | Amino acid sequence (10-2000 residues) |
| `save_path` | str | Optional path to save PDB file |
| `wait_for_result` | bool | Wait for completion (default: True) |
| `timeout_seconds` | int | Max wait time (default: 600) |

**Returns:** Dictionary with `pdb_content`, `confidence` scores, and `job_id`.

---

### predict_protein_complex

Predict multi-chain protein complex structures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | list[str] | List of protein sequences (≥2) |
| `chain_names` | list[str] | Optional names for each chain |
| `save_path` | str | Optional path to save PDB file |

**Returns:** Dictionary with structure, `iptm` interface score, and `interface_contacts`.

---

### predict_protein_nucleic_acid_complex

Predict protein-DNA or protein-RNA complex structures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `protein_sequences` | list[str] | One or more protein sequences |
| `nucleic_acid_sequence` | str | DNA or RNA sequence |
| `nucleic_acid_type` | str | "DNA" or "RNA" |

---

### predict_protein_ligand_complex

Predict protein-small molecule binding.

| Parameter | Type | Description |
|-----------|------|-------------|
| `protein_sequence` | str | Target protein sequence |
| `ligand_smiles` | str | Ligand in SMILES format |

**Returns:** Dictionary with `binding_site` residues and `binding_affinity`.

---

### batch_predict_structures

Submit multiple prediction jobs.

```python
from biomni.tool.structure_prediction import batch_predict_structures

jobs = [
    {"type": "protein", "sequences": ["MKFL..."], "name": "kinase"},
    {"type": "complex", "sequences": ["MKFL...", "MALT..."], "name": "dimer"},
    {"type": "protein_ligand", "sequences": ["MKFL..."], "ligand_smiles": "CCO", "name": "binding"},
]

result = batch_predict_structures(jobs, output_dir="./structures")
```

---

### analyze_structure_confidence

Analyze pLDDT confidence scores from a predicted structure.

```python
from biomni.tool.structure_prediction import analyze_structure_confidence

result = analyze_structure_confidence("./protein.pdb")
print(f"Mean pLDDT: {result['mean_plddt']}")
print(f"Quality: {result['quality_assessment']}")
```

## Confidence Metrics

| Metric | Description |
|--------|-------------|
| **pLDDT** | Per-residue confidence (0-100). >70 = confident, <50 = low confidence |
| **pTM** | Predicted TM-score for overall fold accuracy |
| **ipTM** | Interface TM-score for complex predictions |

## Async Usage

For long-running predictions:

```python
# Submit without waiting
result = predict_protein_structure(sequence, wait_for_result=False)
job_id = result["job_id"]

# Check status later
from biomni.tool.structure_prediction import get_job_status, download_structure

status = get_job_status(job_id)
if status["status"] == "completed":
    download_structure(job_id, "./output.pdb")
```

## Error Handling

All functions return a dictionary with a `success` boolean:

```python
result = predict_protein_structure("INVALID123")
if not result["success"]:
    print(f"Error: {result['error']}")
```

## Limitations

- Maximum sequence length: 2000 residues per chain
- Minimum sequence length: 10 residues
- API rate limits apply based on your token tier
