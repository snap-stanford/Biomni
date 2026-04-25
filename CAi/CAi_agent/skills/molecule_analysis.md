# Comprehensive Molecule Analysis

## Description

Perform comprehensive property analysis and evaluation of a given molecule, including basic descriptor calculation, ADMET prediction, synthetic accessibility assessment, and structure visualization.

## Metadata

**Category**: Molecular Analysis
**Required Tools**: rdkit, calculate_scscore, predict_admet, mol_to_image
**Difficulty**: Easy
**Use Cases**: Candidate compound evaluation, Molecular property verification, Medicinal chemistry assessment

---

## Workflow

When executing this skill, follow these steps:

### 1. Validate Molecular Structure
- Use RDKit to validate SMILES string validity
- Check if the molecule contains unreasonable chemical structures
- Standardize molecular representation (remove salts, neutralize, etc.)

### 2. Calculate Basic Molecular Descriptors
Calculate the following key descriptors:
- Molecular Weight (MW)
- LogP (lipophilicity)
- TPSA (Topological Polar Surface Area)
- Number of hydrogen bond donors/acceptors
- Number of rotatable bonds
- Number of aromatic rings

### 3. Assess Drug-likeness
Check compliance with the following rules:
- Lipinski's Rule of Five
- Veber's Rule
- Other drug-likeness filters

### 4. Predict ADMET Properties
Predict the following properties:
- **Absorption**: Caco-2 permeability, Human Intestinal Absorption
- **Distribution**: Plasma Protein Binding, BBB permeability
- **Metabolism**: CYP450 inhibition/substrate
- **Excretion**: Clearance, Half-life
- **Toxicity**: hERG toxicity, Hepatotoxicity, Mutagenicity

### 5. Assess Synthetic Accessibility
- Use SCScore to evaluate synthetic difficulty (1-5 scale, lower is easier)
- Identify potential synthetic challenges
- Assess commercial availability

### 6. Generate Visualization
- Generate 2D structure diagram
- Optional: Generate 3D conformation
- Annotate key functional groups

### 7. Summarize Analysis Results
Generate a comprehensive report including:
- Basic molecular information
- Drug-likeness assessment results
- ADMET prediction summary
- Synthetic accessibility score
- Medicinal chemistry recommendations

## Example Usage

### Example 1: Analyze Aspirin

**Input Task**: "Analyze the drug properties of aspirin (CC(=O)Oc1ccccc1C(=O)O)"

**Execution Steps**:
```python
# 1. Validate SMILES
from rdkit import Chem
smiles = "CC(=O)Oc1ccccc1C(=O)O"
mol = Chem.MolFromSmiles(smiles)
print(f"Molecule valid: {mol is not None}")

# 2. Calculate descriptors
from rdkit.Chem import Descriptors
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
print(f"Molecular Weight: {mw:.2f}")
print(f"LogP: {logp:.2f}")
print(f"TPSA: {tpsa:.2f}")

# 3. Assess drug-likeness
hbd = Descriptors.NumHDonors(mol)
hba = Descriptors.NumHAcceptors(mol)
lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
print(f"Lipinski's Rule: {'Pass' if lipinski_pass else 'Fail'}")

# 4. Predict ADMET
result = predict_admet(smiles)
print(result)

# 5. Assess synthetic accessibility
scscore = calculate_scscore(smiles)
print(f"SCScore: {scscore:.2f}")

# 6. Generate image
img = mol_to_image(smiles)
```

**Expected Output**:
```
Molecule valid: True
Molecular Weight: 180.16
LogP: 1.19
TPSA: 63.60
Lipinski's Rule: Pass
ADMET Prediction: {...}
SCScore: 1.85 (easy to synthesize)
```

### Example 2: Batch Analysis of Compound Library

**Input Task**: "Analyze the drug properties of the following 3 compounds"

```python
smiles_list = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
]

results = []
for smiles in smiles_list:
    # Execute complete analysis workflow
    analysis = analyze_molecule(smiles)
    results.append(analysis)

# Generate comparison report
generate_comparison_report(results)
```

## Tips and Best Practices

1. **SMILES Validation**: Always validate SMILES validity first to avoid downstream calculation errors
2. **Standardization**: For molecules containing salts or ionized forms, perform standardization first
3. **Multiple Conformations**: For flexible molecules, consider generating multiple conformations for analysis
4. **Threshold Setting**: ADMET prediction results should be judged in combination with specific thresholds
5. **Comprehensive Assessment**: Don't rely on a single metric; consider multiple factors comprehensively

## Common Issues

- **Invalid SMILES**: Check for special characters and stereochemistry notation
- **Abnormal Descriptors**: May indicate unreasonable molecular structure
- **ADMET Prediction Failure**: Check if molecule is within model's applicability domain
- **High SCScore**: May need to simplify molecular structure or find alternative synthetic routes

## Related Skills

- Retrosynthesis Route Planning
- Lead Compound Optimization
- Virtual Screening Workflow
