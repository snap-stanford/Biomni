# Skill: Scaffold-Based Molecular Generation with Docking-First Evaluation

This skill generates molecules from a user-provided scaffold, prioritizes `perform_molecular_docking_vina` when docking inputs are ready, and otherwise falls back to `calculate_scscore`, `predict_antibacterial_pmic`, and `predict_molecule_toxicity` for ranking and reporting.

## Purpose

Use this skill when the user provides a scaffold and wants to generate candidate molecules, evaluate them, and rank them for downstream selection.

This workflow is **docking-first**:
- if docking is feasible, rank molecules primarily by Vina score
- if docking is not feasible, skip Vina and rank molecules using SCScore, pMIC/MIC, and toxicity

---

## Required User Input

The user must provide:
- a scaffold SMILES

The user may additionally provide:
- a target protein file
- a target protein name
- docking center coordinates
- the number of molecules to generate
- preferred generation tools
- preferred ranking metrics

---

## Core Workflow

1. Validate the scaffold input
2. Select compatible scaffold-based generation tool(s)
3. Generate candidate molecules
4. Deduplicate and normalize generated molecules
5. Check whether Vina docking is feasible
6. If docking is feasible:
   - run `perform_molecular_docking_vina`
   - run `predict_antibacterial_pmic`
   - run `predict_molecule_toxicity`
   - optionally run `calculate_scscore`
   - rank molecules primarily by Vina score
7. If docking is not feasible:
   - skip `perform_molecular_docking_vina`
   - run `calculate_scscore`
   - run `predict_antibacterial_pmic`
   - run `predict_molecule_toxicity`
   - rank molecules using available non-docking metrics
8. Return the ranked molecules and clearly state whether Vina was used or skipped

---

## Tool Selection Rules

### 1. Scaffold-based generation tools

Use scaffold-compatible generation tools only.

#### Preferred generation functions
- `generate_scaffold_analogs(smiles, num_analogs=10)`
- `generate_libinvent_decorations(smiles, num_decorations=3)`
- `generate_molecules_reinvent4_libinvent(smiles, num_variants=50)`

#### For complete chiral molecules rather than wildcard scaffolds
- `generate_molecules_reinvent4_mol2mol(smiles, num_variants=50, strategy="beamsearch", temperature=1.0)`

### 2. Do not use by default
- `generate_molecules_drugex(...)`

Do not use `generate_molecules_drugex` unless explicitly requested or confirmed to be stable, because it is currently unreliable.

---

## Scaffold Routing Rules

### Case A: Scaffold contains `*` or `[*]` and does not contain `@@`
Preferred tools:
- `generate_scaffold_analogs`
- `generate_libinvent_decorations`
- `generate_molecules_reinvent4_libinvent`

### Case B: Input is a complete chiral molecule and contains `@@`
Prefer:
- `generate_molecules_reinvent4_mol2mol`

Do not force wildcard scaffold decoration tools on full chiral molecules.

### Case C: User explicitly specifies a generation tool
Use the specified compatible tool if possible.

---

## Molecule Generation Stage

Generate molecules using one or more compatible generation functions.

Possible functions:
- `generate_scaffold_analogs`
- `generate_libinvent_decorations`
- `generate_molecules_reinvent4_libinvent`
- `generate_molecules_reinvent4_mol2mol` for full-molecule analog generation

After generation:
- merge all generated SMILES
- remove duplicates
- keep track of source tool for each molecule

---

## Docking Eligibility Check

Before calling `perform_molecular_docking_vina`, always check whether docking can be executed reliably.

Docking is considered feasible only if:
- a usable target protein input is available
- a docking center is available
- receptor input is ready or can be reliably prepared
- ligand input is ready or can be reliably prepared
- docking box size can be set, defaulting to `[20, 20, 20]`

### Important rule
Do **not** call `perform_molecular_docking_vina` unless docking inputs are ready.

### If docking is not feasible
Skip Vina completely and move to fallback evaluation:
- `calculate_scscore`
- `predict_antibacterial_pmic`
- `predict_molecule_toxicity`

---

## Target Protein Handling Rules

### If the user provides a protein file
Use it as the receptor input candidate for docking.

### If the user provides only a target protein name
The skill may attempt web search or structure lookup to identify a usable protein structure and docking center.

However:
- if no reliable structure can be obtained
- or if receptor preparation is not reliable
then do **not** run Vina

Instead, continue with:
- `calculate_scscore`
- `predict_antibacterial_pmic`
- `predict_molecule_toxicity`

### If the user provides no protein information at all
Do not attempt docking.
Skip Vina and continue with non-docking evaluation only.

---

## Docking Rules

If docking is feasible, use:

- `perform_molecular_docking_vina(receptor_pdbqt_path, ligand_pdbqt_path, center_xyz, box_size_xyz, exhaustiveness=32)`

Default settings:
- `box_size_xyz = [20, 20, 20]`
- `exhaustiveness = 32`

Use Vina as the **primary metric** whenever docking succeeds.

### Important behavior
If docking preparation fails for some molecules:
- continue the workflow
- do not terminate the whole task
- evaluate those molecules with non-docking metrics only
- rank successfully docked molecules separately or ahead of undocked molecules

---

## Secondary Evaluation Rules

For each valid generated molecule, run:

### 1. SCScore
- `calculate_scscore(smiles=...)`

### 2. Antibacterial pMIC / MIC
- `predict_antibacterial_pmic(smiles=...)`

### 3. Toxicity
- `predict_molecule_toxicity(smiles=...)`

These tools are stable fallback evaluation functions when docking is unavailable, and they should also be reported alongside Vina when docking is available.

---

## Ranking Policy

### Path A: If Vina is available
Rank molecules primarily by:
1. Vina score ascending (more negative is better)

Also report:
- pMIC / estimated MIC_uM from `predict_antibacterial_pmic`
- toxicity probability / verdict from `predict_molecule_toxicity`
- SCScore from `calculate_scscore` if available

### Path B: If Vina is not available
Do not output a Vina-based ranking.

Instead, rank molecules by available non-docking metrics:
1. toxicity probability ascending
2. estimated MIC_uM ascending or pMIC descending
3. SCScore ascending

### Missing values
- molecules missing the primary metric for the current ranking path should be placed after molecules with complete scores
- undocked molecules should not be mixed into the top Vina-ranked list

---

## Required Output Content

The final response should include:
- whether Vina was used or skipped
- which generation function(s) were used
- the scaffold input
- the target protein information if available
- the docking center if used
- the docking box size if used
- the total number of generated molecules
- the ranked molecule list

Each molecule entry should include:
- SMILES
- source generation function
- Vina score if available
- pMIC / estimated MIC_uM if available
- toxicity probability / verdict if available
- SCScore if available
- notes on skipped or failed steps if needed

---

## Failure Handling

### Invalid scaffold
Return an error immediately and do not continue.

### Valid scaffold but no usable protein
Skip `perform_molecular_docking_vina` and continue with:
- `calculate_scscore`
- `predict_antibacterial_pmic`
- `predict_molecule_toxicity`

### Docking preparation failure
Skip Vina for affected molecules or skip docking entirely if no molecule is docking-ready.

### Partial success
Always return the best available results rather than failing the entire workflow.

---

## One-Line Decision Rule

Generate molecules from the scaffold using compatible generation functions, run `perform_molecular_docking_vina` only if docking inputs are ready, and otherwise fall back to `calculate_scscore`, `predict_antibacterial_pmic`, and `predict_molecule_toxicity` for ranking and reporting.