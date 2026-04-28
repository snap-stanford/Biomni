# Unified Lead Optimization & Evaluation Protocol

## Description
This is a master skill for end-to-end molecular generation, property evaluation, and lead optimization. It features dynamic tool routing based on molecular structure, strict defensive programming protocols for API handling, a dual-path ranking system (Docking vs. Fallback), and an interactive data visualization output.

## Metadata
* **Category**: Generative Chemistry & Drug Discovery
* **Required Tools**: `rdkit`, `generate_scaffold_analogs`, `generate_libinvent_decorations`, `generate_molecules_reinvent4_libinvent`, `generate_molecules_reinvent4_mol2mol`, `perform_molecular_docking_vina`, `calculate_scscore`, `predict_antibacterial_pmic`, `predict_molecule_toxicity`
* **Difficulty**: Expert / Autonomous

---

## 1. Required User Input
* **Mandatory**: A starting SMILES string.
* **Optional (Triggers Docking)**: Target protein file path, docking center `[x,y,z]`.
* **Optional**: Target number of variants (default 10-30).

---

## 2. Intelligent Tool Routing Rules
DO NOT use generation tools randomly. Inspect the input SMILES first:

* **Case A: Scaffold with Wildcards (`*` or `[*]`) & NO Chirality (`@@`)**
  * *Tools*: `generate_scaffold_analogs`, `generate_libinvent_decorations`, `generate_molecules_reinvent4_libinvent`.
* **Case B: Complete Chiral Molecule (Contains `@@`)**
  * *Tools*: `generate_molecules_reinvent4_mol2mol`. (Do NOT use scaffold tools here).

---

## 3. Strict Execution & Schema Parsing (CRITICAL)
Tools return JSON strings. **You MUST use `json.loads(result)`** and parse exactly using these schemas to prevent execution crashes:

### A. Defensive Programming Mandates
1. **Never crash the loop**: Wrap the evaluation of *each individual molecule* in a `try...except` block.
2. **Handle missing data safely**: If a tool fails or returns an error string, assign `None` to that molecule's metric.
3. **Format safely**: When printing, use conditional formatting (e.g., `f"{val:.4f}" if val is not None else "N/A"`) to prevent `NoneType` errors.

### B. Tool-Specific JSON Schemas
* **Generation Tools (e.g., `generate_scaffold_analogs`, `generate_libinvent_decorations`)**
  * *Success check*: `if data.get("status") == "success":` or `if data.get("success") is True:`
  * *Values*: Extract SMILES from `data.get("molecules")` or `data.get("molecules_smiles")`.
* **Antibacterial (`predict_antibacterial_pmic`)**
  * *Success check*: `if data.get("status") == "success":`
  * *Values*: `data.get("pMIC_value")` and `data.get("estimated_MIC_uM")`
* **Toxicity (`predict_molecule_toxicity`)**
  * *Success check*: `if "verdict" in data:`
  * *Values*: `data.get("verdict")` and `data.get("toxicity_probability")`
* **Synthesizability (`calculate_scscore`)**
  * *Success check*: `if data.get("success") is True:`
  * *Values*: Extract mapping from the `results` list (key: `canonical_smiles`, value: `scscore`).
* **Docking (`perform_molecular_docking_vina`)**
  * *Success check*: `if data.get("status") == "success":`
  * *Values*: `data.get("best_docking_score_kcal_mol")`

---

## 4. The Core Pipeline
1. **Validate**: Parse SMILES with RDKit.
2. **Route & Generate**: Use the correct tools (Section 2).
3. **Deduplicate**: Convert all generated SMILES to Canonical SMILES using RDKit; remove duplicates.
4. **Evaluate Properties**: Run SCScore (batch), pMIC, and Toxicity (loop with `try...except`).
5. **Docking Check**: If protein path and center are provided, run Vina on valid molecules.
6. **Rank & Sort**: Apply the Ranking Policy (Section 5).

---

## 5. Dual-Path Ranking Policy
Sort ONLY the successfully evaluated molecules. Move failed/null evaluations to the bottom of the list.

### Path A: Docking-First (If Vina executed successfully)
1. **Primary**: Vina Score ascending (more negative = better affinity).
2. **Tie-breakers**: Toxicity (Non-Toxic first) > pMIC (descending) > SCScore (ascending).

### Path B: Fallback (No protein target provided)
1. **Safety First**: Molecules with `verdict == "Non-Toxic"` (or probability < 0.5) rank higher.
2. **Efficacy Second**: `pMIC_value` descending (higher is better) / `estimated_MIC_uM` ascending.
3. **Synthesizability Third**: `SCScore` ascending (lower is better).

---

## 6. Output Formatting & Deliverables
Your final response MUST include the following three sections:

### Part 1: Workflow Summary
A Markdown table summarizing the input scaffold, tools used, total unique molecules generated, and the ranking strategy applied.

### Part 2: Top Candidates & SAR Insights
List the top 3-5 molecules with their properties (SMILES, Vina/pMIC, SCScore, Toxicity). Provide 1-2 sentences of SAR (Structure-Activity Relationship) analysis identifying which functional groups contributed to better scores.

### Part 3: Interactive SAR Scatter Plot
**GUARDRAIL**: Only generate the interactive widget below if there are **at least 3 molecules** with successfully evaluated BOTH `SCScore` AND `pMIC_value`. If insufficient data exists, skip the JSON generation and output a brief text explanation stating visual analysis is unavailable.

If data is sufficient, generate the interactive widget using the `json?chameleon` format. **Replace the `[DATA_START]` to `[DATA_END]` block with your actual generated data.**


