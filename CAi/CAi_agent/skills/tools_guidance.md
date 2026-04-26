# Docking-First Scaffold Optimization and Evaluation

## Description

This skill defines a robust, end-to-end workflow for generating molecular analogs from a user-provided scaffold, evaluating their properties, and ranking them. It dynamically routes the generation task based on the input SMILES structure and prioritizes 3D molecular docking (AutoDock Vina) when feasible, smoothly falling back to 2D metrics (pMIC, SCScore, Toxicity) when docking inputs are unavailable.

## Metadata

**Category**: Generative Chemistry & Lead Optimization
**Required Tools**: `generate_scaffold_analogs`, `generate_libinvent_decorations`, `generate_molecules_reinvent4_libinvent`, `generate_molecules_reinvent4_mol2mol`, `perform_molecular_docking_vina`, `calculate_scscore`, `predict_antibacterial_pmic`, `predict_molecule_toxicity`
**Difficulty**: Advanced
**Use Cases**: Scaffold hopping, targeted compound generation, virtual screening, lead optimization

---

## 1. Required User Input

**Mandatory:**
* A starting SMILES string (can be a scaffold with wildcards or a complete chiral molecule).

**Optional (Triggers the Docking Path):**
* Target protein file path (e.g., `.pdb`, `.pdbqt`).
* Docking center coordinates `[x, y, z]`.
* Docking box size (defaults to `[20, 20, 20]`).
* Requested number of generated variants (defaults to 10-50 depending on the tool).

---

## 2. Core Workflow & Logic

1.  **Input Validation**: Check if the provided SMILES is structurally valid and identify key features (e.g., `*`, `[*]`, `@@`).
2.  **Tool Routing**: Select the appropriate generation tools based on the structural features.
3.  **Generation & Deduplication**: Call the selected tools, merge the raw JSON responses, extract the generated SMILES, and deduplicate them using RDKit canonicalization.
4.  **Feasibility Check**: Determine if the Vina docking path can be executed (requires both target protein path and center coordinates).
5.  **Evaluation Phase**:
    * *Primary (if docking feasible)*: Run `perform_molecular_docking_vina`.
    * *Secondary (Always run)*: Run `calculate_scscore` (batch), `predict_antibacterial_pmic`, and `predict_molecule_toxicity`.
6.  **Ranking**: Sort the validated molecules based on the available metrics (Vina score OR a composite 2D score).
7.  **Reporting**: Present the final ranked list to the user in a clean Markdown table.

---

## 3. Tool Selection Rules (Routing)

Do NOT use tools randomly. Select the generation tool strictly based on the input SMILES characteristics:

### Case A: Scaffold contains wildcards (`*` or `[*]`) and NO chirality (`@@`)
* **Preferred Tools:** `generate_scaffold_analogs`, `generate_libinvent_decorations`, `generate_molecules_reinvent4_libinvent`.
* **Action:** You can use one or a combination of these tools to generate a diverse set of decorations.

### Case B: Input is a complete molecule with chirality (`@@`)
* **Preferred Tool:** `generate_molecules_reinvent4_mol2mol`.
* **Action:** Do NOT use scaffold-based tools (Lib-INVENT) on complete chiral molecules as it will result in errors.

### Excluded Tools
* Do NOT use `generate_molecules_reinvent4_denovo` or `generate_molecules_drugex` unless the user explicitly asks for "from scratch" or "fragment-based graph generation".

---

## 4. Execution Rules (Defensive Programming)

**CRITICAL RULE: Never assume an API call succeeds perfectly.**

1.  **Always Parse JSON**: Tools return JSON-formatted strings. You MUST use `json.loads(result)` before attempting to extract data. Do not treat the raw string as a dictionary.
2.  **Check Status Keys**: Before extracting values like `estimated_MIC_uM` or `vina_score`, you MUST verify the tool executed successfully by checking `if data.get("status") == "success":` or `if data.get("success") is True:`.
3.  **Isolate Failures**: Wrap individual molecule evaluations in `try...except` blocks. If one molecule fails toxicity prediction, log the error and continue evaluating the rest. NEVER let one failed API call crash the entire loop.
4.  **Save as Structured Data**: Save the successfully parsed results into a structured List of Dictionaries (e.g., `[{"smiles": "...", "mic_uM": 0.5, "scscore": 2.1}]`) to ensure safe and easy sorting later.

---

## 5. Ranking Policy

Separate successfully evaluated molecules from those with missing metrics. Sort ONLY the successful ones to prevent execution errors.

### Path A: Docking-First (Vina is available)
Rank molecules primarily by **Vina Score (ascending / more negative is better)**.
* *Tie-breakers or Secondary info:* Toxicity, MIC_uM, SCScore.

### Path B: Fallback (Vina is NOT available)
Rank molecules using a composite logic of 2D metrics:
1.  **Toxicity**: Non-toxic preferred over Toxic.
2.  **Antibacterial Activity**: `estimated_MIC_uM` (ascending / lower is better) OR `pMIC_value` (descending / higher is better).
3.  **Synthesizability**: `SCScore` (ascending / closer to 1 is better).

---

## 6. Output Formatting

Conclude your task by presenting the user with a structured report containing:
* **Execution Summary**: Mention the initial scaffold, which generation tools were routed, total unique molecules generated, and whether Docking or Fallback ranking was used.
* **Ranked Table**: A clear Markdown table containing:
    * Rank
    * SMILES
    * Source Tool
    * Vina Score (kcal/mol) - *if applicable*
    * MIC (µM) / pMIC
    * SCScore
    * Toxicity
* **Insights**: Briefly highlight the top 1-2 molecules and explain why they are recommended based on the generated metrics.