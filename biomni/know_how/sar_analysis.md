# SAR Analysis

---

## Metadata

**Short Description**: Comprehensive guide for performing Structure-Activity Relationship (SAR) analysis using RDKit.

**Authors**: Biomni Team

**Version**: 1.0

**Last Updated**: December 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

---

You are an expert in Cheminformatics and Python. Perform a SAR (Structure-Activity Relationship) analysis using RDKit.

**Task Requirements:**

1.  **Data Loading:** Load the CSV file (`Compound Key`, `Smiles`, `Standard Value`).

2.  **Core Identification (MCS):**
    *   Use `rdFMCS.FindMCS` to find a significant common scaffold.
    *   **Pre-processing:** Apply `Chem.AddHs` to molecules before finding MCS.
    *   **Reference Code:** Use the following parameter settings for robust core identification:
        ```python
        mols_for_mcs = [Chem.AddHs(m) for m in mols]
        mcs_res = rdFMCS.FindMCS(
            mols_for_mcs,
            threshold=0.8,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder
        )
        core_mol = Chem.MolFromSmarts(mcs_res.smartsString)
        AllChem.Compute2DCoords(core_mol)
        ```

3.  **R-Group Decomposition & Refinement:**
    *   Perform decomposition based on the Core.
    *   **Refinement:** Exclude any R-group columns that are identical (constant) across all molecules. Remove these constant points from the Core visualization as well.

4.  **Image Generation & Alignment (Strict Coordinate Extraction):**
    *   **Goal:** Ensure Core and R-groups are visually perfectly superimposed on the Original Molecule.
    *   **Drawing Style:** When drawing molecules, always use ACS1996 mode for consistent and professional visualization:
        ```python
        options = drawer.drawOptions()
        options.setACS1996Mode(True)
        ```
    *   **Reference Implementation:** Use this specific alignment logic to guarantee perfect overlay:
        ```python
        def align_substructure_to_parent(sub, parent):
            if not sub or not parent: return False
            try:
                # Strategy 1: Direct match
                match = parent.GetSubstructMatch(sub)
                
                # Strategy 2: Convert dummies to queries (handle R-group attachment points)
                if not match:
                    params = Chem.AdjustQueryParameters()
                    params.makeDummiesQueries = True
                    params.adjustDegree = False
                    params.adjustRingCount = False
                    sub_query = Chem.AdjustQueryProperties(sub, params)
                    match = parent.GetSubstructMatch(sub_query)
                
                # Strategy 3: Try without chirality
                if not match:
                     match = parent.GetSubstructMatch(sub, useChirality=False)

                if match:
                    conf_parent = parent.GetConformer()
                    conf_sub = Chem.Conformer(sub.GetNumAtoms())
                    for sub_idx, parent_idx in enumerate(match):
                        pos = conf_parent.GetAtomPosition(parent_idx)
                        conf_sub.SetAtomPosition(sub_idx, pos)
                    
                    sub.RemoveAllConformers()
                    sub.AddConformer(conf_sub)
                    return True
            except:
                pass
            return False

        # Usage in loop:
        # 1. Align Original Molecule to Core template
        try:
            AllChem.GenerateDepictionMatching2DStructure(m, core_mol)
        except:
            AllChem.Compute2DCoords(m)
            
        # 2. Align fragments (Core/R-groups) to Original Molecule
        # Copy coords FROM original molecule TO fragment
        if not align_substructure_to_parent(fragment, m):
             AllChem.Compute2DCoords(fragment)
        ```

5.  **HTML Output (`sar_analysis_report.html`):**
    *   **Design:** Create a clean, modern, and visually appealing HTML page using CSS styling. Use modern CSS features (e.g., subtle shadows, smooth transitions, clean typography, proper color schemes, responsive design) to enhance readability and visual appeal.
    *   **Table Structure:** `Compound Key`, `Activity`, `Original Molecule`, `Core`, and variable R-groups.
    *   **Activity Heatmap:** Apply a background color gradient to Activity cells (Green for low values/high potency, Red for high values/low potency).
    *   **Image Handling:**
        *   Convert molecules to Base64 PNG strings.
        *   **Validation:** Check `if base64_str and len(base64_str) > 100`. Only embed valid images; otherwise, use a text placeholder (`<td>No Image</td>`).
    *   **Summary:** Include a brief text summary of SAR findings (correlation between R-groups and activity).

6.  **Analysis Text Output:**
    *   Based on the analysis results, generate a concise text analysis of the SAR findings.
    *   **Output Format:** Print this text directly in the conversation (do not save to a file).
    *   **Writing Style:** Concisely respond like the following writing style:
        *   No structural preference.
        *   C-2 carbon must be Sp² hybridized. X = O, NH, or S.
        *   Usually unsubstituted. Drugs with OH here are very active and have short duration of action.
        *   Reduction of the N-4, C-5 double-bond decreases activity.
        *   Halogens at the 2’ or 6’ positions enhance activity. Any other substituent decreases activity.
        *   Substitution at 3’, 4’, or 5’ positions decreases activity.
        *   Small electron withdrawing group (e.g., Cl) at C-7 increases activity.
        *   Substitution at C-6, C-8, or C-9 reduces activity.

**Output:**
*   Provide the final `sar_analysis_report.html` file.
*   Print the Analysis Text in the chat.