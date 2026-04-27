# Flow Cytometry Analysis — Murine Backbone Panel

Standard operating procedure for preprocessing, gating, clustering, and immune cell annotation of unmixed murine flow cytometry data using the backbone panel defined in `murine_backbone/`.

---

## Panel Reference

Fluorochrome–marker mapping: `murine_backbone/fluorochromes.csv`

Channels labelled `Drop-in` in that file are not used in this pipeline unless specified.

---

## Input

- **Format:** Unmixed `.fcs` files (one file per sample)
- **Expected channels:** All non-drop-in fluorochromes in `murine_backbone/fluorochromes.csv` must be present
- **FMO controls:** One FMO `.fcs` file per non-drop-in marker; used to set gate boundaries in Step 2. FMOs must be acquired and unmixed under the same conditions as experimental samples. If FMOs are not provided, gate boundaries must be set manually.
- **FMO controls (optional):** One FMO `.fcs` file per non-drop-in marker. FMOs must be acquired and unmixed under the same conditions as experimental samples. If FMOs are not provided, the pipeline falls back to automated gate placement (see Step 2).
- **Multiple samples:** Each sample is processed independently; outputs are delivered as separate `.zip` archives

---

## Step 1 — Quality Control and Preprocessing

### 1.1 Time-based QC
- Flag and exclude events with anomalous acquisition time patterns (flow rate instability, pressure drops)
- <!-- Stop analysis and report if >X% of events fall in aberrant time windows -->

### 1.2 Debris Exclusion
- Gate on FSC-A vs SSC-A to isolate intact single cells
- Exclude debris (low FSC-A / low SSC-A)

### 1.3 Doublet Exclusion
- Gate FSC-A vs FSC-H; retain singlets along the linear diagonal

### 1.4 Viability Exclusion
- Gate on Comp-APC-Fire 810-A::CD45 vs Comp-LIVE DEAD NIR-A::L_D to isolate CD45+ cells
- <!-- Stop condition: If live cell yield after viability exclusion falls below [threshold — customize], halt analysis and return a QC failure report specifying: sample name, total events acquired, % live, recommended action -->

### 1.5 QC Report Fields
Each sample produces a QC summary containing:
- Total events acquired
- Events after debris gate (% retained)
- Events after doublet gate (% retained)
- Events after viability gate (% live)
- PASS / FAIL status with failure reason

---

## Step 2 — Immune Cell Gating

- Gate hierarchy (population names, marker logic, parent–child relationships): `murine_backbone/gating_strategy.yaml`
- Plot axes for each gate: `murine_backbone/plot_gates_axes.csv`
- **FMO-guided gating:** For each gate, overlay the corresponding FMO control on the same axes to define the negative boundary. The FMO sets the upper limit of background fluorescence for the missing channel; the positive gate is placed just above this boundary. If multiple FMOs are needed for a single 2D gate (e.g., both axes have an FMO), apply each independently.
- **Gate boundary setting:**
  - **FMOs provided:** Overlay the FMO for the relevant channel on each gate plot. Place the positive gate boundary just above the upper edge of the FMO distribution. For 2D gates where both axes have an FMO, apply each independently.
  - **No FMOs provided:** Use automated boundary detection (e.g., valley finding on the fluorescence histogram or density-based inflection point) to place gates. Add a warning in the QC report noting that gates were set without FMO controls and should be reviewed manually before use in publication.

---

## Step 3 — Outputs

All outputs are packaged per sample as a `.zip` archive named `{sample_name}_flow_results.zip`.

### 3.1 Gating Hierarchy Text File
- Filename: `gating_hierarchy.txt`
- Format: Indented tree showing each gate node, parent population, markers used, and event counts / % of parent

### 3.2 Gate Plots
- Filename: `gates/gate_{population_name}.png` (one plot per gate in the hierarchy)
- Axes per gate follow `plot_gates_axes.csv`
- Each plot includes: gate boundary overlay, population label, % of parent population
- <!-- Specify plot resolution, colormap, point size, and density rendering preferences here -->

### 3.3 Population Count Table
- Filename: `population_counts.tsv`
- Columns: `Sample`, `Population`, `Count`, `Percent_of_CD45+`, `Percent_of_Parent`
- One row per terminal population in the gating hierarchy

### 3.4 Bar Plot — Population Frequencies
- Filename: `barplot_populations.png`
- X-axis: Immune cell population (fixed order — see below)
- Y-axis: % of CD45+ cells
- Scale: Log scale if the range of values spans more than 2 orders of magnitude; otherwise linear
- Each bar is labeled with the exact % value
- Population order (left → right):

  1. B cells
  2. pDCs
  3. NK cells
  4. CD3+ T-cells
  5. CD4 T-cells
  6. T regs
  7. CD8 T-cells
  8. cDCs
  9. Macrophages
  10. Ly6C low monocytes
  11. Ly6C high monocytes
  12. Neutrophils

- <!-- Specify bar color scheme, figure dimensions, and font size preferences here -->

### 3.5 UMAP of Annotated Populations
- Filename: `umap_annotated.png`
- Cells downsampled to [N — customize] per sample before embedding
- Markers used for UMAP: all non-drop-in channels from `murine_backbone/fluorochromes.csv`
- Cells colored by annotated population label
- Legend shows population name with cell count
- <!-- Specify UMAP parameters (n_neighbors, min_dist), point size, and colormap here -->

---

## Multi-Sample Behavior

- Each `.fcs` file is processed independently through the full pipeline
- A separate `.zip` is produced per sample
- If a sample fails QC (Step 1.4), it is excluded from further processing and a `QC_FAILED_{sample_name}.txt` report is returned instead of the results zip

---

## Additional Behavior

- **Tone:** Maintain a professional, technical tone focused on scientific workflows unless explicitly asked for design or visualization.
- **Scope:** Do not deviate into unrelated domains; prioritize flow cytometry analysis tasks.
- **Knowledge file usage:** When asked how uploaded files are used in a GPT, explain that knowledge files are attached in the GPT configuration and should be referred to by purpose or filename in instructions, not by local filesystem paths; the GPT retrieves relevant content automatically.
- **Runtime:** Report the runtime when returning results. 
