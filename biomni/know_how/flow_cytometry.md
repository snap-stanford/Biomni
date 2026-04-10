# Flow Cytometry Analysis Agent

---

## Metadata

**Short Description**: Comprehensive guide for automating and assisting with flow cytometry data analysis, including gating, population identification, and result visualization.

**Authors**: MSK Team

**Version**: 1.0

**Last Updated**: April 2026

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed


---

## Overview

This agent provides best practices, workflows, and automation for flow cytometry data analysis. It supports common tasks such as data import, gating strategies, population quantification, and visualization.

## Recommended Workflow

### 1. Data Import
- Accepts FCS data format
- Uses fcsparser or flowio for .fcs file parsing

### 2. Quality Control
- Remove debris, doublets, and dead cells
- Notify user if files are poor-quality
- Visualize scatter plots (FSC/SSC)

### 3. Gating Strategy
- Time gate
- Apply sequential gates (e.g., lymphocyte, singlet, live/dead)
- Use polygon, rectangle, or threshold gates
- Generate and save plots of final gates

### 4. Population Identification
- Use Leiden clustering
- Quantify cell populations based on marker expression
- Export population statistics

### 5. Visualization
- Generate histograms, dot plots, density plots
- Save plots to output directory

### 6. Export Results
- Save gated data and summary statistics as CSV
- Store plots in a temporary directory (see below for cleanup)

## Best Practices
- Always perform compensation and QC before gating
- Prefer asinh transformation. Notify user if there is a reason to use a different transformation
- Document gating strategy and thresholds

## Recommended Packages
### Programming language selection
- Prefer R for statistical analysis and calculations, such as transformation, normalization, unmixing, correlations, and quantification
- Prefer Python for image flow data analysis, such as microscopy and image visualization

### R
- core: flowCore (preferred), flowWorkspace
- QC: flowClean (preferred), flowAI
- Processing and transformation: flowStats (preferred), flowTrans
- Statistical analysis: flowStats
- Compensation and unmixing: flowSpecs, flowUnmix
- Visualization: flowViz, ggplot2, cytoExploreR
- Dimensionality reduction: t-SNE, UMAP
- Clustering: flowSOM, CATALYST
- Automatic gating: flowClust

### Python
- core: FlowKit, pandas, numpy
- Transformation, normalization, compensation, and unmixing: FlowUtils
- Data visualization: matplotlib, seaborn
- Clustering: mini-SOM, Phenograph
- Other: scanpy

## Cleanup Policy
- All temporary files and plots are stored in a `tmp/` directory
- The agent will automatically remove files in `tmp/` when the session is disconnected or exited

## References
- FlowCytometryTools: https://eyurtsev.github.io/FlowCytometryTools/
- Cytobank: https://www.cytobank.org/
- FlowJo: https://www.flowjo.com/

---

## Troubleshooting
- Ensure input files are correctly formatted
- Check for compensation issues if populations are not well separated
- Review gating strategy if results are unexpected
- Notify the user of any QC issues that may affect analysis
