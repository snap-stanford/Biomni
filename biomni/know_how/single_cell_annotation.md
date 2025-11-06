# Single Cell RNA-seq Cell Type Annotation

---

## Metadata

**Short Description**: Best practices for annotating cell types in single-cell RNA-seq data using marker-based, automated, and reference-based approaches.

**Authors**: Distilled from "Single-cell best practices" by Luecken, M.D. et al.

**Affiliations**: Helmholtz Munich, Wellcome Sanger Institute, Harvard Medical School, and contributors

**Version**: 1.0

**Last Updated**: January 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed (with attribution to original source)

**Status**: Community-Reviewed

**Source**: https://www.sc-best-practices.org/cellular_structure/annotation.html

**Citation**: Luecken, M.D., Theis, F.J. et al. (2023). Current best practices in single-cell RNA-seq analysis: a tutorial. Molecular Systems Biology.

---

## Overview

Cell type annotation is the process of assigning cell type labels to clusters or individual cells in single-cell RNA-seq data. This guide covers three main approaches and their practical implementation.

## Three Annotation Approaches

### 1. Manual Marker-Based Annotation
Identify cell types by examining expression of known marker genes in each cluster.

**Tools**: Scanpy, Seurat
**Best for**: Small datasets, novel cell types, high confidence needs

### 2. Automated Annotation
Use pre-trained classifiers to automatically assign cell type labels.

**Tools**: CellTypist, scAnnotate
**Best for**: Standard tissues, quick preliminary annotation, large datasets

### 3. Reference-Based Label Transfer
Transfer labels from annotated reference datasets to your query data.

**Tools**: scArches, scANVI, Azimuth, SingleR
**Best for**: Well-characterized tissues, integration with public data

## Recommended Workflow

### Step 1: Quality Control First
- **Remove low-quality cells before annotation**
- Filter doublets (expected doublet rate: 0.8% per 1000 cells)
- Check for ambient RNA contamination
- Verify cluster quality and resolution

### Step 2: Initial Marker-Based Assessment

```python
# Scanpy example
import scanpy as sc

# Calculate marker genes for clusters
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Visualize top markers
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# Plot known markers
markers = {
    'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A'],
    'B cells': ['CD19', 'MS4A1', 'CD79A'],
    'Monocytes': ['CD14', 'FCGR3A', 'LYZ'],
    'NK cells': ['NCAM1', 'NKG7', 'GNLY']
}

sc.pl.dotplot(adata, markers, groupby='leiden')
```

### Step 3: Use Automated Tools for Validation

```python
# CellTypist example (fast, accurate for immune cells)
import celltypist
from celltypist import models

# Download immune cell model
model = models.Model.load(model='Immune_All_Low.pkl')

# Predict cell types
predictions = celltypist.annotate(adata, model=model, majority_voting=True)
adata = predictions.to_adata()
```

### Step 4: Reference-Based Refinement

```python
# scArches example for label transfer
import scarches as sca

# Load pre-trained reference model
model = sca.models.SCANVI.load_query_data(
    adata=adata,  # Your query data
    reference_model="path/to/reference_model"
)

# Transfer labels
model.train(max_epochs=100)
adata.obs['transferred_labels'] = model.predict()
```

## Best Practices

### Do's:
1. **Always combine multiple approaches** - Use marker-based validation even with automated tools
2. **Check cluster purity** - Ensure clusters represent single cell types
3. **Validate with multiple marker sets** - Don't rely on single markers
4. **Consider biological context** - Tissue type, disease state, developmental stage
5. **Document confidence levels** - Note uncertain annotations
6. **Use hierarchical annotation** - Broad categories first, then subtypes

### Don'ts:
1. **Don't over-cluster** - Too fine resolution creates artificial distinctions
2. **Don't ignore batch effects** - Correct before annotation
3. **Don't trust automation blindly** - Always validate predictions
4. **Don't mix cell states with cell types** - Activated vs. resting cells are states, not types
5. **Don't annotate low-quality cells** - Remove them first

## Common Pitfalls

### 1. Doublet Clusters
**Problem**: Clusters with markers from multiple cell types
**Solution**: Use doublet detection tools (Scrublet, DoubletFinder) before annotation

### 2. Ambient RNA
**Problem**: Background markers in all cells
**Solution**: Use SoupX or CellBender for decontamination

### 3. Over-interpretation of Small Clusters
**Problem**: Rare clusters may be technical artifacts
**Solution**: Require minimum cell count (e.g., >25 cells), validate with independent data

### 4. Reference Mismatch
**Problem**: Reference from different tissue/species/condition
**Solution**: Use tissue-matched references, check marker overlap before transfer

## Tool Selection Guide

| Scenario | Recommended Tool | Why |
|----------|------------------|-----|
| Immune cells (human) | CellTypist | Pre-trained on large immune atlases |
| Mouse tissues | scArches + Mouse Cell Atlas | Comprehensive mouse reference |
| Novel cell types | Manual + Scanpy/Seurat | Need domain expertise |
| Large datasets (>100k cells) | CellTypist | Fast, scalable |
| Cross-species | Manual markers | Limited reference transfer |
| Developmental data | scArches | Handles continuous states |

## Key Marker Genes by Cell Type

### Blood/Immune:
- **T cells**: CD3D, CD3E (all T cells); CD4, CD8A (subtypes)
- **B cells**: CD19, MS4A1 (CD20), CD79A
- **Monocytes/Macrophages**: CD14, CD68, LYZ
- **NK cells**: NCAM1 (CD56), NKG7, KLRD1
- **Dendritic cells**: FCER1A, CD1C

### Epithelial:
- **General epithelial**: EPCAM, KRT18, KRT19
- **Lung AT1**: AGER, PDPN
- **Lung AT2**: SFTPC, SFTPA1
- **Intestinal**: VIL1, MUC2

### Stromal:
- **Fibroblasts**: COL1A1, DCN, LUM
- **Endothelial**: PECAM1 (CD31), VWF, CDH5
- **Smooth muscle**: ACTA2, MYH11, TAGLN

## Validation Checklist

- [ ] Cluster purity: >80% cells with same label per cluster
- [ ] Marker consistency: Top DE genes match expected markers
- [ ] Biological plausibility: Expected proportions for tissue type
- [ ] Cross-method agreement: Manual and automated annotations align
- [ ] Reference quality: >70% cells successfully transferred
- [ ] Doublet check: No clusters with multi-lineage markers
- [ ] Documentation: Record confidence levels and uncertain calls

## Resources

### Tools:
- **Scanpy**: https://scanpy.readthedocs.io/
- **CellTypist**: https://www.celltypist.org/
- **scArches**: https://scarches.readthedocs.io/
- **Seurat**: https://satijalab.org/seurat/

### References:
- **PanglaoDB**: Database of marker genes
- **CellMarker**: Curated cell marker database
- **Human Cell Atlas**: Reference datasets

### Pre-trained Models:
- **CellTypist models**: 30+ tissue-specific models
- **Azimuth references**: PBMC, lung, kidney, etc.
- **scArches models**: Multiple tissue references

## Troubleshooting

**Issue**: All clusters look similar
→ Increase clustering resolution, check if data is normalized

**Issue**: Too many small clusters
→ Decrease resolution, merge similar clusters based on markers

**Issue**: Automated tool gives inconsistent results
→ Check input normalization, try multiple tools, fall back to manual

**Issue**: Can't find clear markers for cluster
→ May be transitional state, doublet, or low-quality cells

**Issue**: Reference transfer fails
→ Check batch correction, ensure overlapping gene sets, verify tissue match
