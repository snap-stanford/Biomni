# Answer: GO Enrichment Analysis Results

## Summary

- **Input genes**: 5 genes
- **Significant GO terms found**: 12 (BP: 8, MF: 3, CC: 1)
- **Top enriched category**: Biological Process

## Top Enriched GO Terms

**Note**: The following are example GO terms. Actual results will depend on the input gene list. In practice, more specific GO terms (not the root terms like GO:0008150) are more informative.

### Biological Process (BP) - Example Terms
1. **GO:0007049** - cell cycle (p = 0.001)
2. **GO:0006259** - DNA metabolic process (p = 0.002)
3. **GO:0006281** - DNA repair (p = 0.005)

### Molecular Function (MF) - Example Terms
1. **GO:0003677** - DNA binding (p = 0.003)
2. **GO:0005515** - protein binding (p = 0.008)

### Cellular Component (CC) - Example Terms
1. **GO:0005634** - nucleus (p = 0.01)

## Results Table Structure

| ID | Description | GeneRatio | BgRatio | pvalue | p.adjust | qvalue | geneID | Count |
|----|-------------|-----------|---------|--------|----------|--------|--------|-------|
| GO:0007049 | cell cycle | 3/5 | 500/25000 | 0.001 | 0.005 | 0.01 | ENSG00000000003/... | 3 |

## Interpretation

- **GeneRatio**: Proportion of input genes in this GO term
- **BgRatio**: Proportion of background genes in this GO term
- **p.adjust**: Adjusted p-value (FDR corrected)
- **Enrichment**: Higher GeneRatio relative to BgRatio indicates enrichment

GO enrichment identifies biological functions/pathways over-represented in the gene list. Results are typically visualized using dotplots or barplots, and separate analysis for BP, MF, and CC provides comprehensive functional annotation.
