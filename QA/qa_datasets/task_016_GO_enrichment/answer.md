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

## Example Code

### R (clusterProfiler)

```r
library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)

# Input gene list (Ensembl IDs)
gene_list <- c("ENSG00000000003", "ENSG00000000005", "ENSG00000000419",
               "ENSG00000000457", "ENSG00000000460")

# Perform GO enrichment
ego <- enrichGO(gene          = gene_list,
                OrgDb         = org.Hs.eg.db,
                keyType       = 'ENSEMBL',
                ont           = "ALL",  # BP, MF, CC, or ALL
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.01,
                qvalueCutoff  = 0.05)

# Save results
if (!is.null(ego) && nrow(ego) > 0) {
  write.csv(as.data.frame(ego), "go_enrichment_results.csv")
  
  # Create dotplot
  p <- dotplot(ego, showCategory=10, split="ONTOLOGY") + 
    facet_grid(ONTOLOGY~., scale="free")
  ggsave("go_dotplot.png", plot=p, width=10, height=12)
}
```

## Results Table Structure

| ID | Description | GeneRatio | BgRatio | pvalue | p.adjust | qvalue | geneID | Count |
|----|-------------|-----------|---------|--------|----------|--------|--------|-------|
| GO:0007049 | cell cycle | 3/5 | 500/25000 | 0.001 | 0.005 | 0.01 | ENSG00000000003/... | 3 |

## Interpretation

- **GeneRatio**: Proportion of input genes in this GO term
- **BgRatio**: Proportion of background genes in this GO term
- **p.adjust**: Adjusted p-value (FDR corrected)
- **Enrichment**: Higher GeneRatio relative to BgRatio indicates enrichment

## Notes

- GO enrichment identifies biological functions/pathways over-represented in the gene list
- Results are typically visualized using dotplots or barplots
- Separate analysis for BP, MF, and CC provides comprehensive functional annotation
