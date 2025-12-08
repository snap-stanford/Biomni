# Task: KEGG Pathway Enrichment Analysis

Perform KEGG pathway enrichment analysis on a list of genes.

## Input

A list of gene IDs (ENTREZ IDs, or convert from Ensembl/Gene Symbol):
```
1017
1019
1021
1026
1027
```

## Requirements

1. Convert gene IDs to ENTREZ IDs if needed (for KEGG analysis)
2. Perform KEGG pathway enrichment analysis:
   - Organism: human (hsa)
   - p-value adjustment: Benjamini-Hochberg (BH)
   - p-value cutoff: 0.05
3. Generate results table showing:
   - KEGG pathway ID and description
   - Gene count and ratio
   - p-value and adjusted p-value
4. Create a dotplot visualization showing top enriched pathways
5. Save results as CSV and plot as PNG

## Expected Output Format

Provide:
- KEGG enrichment results table (CSV format)
- Dotplot image file (PNG format)
- Brief summary of top enriched pathways
