# Task: GO Enrichment Analysis

Perform Gene Ontology (GO) enrichment analysis on a list of genes.

## Input

A list of gene IDs (Ensembl IDs or Gene Symbols):
```
ENSG00000000003
ENSG00000000005
ENSG00000000419
ENSG00000000457
ENSG00000000460
```

## Requirements

1. Perform GO enrichment analysis for:
   - Biological Process (BP)
   - Molecular Function (MF)
   - Cellular Component (CC)
2. Use appropriate statistical thresholds:
   - p-value cutoff: 0.01
   - q-value cutoff: 0.05
   - p-value adjustment: Benjamini-Hochberg (BH)
3. Generate results table showing:
   - GO term ID and description
   - Gene count and ratio
   - p-value and adjusted p-value
4. Create a dotplot visualization showing top enriched terms
5. Save results as CSV and plot as PNG

## Expected Output Format

Provide:
- GO enrichment results table (CSV format)
- Dotplot image file (PNG format)
- Brief summary of top enriched terms
