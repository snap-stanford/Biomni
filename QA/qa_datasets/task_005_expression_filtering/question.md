# Task: Filter Low Expression Genes

Filter out genes with low expression from a gene expression matrix.

## Background

In RNA-seq analysis, genes with very low expression across samples often represent noise and can negatively affect downstream statistical analyses. Filtering these genes improves statistical power and reduces false positives.

## Input Data

**File**: `input_data/expression_matrix.tsv`

A gene expression matrix (genes × samples) where values are log2-transformed expression levels.

Data structure:
- Rows: Genes (Ensembl IDs)
- Columns: Samples (TCGA barcodes)
- Values: Log2 expression values

## Requirements

1. Load the input data file: `input_data/expression_matrix.tsv`
2. Filter genes that have log2 expression > 1 in at least 20% of samples
3. Report:
   - Original number of genes
   - Number of genes after filtering
   - Percentage of genes retained
4. Save the filtered expression matrix

## Filtering Criteria

Keep genes where:
- Log2 expression > 1 in at least 20% of samples

This means: `sum(expression > 1) >= (total_samples * 0.20)`

## Expected Output Format

Provide:
- Summary statistics (before/after filtering)
- Filtered expression matrix saved as CSV or TSV file
