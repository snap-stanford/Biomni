# Task: Differential Expression Analysis (DEG)

Perform differential expression analysis to identify genes that are differentially expressed between two groups (tumor vs normal).

## Input Data

**Expression matrix**: `input_data/expression_matrix.tsv`
- Rows: Genes (Ensembl IDs)
- Columns: Samples (TCGA barcodes)
- Values: Log2-transformed expression values

**Sample groups**:
- Samples ending with `-01A` are **Tumor** samples
- Samples ending with `-11A` are **Normal** samples

## Requirements

1. Load the expression matrix from `input_data/expression_matrix.tsv`
2. Separate samples into tumor and normal groups based on TCGA barcode
3. Calculate mean expression for each group
4. Calculate log2 Fold Change (log2FC) for each gene
5. Perform statistical test (t-test) to compare groups
6. Adjust p-values for multiple testing (FDR correction using Benjamini-Hochberg method)
7. Identify significant DEGs using thresholds:
   - |log2FC| > 1
   - Adjusted p-value (padj) < 0.05
8. Sort results by adjusted p-value
9. Report:
   - Total number of significant DEGs
   - Number of up-regulated and down-regulated genes
   - Top 10 most significant DEGs

## Expected Output Format

Provide:
- Summary statistics (number of DEGs found)
- Table showing top DEGs with columns: Gene ID, log2FoldChange, pvalue, padj
- Brief explanation of the analysis method
