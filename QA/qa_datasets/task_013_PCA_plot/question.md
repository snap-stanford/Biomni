# Task: Perform PCA and Create PCA Plot

Perform Principal Component Analysis (PCA) on gene expression data and visualize the results.

## Input Data

**Expression matrix**: `input_data/expression_matrix.tsv`
- Rows: Genes (Gene IDs)
- Columns: Samples
- Values: Log2-transformed expression values

**Sample groups**:
- Samples ending with `-01A` are **Tumor** samples
- Samples ending with `-11A` are **Normal** samples

## Requirements

1. Load the expression matrix from `input_data/expression_matrix.tsv`
2. Perform PCA on the expression matrix (samples as observations, genes as features)
3. Extract principal components (PC1, PC2)
4. Create a 2D scatter plot:
   - X-axis: PC1
   - Y-axis: PC2
5. Color-code samples by group (Tumor vs Normal)
6. Include:
   - Percentage of variance explained by each PC
   - Legend for groups
   - Title and axis labels
7. Save as PNG file

## Expected Output Format

Provide:
- PCA plot image file (PNG format, named `pca_plot.png`)
- Variance explained by PC1 and PC2
- Brief interpretation of the plot (do samples separate by group?)
