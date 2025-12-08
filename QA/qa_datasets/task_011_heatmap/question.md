# Task: Generate Heatmap Visualization

Create a clustered heatmap to visualize gene expression patterns across samples.

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
2. Create a clustered heatmap with:
   - Rows: Genes
   - Columns: Samples
   - Color scale: Expression levels (z-score normalized by row recommended)
   - Hierarchical clustering: Both rows and columns
3. Add sample type annotations (color bar for groups)
4. Include appropriate labels:
   - Gene names/labels on y-axis
   - Title
   - Color legend for sample types
5. Save as PNG file with high resolution (300 DPI)

## Expected Output Format

Provide:
- Heatmap image file (PNG format, named `heatmap.png`)
- Brief description of what the heatmap shows
- Explanation of clustering patterns
