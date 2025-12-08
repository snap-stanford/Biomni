# Task: Create Volcano Plot

Create a volcano plot to visualize differential expression analysis results.

## Input Data

**DEG results**: `input_data/deg_results.tsv`
- Columns:
  - `Gene_ID`: Gene identifier
  - `log2FoldChange`: Log2 fold change value
  - `pvalue`: P-value from statistical test

## Requirements

1. Load the DEG results from `input_data/deg_results.tsv`
2. Create a volcano plot with:
   - X-axis: log2FoldChange
   - Y-axis: -log10(p-value)
3. Color-code points based on significance:
   - Significant and up-regulated: Red (log2FC > 1, p < 0.05)
   - Significant and down-regulated: Blue (log2FC < -1, p < 0.05)
   - Not significant: Gray
4. Add threshold lines:
   - Vertical lines at log2FC = ±1
   - Horizontal line at p-value = 0.05 (or -log10(0.05))
5. Include appropriate labels, title, and legend
6. Save as PNG file

## Expected Output Format

Provide:
- Volcano plot image file (PNG format, named `volcano_plot.png`)
- Brief description of the plot
- Count of significant genes in each category (up, down, not significant)
