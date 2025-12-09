# Answer: Volcano Plot

## Plot Description

The volcano plot visualizes differential expression results, showing the relationship between fold change (x-axis) and statistical significance (y-axis). Genes in the upper left and upper right quadrants are significantly differentially expressed.

## Summary Statistics

- **Total genes**: 30
- **Significant up-regulated** (log2FC > 1, p < 0.05): 6
- **Significant down-regulated** (log2FC < -1, p < 0.05): 10
- **Not significant**: 14

## Plot Features

- **X-axis**: log2FoldChange (fold change on log2 scale)
- **Y-axis**: -log10(p-value) (statistical significance)
- **Color coding**:
  - Red: Up-regulated and significant
  - Blue: Down-regulated and significant
  - Gray: Not significant
- **Threshold lines**: 
  - Vertical at log2FC = ±1
  - Horizontal at p = 0.05 (-log10(0.05) ≈ 1.3)

## Interpretation

- **Upper right quadrant**: Significantly up-regulated genes (high fold change, low p-value)
- **Upper left quadrant**: Significantly down-regulated genes (low fold change, low p-value)
- **Lower regions**: Not significantly different genes
- **Distance from origin**: Indicates strength of differential expression
