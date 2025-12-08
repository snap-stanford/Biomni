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

## Example Code

### Python (matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load DEG results
deg_results = pd.read_csv('input_data/deg_results.tsv', sep='\t')

# Calculate -log10(p-value)
deg_results['neg_log10_p'] = -np.log10(deg_results['pvalue'])

# Define significance categories
deg_results['significance'] = 'Not significant'
deg_results.loc[(deg_results['log2FoldChange'] > 1) & 
                (deg_results['pvalue'] < 0.05), 'significance'] = 'Up-regulated'
deg_results.loc[(deg_results['log2FoldChange'] < -1) & 
                (deg_results['pvalue'] < 0.05), 'significance'] = 'Down-regulated'

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot points by category
for category, color in [('Up-regulated', 'red'), 
                         ('Down-regulated', 'blue'), 
                         ('Not significant', 'gray')]:
    data = deg_results[deg_results['significance'] == category]
    ax.scatter(data['log2FoldChange'], data['neg_log10_p'], 
               c=color, alpha=0.5, s=20, label=category)

# Add threshold lines
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.axvline(x=-1, color='black', linestyle='--', linewidth=1)
ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1)

# Labels and title
ax.set_xlabel('log2 Fold Change', fontsize=12)
ax.set_ylabel('-log10(p-value)', fontsize=12)
ax.set_title('Volcano Plot: Differential Expression', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volcano_plot.png', dpi=300, bbox_inches='tight')
```

### R (ggplot2)

```r
library(ggplot2)

# Load DEG results
deg_results <- read.delim('input_data/deg_results.tsv', sep='\t')

# Calculate -log10(p-value)
deg_results$neg_log10_p <- -log10(deg_results$pvalue)

# Define significance
deg_results$significance <- "Not significant"
deg_results$significance[deg_results$log2FoldChange > 1 & 
                        deg_results$pvalue < 0.05] <- "Up-regulated"
deg_results$significance[deg_results$log2FoldChange < -1 & 
                        deg_results$pvalue < 0.05] <- "Down-regulated"

# Create plot
ggplot(deg_results, aes(x=log2FoldChange, y=neg_log10_p, color=significance)) +
  geom_point(alpha=0.5, size=1) +
  geom_vline(xintercept=c(-1, 1), linetype="dashed", color="black") +
  geom_hline(yintercept=-log10(0.05), linetype="dashed", color="black") +
  scale_color_manual(values=c("Up-regulated"="red", 
                              "Down-regulated"="blue", 
                              "Not significant"="gray")) +
  labs(x="log2 Fold Change", 
       y="-log10(p-value)", 
       title="Volcano Plot: Differential Expression") +
  theme_minimal() +
  theme(legend.position="right")

ggsave("volcano_plot.png", width=10, height=8, dpi=300)
```

## Interpretation

- **Upper right quadrant**: Significantly up-regulated genes (high fold change, low p-value)
- **Upper left quadrant**: Significantly down-regulated genes (low fold change, low p-value)
- **Lower regions**: Not significantly different genes
- **Distance from origin**: Indicates strength of differential expression

## Notes

- Volcano plots are excellent for visualizing large-scale differential expression results
- The plot helps identify genes with both large fold changes and statistical significance
- Thresholds can be adjusted based on biological context
