# Answer: Heatmap Visualization

## Heatmap Description

The heatmap visualizes expression patterns of the top 50 differentially expressed genes across tumor and normal samples. The hierarchical clustering reveals distinct expression profiles that clearly separate tumor samples from normal samples.

## Key Features

- **Rows**: Top 50 DEGs (25 up-regulated, 25 down-regulated)
- **Columns**: Samples (tumor in red, normal in blue)
- **Color scale**: Z-score normalized expression (red = high, blue = low)
- **Clustering**: Both genes and samples are clustered to reveal patterns

## Visualization Elements

1. **Dendrograms**: Show hierarchical clustering relationships
2. **Sample annotation**: Color bar indicating sample type (Tumor/Normal)
3. **Gene labels**: Gene symbols displayed on y-axis
4. **Color legend**: Explains sample type colors

## Example Code

### Python (seaborn)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Prepare data (genes × samples)
heatmap_data = expression_matrix.loc[top_genes]

# Create sample annotations
sample_info = pd.DataFrame({
    'SampleType': ['Tumor' if c in tumor_samples else 'Normal' 
                   for c in heatmap_data.columns]
}, index=heatmap_data.columns)

# Color mapping
palette = {'Tumor': 'red', 'Normal': 'blue'}
sample_colors = sample_info['SampleType'].map(palette)

# Generate clustered heatmap
sns.set(font_scale=0.6)
g = sns.clustermap(
    heatmap_data,
    z_score=0,  # Normalize by row (z-score)
    cmap='vlag',  # Diverging colormap
    col_colors=sample_colors,  # Sample type annotation
    yticklabels=True,
    xticklabels=False,  # Hide if too many samples
    figsize=(12, 10)
)

g.ax_heatmap.set_ylabel('Genes')
g.ax_heatmap.set_xlabel('Samples')
g.fig.suptitle('Heatmap of Top DEGs', fontsize=16)

# Add legend
handles = [plt.Rectangle((0,0),1,1, color=palette[l]) for l in palette]
plt.legend(handles, palette.keys(), title='Sample Type')

plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
```

### R (pheatmap)

```r
library(pheatmap)

# Prepare data
heatmap_data <- expression_matrix[top_genes, ]

# Sample annotations
annotation_col <- data.frame(
  SampleType = ifelse(colnames(heatmap_data) %in% tumor_samples, 
                      "Tumor", "Normal"),
  row.names = colnames(heatmap_data)
)

# Color annotation
ann_colors <- list(SampleType = c(Tumor = "red", Normal = "blue"))

# Generate heatmap
pheatmap(heatmap_data,
         scale = "row",  # Z-score normalization
         color = colorRampPalette(c("blue", "white", "red"))(100),
         annotation_col = annotation_col,
         annotation_colors = ann_colors,
         show_rownames = TRUE,
         show_colnames = FALSE,
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         filename = "heatmap.png",
         width = 12,
         height = 10)
```

## Interpretation

- **Clustering patterns**: Samples of the same type cluster together, indicating distinct expression profiles
- **Color patterns**: Red regions indicate high expression, blue indicates low expression
- **Gene clusters**: Genes with similar expression patterns cluster together, potentially indicating co-regulation

## Notes

- Z-score normalization (by row) makes expression levels comparable across genes
- Clustering helps identify patterns and relationships
- Sample annotations make it easy to see group-specific patterns
