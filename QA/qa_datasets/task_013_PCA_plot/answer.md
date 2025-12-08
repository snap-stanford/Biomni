# Answer: PCA Analysis and Visualization

## PCA Summary

- **Total variance explained by PC1**: 97.9%
- **Total variance explained by PC2**: 1.1%
- **Cumulative variance (PC1 + PC2)**: 99.0%

## Plot Description

Using the provided expression matrix (`input_data/expression_matrix.tsv`), PCA was performed on the sample profiles (samples × genes). Without additional scaling, PC1 captures almost all variance and cleanly separates Tumor (`-01A`) vs Normal (`-11A`); PC2 adds minor within-group variation.

## Key Observations

- **Sample separation**: Tumor and Normal samples form distinct clusters.
- **PC1**: Separates Tumor (right) from Normal (left).
- **PC2**: Captures additional within-group variation.
- **Group counts**: Tumor samples = 5, Normal samples = 5.

## Example Code

### Python (sklearn, matplotlib)

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load expression data (genes x samples)
df = pd.read_csv('input_data/expression_matrix.tsv', sep='\t')
expression_matrix = df.set_index('Gene_ID').T  # samples x genes

# Perform PCA (no extra scaling)
pca = PCA()
pca_result = pca.fit_transform(expression_matrix)

# Extract PC1 and PC2
pc1 = pca_result[:, 0]
pc2 = pca_result[:, 1]

# Calculate variance explained
variance_pc1 = pca.explained_variance_ratio_[0] * 100
variance_pc2 = pca.explained_variance_ratio_[1] * 100

# Sample groups
sample_names = expression_matrix.index.tolist()
groups = ['Tumor' if s.endswith('-01A') else 'Normal' for s in sample_names]

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
for group, color in [('Tumor', 'red'), ('Normal', 'blue')]:
    mask = [g == group for g in groups]
    ax.scatter(pc1[mask], pc2[mask], c=color, label=group, alpha=0.6, s=50)

ax.set_xlabel(f'PC1 ({variance_pc1:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({variance_pc2:.1f}% variance)', fontsize=12)
ax.set_title('PCA Plot: Gene Expression', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_plot.png', dpi=300, bbox_inches='tight')
```

### R (prcomp, ggplot2)

```r
library(ggplot2)

# Load expression data
df <- read.delim('input_data/expression_matrix.tsv', sep='\t')
rownames(df) <- df$Gene_ID
df$Gene_ID <- NULL

# Perform PCA (samples x genes, no extra scaling)
pca_result <- prcomp(t(df), scale.=FALSE)

# Extract PC1 and PC2
pc_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Sample = rownames(pca_result$x)
)

# Add group labels
pc_data$Group <- ifelse(grepl('-01A$', pc_data$Sample), 'Tumor', 'Normal')

# Variance explained
variance_pc1 <- summary(pca_result)$importance[2, 1] * 100
variance_pc2 <- summary(pca_result)$importance[2, 2] * 100

# Create plot
ggplot(pc_data, aes(x=PC1, y=PC2, color=Group)) +
  geom_point(size=2, alpha=0.6) +
  scale_color_manual(values=c("Tumor"="red", "Normal"="blue")) +
  labs(x=paste0("PC1 (", round(variance_pc1, 1), "% variance)"),
       y=paste0("PC2 (", round(variance_pc2, 1), "% variance)"),
       title="PCA Plot: Gene Expression") +
  theme_minimal() +
  theme(legend.position="right")

ggsave("pca_plot.png", width=10, height=8, dpi=300)
```

## Interpretation

- **PC1**: Dominant axis separating Tumor vs Normal in this dataset.
- **PC2**: Adds minor secondary variation within each group.
- **Sample clustering**: Samples of the same type cluster together, indicating similar expression profiles.

## Notes

- Standardization (scaling) is recommended before PCA.
- Variance explained values above are computed from the provided `input_data/expression_matrix.tsv`.
