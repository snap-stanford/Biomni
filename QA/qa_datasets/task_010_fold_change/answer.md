# Answer: Log2 Fold Change Calculation

## Calculation

### Given Values
- **Group 1 (Tumor) mean**: 12.5 (already log2-transformed)
- **Group 2 (Normal) mean**: 10.3 (already log2-transformed)

### Log2 Fold Change Formula (data already log2-transformed)
```
log2FC = mean_group1 - mean_group2
log2FC = 12.5 - 10.3 = 2.2
```

### Result
- **Log2 Fold Change (log2FC)**: 2.2

## Interpretation

### Up-regulation
- **log2FC = 2.2 > 0**: The gene is **up-regulated** in Group 1 (Tumor) compared to Group 2 (Normal)

### Fold Change in Linear Scale
To convert log2FC to linear fold change:
```
Fold Change = 2^(log2FC)
Fold Change = 2^2.2 = 4.59
```

This means the gene expression is **4.59 times higher** in Tumor compared to Normal.

### Common Thresholds
- **|log2FC| > 1**: At least 2-fold change (commonly used threshold)
- **|log2FC| > 2**: At least 4-fold change (more stringent)
- **log2FC = 2.2**: Exceeds the 2-fold threshold, indicating strong up-regulation
