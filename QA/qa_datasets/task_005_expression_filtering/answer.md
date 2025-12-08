# Answer: Expression Filtering Results

## Filtering Summary

- **Original number of genes**: 30
- **Number of genes after filtering**: 26
- **Genes retained**: 86.7%
- **Genes removed**: 4 (13.3%)

## Filtering Criteria Applied

Genes were retained if they had log2 expression > 1 in at least 20% of samples (i.e., at least 2 out of 10 samples).

### Calculation

```python
import pandas as pd
import numpy as np

# Load the input data
df = pd.read_csv('input_data/expression_matrix.tsv', sep='\t', index_col=0)

# Calculate minimum number of samples required (20% of total)
min_samples = int(df.shape[1] * 0.20)  # 10 * 0.20 = 2

# Filter: keep genes with expression > 1 in at least min_samples
filtered_df = df[np.sum(df > 1, axis=1) >= min_samples]

print(f"Original number of genes: {df.shape[0]}")
print(f"Number of genes after filtering: {filtered_df.shape[0]}")
print(f"Percentage retained: {filtered_df.shape[0] / df.shape[0] * 100:.1f}%")

# Save filtered data
filtered_df.to_csv('filtered_expression_matrix.tsv', sep='\t')
```

## Genes Removed (low expression)

The following genes had log2 expression > 1 in fewer than 2 samples:
- **ENSG00000000005**
- **ENSG00000000938**
- **ENSG00000001631**
- **ENSG00000002726**

## Rationale

- **Why filter low expression genes?**
  - Low expression genes often represent technical noise or very low biological signal
  - Including them increases multiple testing burden without adding meaningful information
  - Filtering improves statistical power for detecting true differential expression

- **Why log2 > 1 threshold?**
  - Log2 expression > 1 corresponds to raw expression > 2
  - This threshold helps remove genes that are essentially not expressed
  - The 20% sample threshold ensures the gene is expressed in a reasonable number of samples

## Notes

- The filtered expression matrix should maintain the same structure (genes × samples)
- Gene IDs (row names) and sample IDs (column names) should be preserved
- This filtering step is typically performed before differential expression analysis
