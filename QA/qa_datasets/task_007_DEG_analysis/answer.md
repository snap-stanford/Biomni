# Answer: Differential Expression Analysis Results

## Analysis Summary

- **Total genes analyzed**: 20
- **Tumor samples**: 5 (ending with -01A)
- **Normal samples**: 5 (ending with -11A)
- **Significant DEGs found**: 10
- **Up-regulated genes**: 3 (log2FC > 1, padj < 0.05)
- **Down-regulated genes**: 7 (log2FC < -1, padj < 0.05)

## Top 10 Most Significant DEGs

| Gene ID | log2FoldChange | pvalue | padj | Direction |
|---------|----------------|--------|------|-----------|
| ENSG00000066405 | -6.53 | 4.36e-09 | 5.21e-08 | Down |
| ENSG00000168481 | -5.74 | 5.21e-09 | 5.21e-08 | Down |
| ENSG00000168484 | -7.78 | 3.14e-08 | 1.57e-07 | Down |
| ENSG00000171885 | -4.38 | 3.92e-08 | 1.57e-07 | Down |
| ENSG00000134115 | -4.56 | 3.22e-08 | 1.57e-07 | Down |
| ENSG00000112782 | -4.33 | 5.10e-08 | 1.70e-07 | Down |
| ENSG00000233221 | 3.67 | 7.01e-08 | 2.00e-07 | Up |
| ENSG00000115361 | -3.93 | 9.00e-08 | 2.25e-07 | Down |
| ENSG00000230798 | 3.71 | 5.53e-07 | 1.23e-06 | Up |
| ENSG00000236212 | 3.18 | 1.35e-06 | 2.70e-06 | Up |

## Analysis Method

### Step 1: Load Data and Separate Groups
```python
import pandas as pd
import numpy as np

df = pd.read_csv('input_data/expression_matrix.tsv', sep='\t', index_col=0)

# Separate tumor and normal samples
tumor_samples = [col for col in df.columns if '-01A' in col]
normal_samples = [col for col in df.columns if '-11A' in col]

tumor_df = df[tumor_samples]
normal_df = df[normal_samples]
```

### Step 2: Calculate Log2 Fold Change
Since data is already log2-transformed:
```python
tumor_mean = tumor_df.mean(axis=1)
normal_mean = normal_df.mean(axis=1)
log2fc = tumor_mean - normal_mean
```

### Step 3: Perform t-test
```python
from scipy.stats import ttest_ind

p_values = []
for gene in df.index:
    t_stat, p_val = ttest_ind(tumor_df.loc[gene], normal_df.loc[gene], equal_var=False)
    p_values.append(p_val)
```

### Step 4: Multiple Testing Correction
```python
from statsmodels.stats.multitest import multipletests

padj = multipletests(p_values, method='fdr_bh')[1]
```

### Step 5: Filter Significant DEGs
```python
deg_results = pd.DataFrame({
    'log2FoldChange': log2fc,
    'pvalue': p_values,
    'padj': padj
})

significant_degs = deg_results[
    (abs(deg_results['log2FoldChange']) > 1) & 
    (deg_results['padj'] < 0.05)
]
```

## Interpretation

- **log2FoldChange > 1**: Gene is up-regulated in tumor (at least 2-fold higher)
- **log2FoldChange < -1**: Gene is down-regulated in tumor (at least 2-fold lower)
- **padj < 0.05**: Statistically significant after multiple testing correction

## Notes

- Pre-filtering low expression genes is recommended before DEG analysis
- The thresholds (|log2FC| > 1, padj < 0.05) are common but can be adjusted based on biological context
- Results should be sorted by padj to identify most significant changes
