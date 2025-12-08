# Answer: Multiple Testing Correction Results

## FDR Correction Results

| Test | Original p-value | Adjusted p-value (padj) | Significant (α=0.05) |
|------|------------------|------------------------|---------------------|
| 1 | 0.001 | 0.010 | Yes (both) |
| 2 | 0.003 | 0.015 | Yes (both) |
| 3 | 0.015 | 0.050 | Yes (both) |
| 4 | 0.025 | 0.0625 | Yes → No |
| 5 | 0.05 | 0.10 | Yes → No |
| 6 | 0.08 | 0.133 | No (both) |
| 7 | 0.12 | 0.171 | No (both) |
| 8 | 0.15 | 0.188 | No (both) |
| 9 | 0.20 | 0.222 | No (both) |
| 10 | 0.30 | 0.30 | No (both) |

## Summary

- **Total tests**: 10
- **Significant before correction (p < 0.05)**: 5 tests
- **Significant after FDR correction (padj < 0.05)**: 3 tests
- **Tests that lost significance**: 2 tests (tests 4 and 5)

## Explanation

### What is FDR Correction?

**False Discovery Rate (FDR)** correction controls the expected proportion of false positives among all rejected hypotheses. The Benjamini-Hochberg method is a widely used FDR correction procedure.

### Why is it Necessary?

When performing multiple statistical tests:
- **Without correction**: If you test 1000 hypotheses at α = 0.05, you expect ~50 false positives by chance alone
- **With FDR correction**: Controls the proportion of false discoveries, making results more reliable

### Benjamini-Hochberg Method

1. Sort p-values in ascending order
2. For each p-value, calculate: `padj = p × (total_tests / rank)`
3. Apply monotonicity constraint (ensure padj increases)

## Example Code

### Python (statsmodels)

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

p_values = np.array([0.001, 0.003, 0.015, 0.025, 0.05, 
                     0.08, 0.12, 0.15, 0.20, 0.30])

# Apply FDR correction (Benjamini-Hochberg)
rejected, padj, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

print("Original p-values:", p_values)
print("Adjusted p-values:", padj)
print("Significant tests:", np.sum(rejected))
```

### R

```r
p_values <- c(0.001, 0.003, 0.015, 0.025, 0.05, 
              0.08, 0.12, 0.15, 0.20, 0.30)

# Apply FDR correction
padj <- p.adjust(p_values, method = "BH")

# Check significance
significant <- padj < 0.05
sum(significant)
```

## Notes

- **FDR** is less conservative than Bonferroni correction (which controls Family-Wise Error Rate)
- **FDR** is appropriate when you want to control the proportion of false discoveries
- **Bonferroni** is more appropriate when you need to control the probability of ANY false positive
