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

**False Discovery Rate (FDR)** correction controls the expected proportion of false positives among all rejected hypotheses. The Benjamini-Hochberg method is a widely used FDR correction procedure.

When performing multiple statistical tests, without correction you would expect many false positives by chance alone. With FDR correction, the proportion of false discoveries is controlled, making results more reliable.
