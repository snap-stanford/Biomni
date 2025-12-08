# Task: Multiple Testing Correction (FDR)

Apply False Discovery Rate (FDR) correction to a list of p-values using the Benjamini-Hochberg method.

## Input

A list of p-values from multiple statistical tests:
```
0.001, 0.003, 0.015, 0.025, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30
```

## Requirements

1. Apply Benjamini-Hochberg FDR correction to the p-values
2. Report:
   - Original p-values
   - Adjusted p-values (padj)
   - Number of significant tests at α = 0.05 (before and after correction)
3. Explain what FDR correction does and why it's necessary

## Expected Output Format

Provide:
- Table showing original and adjusted p-values
- Summary of significant tests before/after correction
- Brief explanation of FDR correction
