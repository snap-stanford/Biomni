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

## Interpretation

- **log2FoldChange > 1**: Gene is up-regulated in tumor (at least 2-fold higher)
- **log2FoldChange < -1**: Gene is down-regulated in tumor (at least 2-fold lower)
- **padj < 0.05**: Statistically significant after multiple testing correction

The analysis was performed by separating samples into tumor and normal groups based on TCGA barcode, calculating mean expression for each group, computing log2 fold change, performing t-tests, and applying Benjamini-Hochberg FDR correction for multiple testing.
