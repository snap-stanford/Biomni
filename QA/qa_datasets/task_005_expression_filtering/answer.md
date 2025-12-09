# Answer: Expression Filtering Results

## Filtering Summary

- **Original number of genes**: 30
- **Number of genes after filtering**: 26
- **Genes retained**: 86.7%
- **Genes removed**: 4 (13.3%)

## Filtering Criteria Applied

Genes were retained if they had log2 expression > 1 in at least 20% of samples (i.e., at least 2 out of 10 samples).

## Genes Removed (low expression)

The following genes had log2 expression > 1 in fewer than 2 samples:
- **ENSG00000000005**
- **ENSG00000000938**
- **ENSG00000001631**
- **ENSG00000002726**

## Rationale

Low expression genes often represent technical noise or very low biological signal. Including them increases multiple testing burden without adding meaningful information. Filtering improves statistical power for detecting true differential expression.

The log2 expression > 1 threshold corresponds to raw expression > 2, which helps remove genes that are essentially not expressed. The 20% sample threshold ensures the gene is expressed in a reasonable number of samples.
