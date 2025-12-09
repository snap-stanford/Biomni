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

## Interpretation

- **PC1**: Dominant axis separating Tumor vs Normal in this dataset.
- **PC2**: Adds minor secondary variation within each group.
- **Sample clustering**: Samples of the same type cluster together, indicating similar expression profiles.
