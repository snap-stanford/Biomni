I need to perform a differential expression analysis on the dataset `input_data/expression_matrix.tsv`. The samples ending in `-01A` are Tumor, and those ending in `-11A` are Normal.

Could you identify the differentially expressed genes (DEGs) between these two groups? Please use a t-test and correct the p-values using the Benjamini-Hochberg FDR method.

I'm interested in genes with an absolute log2 fold change greater than 1 and an adjusted p-value less than 0.05. Please list the top 10 most significant DEGs.
