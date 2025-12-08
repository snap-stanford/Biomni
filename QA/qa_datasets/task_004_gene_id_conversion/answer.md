# Answer: Gene ID Conversion Results

## Conversion Results

| Ensembl ID | Gene Symbol |
|------------|-------------|
| ENSG00000000003 | TSPAN6 |
| ENSG00000000005 | TNMD |
| ENSG00000000419 | DPM1 |
| ENSG00000000457 | SCYL3 |
| ENSG00000000460 | C1orf112 |

## Summary

- **Total input IDs**: 5
- **Successfully mapped**: 5 (100%)
- **Failed to map**: 0

## Method

The conversion was performed using the `org.Hs.eg.db` annotation package in R, which provides mappings between Ensembl IDs and Gene Symbols.

### Example Code (R)

```r
library(AnnotationDbi)
library(org.Hs.eg.db)

ensembl_ids <- c("ENSG00000000003", "ENSG00000000005", "ENSG00000000419", 
                 "ENSG00000000457", "ENSG00000000460")

gene_symbols <- mapIds(org.Hs.eg.db,
                       keys = ensembl_ids,
                       column = "SYMBOL",
                       keytype = "ENSEMBL",
                       multiVals = "first")

# Create mapping dataframe
mapping_df <- data.frame(
  ENSEMBL = names(gene_symbols),
  SYMBOL = gene_symbols,
  stringsAsFactors = FALSE
)
```

### Alternative Method (Python)

```python
import mygene

mg = mygene.MyGeneInfo()
ensembl_ids = ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419",
               "ENSG00000000457", "ENSG00000000460"]

results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
```

## Notes

- Some Ensembl IDs may not have corresponding Gene Symbols (will be NA)
- The mapping is based on current annotation databases and may change over time
- For large-scale conversions, batch processing is recommended
