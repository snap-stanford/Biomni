# Task: Gene ID Conversion (Ensembl ID to Gene Symbol)

Convert a list of Ensembl gene IDs to their corresponding Gene Symbols.

## Input

A list of Ensembl gene IDs:
```
ENSG00000000003
ENSG00000000005
ENSG00000000419
ENSG00000000457
ENSG00000000460
```

## Requirements

1. Convert each Ensembl ID to its corresponding Gene Symbol
2. Handle cases where mapping might not be available (NA or missing)
3. Provide a mapping table showing:
   - Ensembl ID
   - Gene Symbol
4. Report any IDs that could not be mapped

## Expected Output Format

Provide a table or list showing the conversion results in markdown format.

## Notes

- Use appropriate annotation database (e.g., org.Hs.eg.db in R, or mygene in Python)
- Ensembl IDs should be provided without version numbers (e.g., ENSG00000000003, not ENSG00000000003.15)
