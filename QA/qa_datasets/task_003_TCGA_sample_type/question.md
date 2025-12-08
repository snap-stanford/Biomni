# Task: TCGA Sample Type Classification

Given a list of TCGA sample barcodes, classify each sample as either "Tumor" or "Normal" based on the TCGA barcode format.

## TCGA Barcode Format

TCGA barcodes follow this format: `TCGA-XX-XXXX-XX-X`

The fourth part of the barcode (e.g., '01A' in 'TCGA-38-7271-01A') indicates the sample type:
- **Codes 01-09**: Tumor samples
- **Codes 10-19**: Normal samples

## Input

A list of TCGA sample barcodes:
```
TCGA-38-7271-01A
TCGA-55-7914-01A
TCGA-55-6978-11A
TCGA-38-4626-11A
TCGA-95-7043-01A
TCGA-55-6986-11A
```

## Requirements

1. Parse each TCGA barcode to extract the sample type code
2. Classify each sample as "Tumor" or "Normal"
3. Provide a summary showing:
   - Total number of samples
   - Number of tumor samples
   - Number of normal samples
4. Optionally, create a table or list showing the classification result for each sample

## Expected Output Format

Provide a clear summary in markdown format showing:
- Classification results
- Summary statistics
