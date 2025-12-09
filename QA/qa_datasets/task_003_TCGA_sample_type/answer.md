# Answer: TCGA Sample Type Classification Results

## Classification Results

| Sample Barcode | Sample Type Code | Classification |
|----------------|------------------|----------------|
| TCGA-38-7271-01A | 01 | Tumor |
| TCGA-55-7914-01A | 01 | Tumor |
| TCGA-55-6978-11A | 11 | Normal |
| TCGA-38-4626-11A | 11 | Normal |
| TCGA-95-7043-01A | 01 | Tumor |
| TCGA-55-6986-11A | 11 | Normal |

## Summary Statistics

- **Total samples**: 6
- **Tumor samples**: 3 (50.0%)
- **Normal samples**: 3 (50.0%)

## Explanation

The TCGA barcode format is: `TCGA-XX-XXXX-XX-X`

The fourth part (e.g., '01A' or '11A') contains the sample type code:
- Extract the first two digits of the fourth part
- If the code is between 01-09: **Tumor sample**
- If the code is between 10-19: **Normal sample**

Codes 01-09 represent different types of tumor samples (primary tumor, recurrent tumor, etc.), and codes 10-19 represent different types of normal samples (blood normal, solid tissue normal, etc.).
