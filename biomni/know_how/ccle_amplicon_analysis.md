# CCLE Amplicon Repository Data Analysis

---

## Metadata

**Short Description**: Guide to analyzing ecDNA and amplicon data from the Cancer Cell Line Encyclopedia (CCLE) using Python and pandas.

**Authors**: Amplicon Repository Team

**Version**: 1.0

**Last Updated**: February 2026

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

**Data Source**: CCLE amplicon analysis pipeline results (AmpliconArchitect + AmpliconClassifier)

---

## Overview

This guide covers amplicon and ecDNA data from the CCLE: **1,234** amplicon features across ~1,157 cancer cell lines, with classifications (ecDNA, Linear, BFB, Complex-non-cyclic), gene annotations, copy numbers, and tissue metadata. The dataset is loaded from **CCLE.csv** in the agent data path (same as the `query_amplicons` tool): data path is the agent path (default `./data`), or `BIOMNI_PATH` / `BIOMNI_DATA_PATH` if set.

```python
import pandas as pd
import os

# Same path resolution as query_amplicons in amplicon_table.py
data_path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH") or "./data"
csv_path = os.path.join(data_path, "CCLE.csv")
df = pd.read_csv(csv_path)  # Shape: (1234, 30)
```

Below is a walkthrough of all **30 columns** in the dataset.

---

## Column Reference (all 30 columns)

### Identification Columns

**`Unnamed: 0`** (int)  
Sequential index from original data. Can be ignored; pandas will create its own index.

**`Sample name`** (str)  
Unique identifier for cell line sample. Format: `{CELL_LINE}_{TISSUE}` (e.g. `"22RV1_PROSTATE"`). Primary key for grouping analyses by cell line.

**`AA amplicon number`** (float/NaN)  
AmpliconArchitect's amplicon number for this feature. NaN for samples with no detected amplicons. Used to identify multiple amplicons within the same sample.

**`Feature ID`** (str)  
Unique identifier combining sample and amplicon. Format: `{SAMPLE_NAME}_{AMPLICON_NUM}` or `{SAMPLE_NAME}_NA`. Use as the primary key for individual amplicon features.

### Classification Column

**`Classification`** (str)  
Amplicon structural classification from AmpliconClassifier. **Values**: `"ecDNA"` (297), `"Linear"` (550), `"BFB"` (114), `"Complex-non-cyclic"` (196), `NaN` (77 = no amplicon). Most important column for categorizing amplicon types.

### Genomic Location and Genes

**`Location`** (str)  
Genomic coordinates of the amplicon. **Format**: string representation of a list of strings, e.g. `["'chr6:20350615-22839372'"]`. Parse with `ast.literal_eval()` to get a Python list.

**`Oncogenes`** (str)  
Known oncogenes in the amplicon. **Format**: string representation of a list of gene names, e.g. `["'E2F3'", "'SOX4'"]`. Empty when none: `["''"]`. Filter out `"''"` after parsing.

**`All genes`** (str)  
All genes (oncogenes + other) in the amplicon. Same format and parsing as Oncogenes.

**`NCBI Gene IDs`** (str)  
NCBI Gene IDs corresponding to genes. String representation of a list; often `[]` or list of numeric IDs. Less commonly used than gene symbols.

### Amplicon Features

**`Complexity score`** (float)  
Quantitative measure of amplicon structural complexity. Range typically 1.0 to ~10+ (higher = more complex). NaN for samples without amplicons.

**`ecDNA context`** (str)  
Additional context about ecDNA structure. Often NaN or detailed structural info; less used in basic analyses.

**`Captured interval length`** (float)  
Total length of captured genomic intervals in the amplicon (base pairs). NaN when no amplicon.

**`Feature median copy number`** (float)  
Median copy number across the amplicon. Key metric for amplification strength. NaN when no amplicon.

**`Feature maximum copy number`** (float)  
Maximum copy number within the amplicon; usually higher than median. Useful for peak amplification regions.

**`Filter flag`** (str/NaN)  
Quality control flag. NaN = passed all filters (most common); non-NaN may indicate low-confidence feature.

### Reference and Versions

**`Reference version`** (str)  
Genome reference build; `"GRCh38"` for all samples.

**`AS-p version`**, **`AA version`**, **`AC version`** (str)  
Software versions: AmpliconSuite-pipeline, AmpliconArchitect, AmpliconClassifier. Useful for reproducibility.

### Sample Metadata

**`Tissue of origin`** (str)  
Tissue type of the cancer cell line (e.g. `"prostate"`, `"lung"`, `"breast"`). Lowercase, standardized. Important for tissue-specific analyses.

**`Sample type`** (str)  
Type of sample; `"cell line"` for all samples in this dataset.

### File Paths

**`Feature BED file`**, **`CNV BED file`** (str)  
Paths to BED files with genomic coordinates. Often `"Not Provided"` when no amplicon.

**`AA PNG file`**, **`AA PDF file`** (str)  
Paths to AmpliconArchitect visualization files. Often `"Not Provided"` when no amplicon.

**`AA summary file`** (str)  
Path to AmpliconArchitect summary text file.

**`Run metadata JSON`**, **`Sample metadata JSON`** (str)  
Paths to JSON files with pipeline and sample metadata.

**`AA directory`**, **`cnvkit directory`** (str)  
Paths to `.tar.gz` archives with full pipeline outputs.

---

## Critical: Parsing gene lists and locations

Gene and location columns are **string representations of Python lists**. Parse and clean before use:

```python
import ast

def parse_genes(gene_str):
    if pd.isna(gene_str):
        return []
    try:
        genes = ast.literal_eval(gene_str)
        return [g.strip("'") for g in genes if g != "''"]
    except (ValueError, SyntaxError):
        return []

df['oncogenes_list'] = df['Oncogenes'].apply(parse_genes)
```

- Use `ast.literal_eval()` only (never `eval()`).
- Filter out `"''"` — empty lists are stored as `["''"]`.
- For amplicon-level stats, use `df[df['Classification'].notna()]`.

---

## Example: Count amplicons by classification

```python
classification_counts = df['Classification'].value_counts(dropna=False)
classification_pct = df['Classification'].value_counts(normalize=True) * 100
# e.g. ecDNA percentage: classification_pct.get('ecDNA', 0)
```

---

## Example: Find samples with a specific gene

```python
def has_gene(gene_list_str, target_gene):
    if pd.isna(gene_list_str):
        return False
    try:
        genes = ast.literal_eval(gene_list_str)
        genes_clean = [g.strip("'") for g in genes if g != "''"]
        return target_gene in genes_clean
    except (ValueError, SyntaxError):
        return False

target_gene = 'MYC'
myc_samples = df[df['Oncogenes'].apply(lambda x: has_gene(x, target_gene))]
# myc_samples[['Sample name', 'Classification', 'Oncogenes', 'Feature median copy number', 'Tissue of origin']]
```

---

## Example: Tissue distribution and ecDNA rate by tissue

```python
tissue_counts = df['Tissue of origin'].value_counts()
ecdna_df = df[df['Classification'] == 'ecDNA']
ecdna_by_tissue = ecdna_df['Tissue of origin'].value_counts()

# ecDNA rate by tissue (tissues with 10+ total amplicons)
tissue_stats = df.groupby('Tissue of origin').agg({
    'Classification': lambda x: (x == 'ecDNA').sum(),
    'Sample name': 'count'
}).rename(columns={'Classification': 'ecDNA_count', 'Sample name': 'total_count'})
tissue_stats = tissue_stats[tissue_stats['total_count'] >= 10]
tissue_stats['ecDNA_rate'] = tissue_stats['ecDNA_count'] / tissue_stats['total_count']
tissue_stats_sorted = tissue_stats.sort_values('ecDNA_rate', ascending=False)
```

---

## Example: Copy number analysis

```python
amplicons = df[df['Classification'].notna()]
print(amplicons['Feature median copy number'].describe())

high_amp = amplicons[amplicons['Feature median copy number'] > 50]

cn_by_class = amplicons.groupby('Classification')['Feature median copy number'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
])
```

---

## Example: Complexity score and most complex amplicons

```python
amplicons = df[df['Classification'].notna()]
complexity_by_class = amplicons.groupby('Classification')['Complexity score'].agg([
    'count', 'mean', 'median', 'min', 'max'
])

most_complex = amplicons.nlargest(10, 'Complexity score')[
    ['Sample name', 'Classification', 'Complexity score', 'Oncogenes', 'Tissue of origin']
]
```

---

## Example: Parse genomic locations and filter by chromosome

```python
import re

def parse_location(location_str):
    if pd.isna(location_str):
        return None
    try:
        locations = ast.literal_eval(location_str)
        parsed = []
        for loc in locations:
            loc_clean = loc.strip("'")
            if loc_clean and ':' in loc_clean:
                match = re.match(r'(chr[\dXY]+):(\d+)-(\d+)', loc_clean)
                if match:
                    chrom, start, end = match.groups()
                    parsed.append({
                        'chromosome': chrom,
                        'start': int(start),
                        'end': int(end),
                        'length': int(end) - int(start)
                    })
        return parsed
    except (ValueError, SyntaxError):
        return None

df['parsed_locations'] = df['Location'].apply(parse_location)
chr8_amplicons = df[df['parsed_locations'].apply(
    lambda x: x and any(loc['chromosome'] == 'chr8' for loc in x)
)]

def total_span(parsed_locs):
    return sum(loc['length'] for loc in parsed_locs) if parsed_locs else 0
df['total_genomic_span'] = df['parsed_locations'].apply(total_span)
```

---

## Example: Multi-gene and gene-rich amplicons

```python
def count_genes(gene_list_str):
    if pd.isna(gene_list_str):
        return 0
    try:
        genes = ast.literal_eval(gene_list_str)
        return len([g for g in genes if g != "''"])
    except (ValueError, SyntaxError):
        return 0

df['num_all_genes'] = df['All genes'].apply(count_genes)
df['num_oncogenes'] = df['Oncogenes'].apply(count_genes)

amplicons = df[df['Classification'].notna()]
multi_onc = amplicons[amplicons['num_oncogenes'] >= 2]
gene_rich = amplicons.nlargest(10, 'num_all_genes')[
    ['Sample name', 'Classification', 'num_all_genes', 'All genes']
]
```

---

## Example: End-to-end — MYC-amplified ecDNA by tissue

```python
df['oncogenes_list'] = df['Oncogenes'].apply(parse_genes)
ecdna = df[df['Classification'] == 'ecDNA'].copy()
ecdna['has_MYC'] = ecdna['oncogenes_list'].apply(lambda x: 'MYC' in x)
myc_ecdna = ecdna[ecdna['has_MYC']]

tissue_stats = ecdna.groupby('Tissue of origin').agg(
    total_ecdna=('Sample name', 'count'),
    myc_ecdna=('has_MYC', 'sum')
)
tissue_stats['myc_rate'] = tissue_stats['myc_ecdna'] / tissue_stats['total_ecdna']
tissue_stats_filtered = tissue_stats[tissue_stats['total_ecdna'] >= 5]
tissue_stats_sorted = tissue_stats_filtered.sort_values('myc_rate', ascending=False)
# tissue_stats_sorted.head(10); myc_ecdna[['Sample name', 'Tissue of origin', 'Oncogenes', ...]].head()
```

---

## Pitfalls

- **String-encoded lists**: Parse with `ast.literal_eval()` before searching (e.g. for `'MYC'`).
- **Empty genes**: `["''"]` has length 1; filter with `[g for g in lst if g != "''"]`.
- **77 NaN classifications**: Use `value_counts(dropna=False)` or `df['Classification'].notna()` as needed.
- **Tissue names**: Stored lowercase; match with `== 'lung'` or `.str.lower()`.
- **Copy number NaN**: Filter to amplicons first or use `.notna()` before comparisons.

---

## Quick reference

```python
df = pd.read_csv(os.environ['CCLE_AMPLICON_CSV'])
df['oncogenes_list'] = df['Oncogenes'].apply(parse_genes)

ecdna = df[df['Classification'] == 'ecDNA']
amplicons_only = df[df['Classification'].notna()]
```

If `CCLE_AMPLICON_CSV` is not set, set it to the path of your `aggregated_results.csv`.
