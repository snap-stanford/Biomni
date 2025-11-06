# sgRNA Design Guide: Three-Tiered Approach

---

## Metadata

**Short Description**: Comprehensive guide for finding or designing sgRNAs using validated sequences, CRISPick datasets, or de novo design tools.

**Authors**: Biomni Team

**Version**: 1.0

**Last Updated**: November 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed (with attribution to original sources)

**Status**: Active

---

## Citations and Acknowledgments

### If you use validated sgRNAs from our database (Option 1):
- **Database Source**: Addgene (https://www.addgene.org)
- **Citation**: Always cite the original publication associated with each sgRNA using the PubMed ID provided in the database
- **Acknowledgment**: "Validated sgRNA sequences obtained from Addgene (https://www.addgene.org/crispr/reference/grna-sequence/)"

### If you use CRISPick designs (Option 2):
- **Acknowledgment Statement**: "Guide designs provided by the CRISPick web tool of the GPP at the Broad Institute"
- **Citation for Cas9 designs (SpCas9, SaCas9)**: Sanson KR, et al. Optimized libraries for CRISPR-Cas9 genetic screens with multiple modalities. Nat Commun. 2018;9(1):5416. PMID: 30575746
- **Citation for Cas12a designs (AsCas12a, enAsCas12a)**: DeWeirdt PC, et al. Optimization of AsCas12a for combinatorial genetic screens in human cells. Nat Biotechnol. 2021;39(1):94-104. PMID: 32661438
  - Note: This paper describes enAsCas12a optimization; specify which variant you used in your methods

---

## Overview

This guide provides a three-tiered approach to sgRNA design, prioritizing validated sequences before moving to computational predictions. Always start with Option 1 and proceed to subsequent options only if needed.

---

## Option 1: Search Validated sgRNA Sequences (Recommended First)

### 1.1 Search for Validated sgRNAs

**IMPORTANT**: You MUST complete BOTH Method 1 AND Method 2 before proceeding to Option 2. Do not skip Method 2 even if Method 1 finds no results.

#### Method 1: Search Our Database (Fastest)

We maintain a curated database of 300+ validated sgRNA sequences from Addgene with experimental evidence.

**Location**: `biomni/know_how/addgene_grna_sequences.csv`

**Search the database**:
```python
import pandas as pd

# Load the database
df = pd.read_csv('addgene_grna_sequences.csv')

# Search for your gene
gene_name = "TP53"
results = df[df['Target_Gene'].str.upper() == gene_name.upper()]

# Filter by species and application
results_filtered = results[
    (results['Target_Species'] == 'H. sapiens') &
    (results['Application'] == 'cut')  # or 'activate', 'RNA targeting'
]

# Display results with references
print(results_filtered[['Target_Gene', 'Target_Sequence',
                        'Plasmid_ID', 'PubMed_ID', 'Depositor']])
```

**Database columns**:
- `Target_Gene`: Gene symbol
- `Target_Species`: Organism (H. sapiens, M. musculus, etc.)
- `Target_Sequence`: 20bp sgRNA sequence (5' to 3')
- `Application`: cut (knockout), activate (CRISPRa), RNA targeting (CRISPRi)
- `Cas9_Species`: S. pyogenes, S. aureus, etc.
- `Plasmid_ID`: Addgene plasmid number
- `Plasmid_URL`: Direct link to plasmid page
- `PubMed_ID`: **Publication reference** (cite this in your work)
- `PubMed_URL`: Direct link to paper
- `Depositor`: Research lab that contributed the sequence

#### Method 2: Advanced Web Search (REQUIRED - Do Not Skip)

**CRITICAL**: Even if Method 1 found no results, you MUST perform this literature search before moving to Option 2. Many validated sgRNAs are published in literature but not in the Addgene database.

Use `advanced_web_search_claude` from `biomni.tool.literature` to find validated sgRNAs from literature and databases:

```python
from biomni.tool.literature import advanced_web_search_claude

# Example usage
results = advanced_web_search_claude("sgRNA TP53 validated H. sapiens experimental")
```

**Search queries to try (use multiple):**
```
"sgRNA" OR "guide RNA" "[GENE_NAME]" validated experimental
"CRISPR knockout" "[GENE_NAME]" sgRNA sequence validated
"[GENE_NAME]" sgRNA "cutting efficiency" OR "on-target"
"[GENE_NAME]" "guide sequence" CRISPR validated
```

**Example for TP53:**
```
"sgRNA" "TP53" validated "H. sapiens" experimental
"CRISPR knockout" "TP53" guide sequence validated
```

**What to search for in results:**
- Published papers with validated sgRNA sequences
- Supplementary materials containing sgRNA sequences
- Other CRISPR databases (e.g., GenScript, Horizon Discovery)
- Laboratory protocols with specific sgRNA sequences

**IMPORTANT**: Spend adequate time searching literature. Look through at least the first 10-15 search results and check supplementary materials of relevant papers.

#### What to Do with Results:

✅ **If you find matching sgRNAs (from either method):**
1. **Record the sgRNA sequence** (20bp target sequence)
2. **Note the reference**:
   - From database: Use `PubMed_ID` to cite the original paper
   - From web search: Record the publication DOI/PubMed ID
3. **Record validation details**: Cell line, cutting efficiency, any reported off-targets


**Example result format:**
```
Gene: TP53
sgRNA sequence: GAGGTTGTGAGGCGCTGCCC
Species: H. sapiens (human)
Application: Knockout (cut)
Reference: PubMed ID 24336569 (Ran et al., 2013)
Validation: Tested in HEK293T cells, 85% cutting efficiency
```

❌ **If no matches found in BOTH Method 1 AND Method 2:**
Only then proceed to **Option 2: Download CRISPick Dataset**

---

## Option 2: Download Pre-Computed sgRNAs from CRISPick

### When to Use This Option?
- No validated sgRNAs found in Addgene
- Need comprehensive coverage of a gene
- Want multiple sgRNA options with predicted scores
- Working with genome-scale screening

### 2.1 Overview of CRISPick

**CRISPick** (from Broad Institute GPP) provides pre-computed sgRNA designs for entire genomes with 238 available datasets covering:
- Human (GRCh38, GRCh37)
- Mouse (GRCm38, GRCm39)
- Dog, Cow, Monkey, and other model organisms
- Multiple Cas enzymes (SpCas9, SaCas9, AsCas12a, enAsCas12a)
  - **IMPORTANT**: AsCas12a and enAsCas12a are DIFFERENT enzymes
  - **AsCas12a**: Wild-type Acidaminococcus sp. Cas12a
  - **enAsCas12a**: Enhanced variant with improved activity
  - **Critical**: sgRNAs designed for enAsCas12a may NOT work with wild-type AsCas12a
  - Always match the dataset to your specific Cas12a variant
- Different applications (knockout, activation, inhibition)

### 2.2 Find the Download Link

**All 238 download links are available in**: `biomni/know_how/download_links.txt`

#### Step 1: Understand File Naming Convention

Files are named: `sgRNA_design_{TAXID}_{GENOME}_{CAS}_{APPLICATION}_{ALGORITHM}_{SOURCE}_{DATE}.txt.gz`

**Common datasets:**

| Organism | Genome | Cas9 | Application | Search Pattern |
|----------|--------|------|-------------|----------------|
| Human | GRCh38 | SpCas9 | Knockout | `9606_GRCh38_SpyoCas9_CRISPRko` |
| Human | GRCh38 | SpCas9 | Activation | `9606_GRCh38_SpyoCas9_CRISPRa` |
| Human | GRCh38 | SaCas9 | Knockout | `9606_GRCh38_SaurCas9_CRISPRko` |
| Mouse | GRCm38 | SpCas9 | Knockout | `10090_GRCm38_SpyoCas9_CRISPRko` |
| Mouse | GRCm38 | SpCas9 | Activation | `10090_GRCm38_SpyoCas9_CRISPRa` |

**Key components:**
- **TAXID**: `9606` (Human), `10090` (Mouse), `9615` (Dog), `9913` (Cow)
- **CAS**:
  - `SpyoCas9` (SpCas9, NGG PAM)
  - `SaurCas9` (SaCas9, NNGRRT PAM)
  - `AsCas12a` (Wild-type Cas12a, TTTV PAM)
  - `enAsCas12a` (Enhanced Cas12a, TTTV PAM)
  - **WARNING**: Do NOT use AsCas12a datasets if you have enAsCas12a, and vice versa
- **APPLICATION**: `CRISPRko` (knockout), `CRISPRa` (activation), `CRISPRi` (inhibition)

#### Step 2: Find Your Download URL

```bash
# Search the download links file
grep "9606_GRCh38_SpyoCas9_CRISPRko" biomni/know_how/download_links.txt

# Or for mouse
grep "10090_GRCm38_SpyoCas9_CRISPRko" biomni/know_how/download_links.txt
```

#### Step 3: Download and Extract

```bash
# Copy the URL from download_links.txt, then:
wget [PASTE_URL_HERE]

# Extract the file
gunzip sgRNA_design_*.txt.gz
```

**File sizes**: Knockout (300-700 MB), Activation (50-100 MB), Summary files (1-3 MB)

### 2.4 Understanding the File Format

The `.txt` file is tab-delimited. Column names differ between knockout and activation/inhibition datasets.

**Essential Columns (All files):**
- **sgRNA Sequence**: The 20bp guide RNA sequence (5' to 3')
- **Target Gene Symbol**: Gene name (e.g., "TP53", "BRCA1")
- **Combined Rank**: Overall ranking (lower = better) - **Use this by default**
- **On-Target Rank**: Ranking by efficiency only (lower = better)
- **Off-Target Rank**: Ranking by specificity only (lower = better)
- **PAM Sequence**: The PAM sequence (e.g., "AGG", "TGG")

**Knockout-specific columns:**
- **sgRNA Cut Position (1-based)**: Genomic coordinate of cut site
- **Exon Number**: Which exon is targeted
- **Target Cut %**: Percentage of protein affected

**Activation/Inhibition-specific columns:**
- **sgRNA 'Cut' Position**: Position relative to gene
- **sgRNA 'Cut' Site TSS Offset**: Distance from transcription start site (bp)
- **DHS Score**: DNase hypersensitivity score (for CRISPRa)

### 2.5 Extract and Select sgRNAs

#### Step 1: Load Data and Filter for Your Gene

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_*.txt',
                 sep='\t', low_memory=False)

# Filter for your gene
gene_name = "TP53"
gene_sgrnas = df[df['Target Gene Symbol'] == gene_name].copy()

print(f"Found {len(gene_sgrnas)} sgRNAs for {gene_name}")
```

#### Step 2: Select Top sgRNAs

**Default: Use Combined Rank (balances efficiency and specificity)**
```python
# Sort by Combined Rank (lower is better)
top_sgrnas = gene_sgrnas.nsmallest(10, 'Combined Rank')

print(top_sgrnas[['sgRNA Sequence', 'Combined Rank',
                   'Exon Number', 'sgRNA Cut Position (1-based)']])
```

**Option A: Prioritize On-Target Efficiency**
```python
# Sort by On-Target Rank (for maximum cutting efficiency)
efficient_sgrnas = gene_sgrnas.nsmallest(10, 'On-Target Rank')
```

**Option B: Prioritize Off-Target Specificity**
```python
# Sort by Off-Target Rank (for maximum specificity)
specific_sgrnas = gene_sgrnas.nsmallest(10, 'Off-Target Rank')
```

#### Step 3: Filter by Custom Criteria (Optional)

**Filter by Exon Number:**
```python
# Target specific exon (e.g., exon 5)
exon5_sgrnas = gene_sgrnas[gene_sgrnas['Exon Number'] == 5]
top_exon5 = exon5_sgrnas.nsmallest(5, 'Combined Rank')
```

**Filter by Genomic Position:**
```python
# Target specific genomic range
position_filtered = gene_sgrnas[
    (gene_sgrnas['sgRNA Cut Position (1-based)'] >= 7572000) &
    (gene_sgrnas['sgRNA Cut Position (1-based)'] <= 7575000)
]
```

**Target Early Exons for Knockout:**
```python
# Get sgRNAs from first 3 exons
early_exons = gene_sgrnas[gene_sgrnas['Exon Number'] <= 3]
top_early = early_exons.nsmallest(10, 'Combined Rank')
```

**Filter by Target Cut Percentage:**
```python
# Target sgRNAs that affect significant portion of protein
high_impact = gene_sgrnas[gene_sgrnas['Target Cut %'] <= 50]  # Cut in first 50%
top_high_impact = high_impact.nsmallest(10, 'Combined Rank')
```

#### Step 4: Select Multiple sgRNAs for Validation

```python
# Get top 4 sgRNAs from different exons for redundancy
final_selection = gene_sgrnas.sort_values('Combined Rank').groupby('Exon Number').head(1).head(4)

# Save results
final_selection.to_csv(f'{gene_name}_selected_sgRNAs.csv', index=False)

print("\nSelected sgRNAs:")
print(final_selection[['sgRNA Sequence', 'Exon Number', 'Combined Rank']])
```

### 2.6 What to Do with Results

✅ **Once you have selected sgRNAs:**
1. Choose **3-4 sgRNAs** (use Combined_Rank by default)

❌ **If dataset doesn't cover your gene or organism:**
Proceed to **Option 3: De Novo sgRNA Design**

---

## Option 3: General sgRNA Design Guidelines (Last Resort)

### When to Use This Option?
- Gene not covered in CRISPick datasets
- Non-model organism
- Custom design requirements

### General Design Rules

#### Essential Requirements:
1. **Length**:
   - 20 bp for SpCas9 and SaCas9
   - 23-25 bp for Cas12a variants (AsCas12a, enAsCas12a)
2. **PAM sequence**:
   - **SpCas9**: Requires **NGG** immediately after target (3' end)
   - **SaCas9**: Requires **NNGRRT** immediately after target (3' end)
   - **AsCas12a** (wild-type): Requires **TTTV** before target (5' end)
   - **enAsCas12a** (enhanced): Requires **TTTV** before target (5' end)
   - **CRITICAL**: Guides designed for enAsCas12a may not work with wild-type AsCas12a due to different activity profiles
3. **GC content**: 40-60% is optimal
4. **Avoid**:
   - TTTT sequences (terminates transcription)
   - Long runs of same nucleotide (>4 repeats)

#### Target Location:
- **For knockout**: Target early exons (first 50% of gene)
- **For activation**: Target -200 to +1 bp from transcription start site (TSS)
- **For inhibition**: Target -50 to +300 bp from TSS

#### Best Practices:
- **Test 3-4 different sgRNAs** per target gene
- Select sgRNAs with high efficiency scores (>0.5) and minimal off-targets
- Validate experimentally with Sanger sequencing

---

## Quick Start Examples

### Example 1: Knockout TP53 in Human Cells

**Step 1**: Check Addgene
```python
df = pd.read_csv('addgene_grna_sequences.csv')
tp53_results = df[(df['Target_Gene'] == 'TP53') &
                  (df['Target_Species'] == 'H. sapiens') &
                  (df['Application'] == 'cut')]
# Result: Found 0 entries → Proceed to Option 2
```

**Step 2**: Download CRISPick dataset
```bash
# Download human GRCh38 SpCas9 knockout dataset
wget https://portals.broadinstitute.org/gppx/public/sgrna_design/api/downloads/\
sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_RS3seq-Chen2013+RS3target_NCBI_20241104.txt.gz

gunzip sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_*.txt.gz
```

**Step 3**: Extract TP53 sgRNAs
```python
df = pd.read_csv('sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_*.txt', sep='\t')
tp53 = df[df['Gene_Symbol'] == 'TP53']
top_sgrnas = tp53[
    (tp53['sgRNA_score'] > 0.6) &
    (tp53['Off_target_stringency'] > 0.5)
].sort_values('sgRNA_score', ascending=False).head(4)

print(top_sgrnas[['sgRNA_sequence', 'sgRNA_score', 'Exon_ID']])
```

### Example 2: Activate OCT4 in Human iPSCs

**Step 1**: Check Addgene
```python
oct4_results = df[(df['Target_Gene'] == 'OCT4') &
                  (df['Application'] == 'activate')]
# Found 1 validated sgRNA!
print(oct4_results['Target_Sequence'].values[0])
# Use this sequence ✅
```

---

**Remember**: Always start with validated sequences (Option 1), then move to pre-computed designs (Option 2), and only use de novo design (Option 3) when necessary. Testing 3-4 sgRNAs per gene is standard practice regardless of prediction scores.
