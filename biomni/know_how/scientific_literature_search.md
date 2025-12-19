# Scientific Literature Search Guide

---

## Metadata

**Short Description**: Best practices for searching, retrieving, and analyzing scientific literature using database-specific queries, structured search strategies, and AI-assisted tools.

**Authors**: Biomni Team, adapted from Consensus AI, Elicit, PubMed, and Semantic Scholar best practices

**Version**: 1.0

**Last Updated**: December 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

**Source References**:
- PubMed Search Guide (NCBI)
- Semantic Scholar API Documentation
- Consensus AI Search Guidelines
- Elicit Research Assistant Best Practices

---

## Overview

This guide provides a systematic approach to scientific literature search, combining database-specific query strategies with AI-assisted analysis. The workflow is designed to maximize recall (finding all relevant papers) while maintaining precision (avoiding irrelevant results).

## Three-Tiered Search Strategy

### Tier 1: Database-Specific Searches (Most Reliable)

Start with established academic databases for peer-reviewed content.

**Recommended for**: Finding specific papers, systematic reviews, clinical evidence

### Tier 2: AI-Assisted Web Search (Comprehensive)

Use AI tools for broader context and synthesis.

**Recommended for**: Understanding research landscape, finding recent developments, complex multi-faceted questions

### Tier 3: Direct Content Extraction (Deep Dive)

Extract and analyze full-text content from papers and supplementary materials.

**Recommended for**: Detailed methodology extraction, data retrieval, protocol identification

## Tier 1: Database-Specific Searches

### 1.1 PubMed Search (Biomedical Literature)

PubMed is the primary database for biomedical and life science literature. Use `query_pubmed` from `biomni.tool.literature`.

```python
from biomni.tool.literature import query_pubmed

# Basic search
results = query_pubmed("CRISPR gene editing", max_papers=10)

# Advanced search with MeSH terms
results = query_pubmed(
    '"CRISPR-Cas Systems"[MeSH] AND "Gene Editing"[MeSH]',
    max_papers=20
)
```

#### PubMed Query Syntax (PICO Framework)

For clinical questions, structure your query using **PICO**:
- **P**opulation: Who are you studying?
- **I**ntervention: What treatment/exposure?
- **C**omparison: Alternative treatment?
- **O**utcome: What result?

**Example**: Does metformin reduce cardiovascular events in diabetic patients?
```
"Diabetes Mellitus"[MeSH] AND "Metformin"[MeSH] AND "Cardiovascular Diseases"[MeSH] AND ("clinical trial"[Publication Type] OR "meta-analysis"[Publication Type])
```

#### PubMed Field Tags

| Tag | Description | Example |
|-----|-------------|---------|
| `[MeSH]` | Medical Subject Heading | "Neoplasms"[MeSH] |
| `[Title]` | Title only | "CRISPR"[Title] |
| `[Title/Abstract]` | Title or Abstract | "gene therapy"[Title/Abstract] |
| `[Author]` | Author name | "Zhang F"[Author] |
| `[Journal]` | Journal name | "Nature"[Journal] |
| `[Publication Type]` | Article type | "Review"[Publication Type] |
| `[Date - Publication]` | Date range | "2020/01/01"[Date - Publication]:"2024/12/31"[Date - Publication] |

#### Boolean Operators

```python
# AND: All terms must be present
results = query_pubmed("CRISPR AND cancer AND therapy")

# OR: Any term can be present (use for synonyms)
results = query_pubmed("(tumor OR tumour OR neoplasm) AND immunotherapy")

# NOT: Exclude terms (use sparingly)
results = query_pubmed("cancer immunotherapy NOT review")
```

### 1.2 arXiv Search (Physics, Math, CS, Biology Preprints)

Use `query_arxiv` for preprints in quantitative fields, including computational biology.

```python
from biomni.tool.literature import query_arxiv

# Basic search
results = query_arxiv("protein structure prediction", max_papers=10)

# Author-specific search
results = query_arxiv("au:Jumper AND AlphaFold", max_papers=5)

# Subject category search
results = query_arxiv("cat:q-bio.BM AND machine learning", max_papers=10)
```

#### arXiv Subject Categories (Biology-related)

| Category | Description |
|----------|-------------|
| `q-bio.BM` | Biomolecules |
| `q-bio.CB` | Cell Behavior |
| `q-bio.GN` | Genomics |
| `q-bio.MN` | Molecular Networks |
| `q-bio.NC` | Neurons and Cognition |
| `q-bio.QM` | Quantitative Methods |
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |

### 1.3 Google Scholar Search (Broad Academic Coverage)

Use `query_scholar` for broader academic searches across disciplines.

```python
from biomni.tool.literature import query_scholar

# Returns first most relevant result
result = query_scholar("single cell RNA sequencing analysis methods")
```

**Note**: Google Scholar has rate limits. Use sparingly and consider delays between requests.

## Tier 2: AI-Assisted Web Search

### 2.1 Advanced Web Search with Claude

Use `advanced_web_search_claude` for comprehensive, AI-synthesized research queries.

```python
from biomni.tool.literature import advanced_web_search_claude

# Complex research question
results = advanced_web_search_claude(
    "What are the latest developments in CAR-T cell therapy for solid tumors in 2024?",
    max_searches=3
)

# Comparative analysis
results = advanced_web_search_claude(
    "Compare different CRISPR delivery methods for in vivo gene editing: viral vectors vs lipid nanoparticles",
    max_searches=5
)
```

#### When to Use AI-Assisted Search

✅ **Good use cases**:
- Understanding research landscape and current state of the field
- Finding recent developments not yet in academic databases
- Complex questions requiring synthesis of multiple sources
- Identifying key researchers, labs, and institutions
- Finding methodology comparisons and reviews

❌ **Avoid for**:
- Specific paper lookups (use database searches)
- Citation counts and impact factors (use Google Scholar)
- Systematic reviews requiring reproducibility
- When exact search terms must be documented

### 2.2 General Web Search

Use `search_google` (via DuckDuckGo) for protocols, tutorials, and general information.

```python
from biomni.tool.literature import search_google

# Find protocols
results = search_google("Western blot protocol step by step", num_results=5)

# Find software documentation
results = search_google("Seurat single cell analysis tutorial", num_results=3)
```

## Tier 3: Content Extraction

### 3.1 Extract URL Content

Use `extract_url_content` to get clean text from web pages.

```python
from biomni.tool.literature import extract_url_content

# Extract article content
content = extract_url_content("https://www.nature.com/articles/nature12373")
```

### 3.2 Extract PDF Content

Use `extract_pdf_content` to extract text from PDF files.

```python
from biomni.tool.literature import extract_pdf_content

# Direct PDF URL
content = extract_pdf_content("https://arxiv.org/pdf/1706.03762.pdf")

# Page with PDF link (will find and download PDF)
content = extract_pdf_content("https://www.nature.com/articles/nature12373")
```

### 3.3 Fetch Supplementary Materials

Use `fetch_supplementary_info_from_doi` to download supplementary data.

```python
from biomni.tool.literature import fetch_supplementary_info_from_doi

# Download supplementary files using DOI
log = fetch_supplementary_info_from_doi(
    "10.1038/nature12373",
    output_dir="./supplementary_materials"
)
```

## Recommended Search Workflow

### Step 1: Define Your Research Question

Before searching, clearly define:
1. **Main concept**: What is the primary topic?
2. **Population/Model**: Humans, mice, cell lines, etc.?
3. **Intervention/Method**: What technique or treatment?
4. **Outcome**: What results are you looking for?
5. **Time frame**: Recent only or historical?

**Example**: "Find recent papers on CRISPR base editing efficiency in human iPSCs"
- Main concept: CRISPR base editing
- Model: Human iPSCs
- Outcome: Efficiency data
- Time frame: Last 3 years

### Step 2: Construct Search Queries

#### Start Broad, Then Narrow

```python
# Step 1: Broad search to understand the field
results = query_pubmed("CRISPR base editing iPSC", max_papers=20)

# Step 2: Add specific terms based on initial results
results = query_pubmed(
    '"CRISPR-Cas Systems"[MeSH] AND "base editing" AND "induced pluripotent stem cells" AND efficiency',
    max_papers=20
)

# Step 3: Filter by date and article type
results = query_pubmed(
    '"CRISPR-Cas Systems"[MeSH] AND "base editing" AND "induced pluripotent stem cells" AND efficiency AND ("2022"[Date - Publication]:"2024"[Date - Publication])',
    max_papers=20
)
```

### Step 3: Evaluate and Filter Results

When reviewing results, prioritize:
1. **Relevance**: Does the paper address your question?
2. **Recency**: Is it recent enough for your needs?
3. **Quality indicators**: 
   - Published in peer-reviewed journals
   - High citation count
   - Reputable authors/institutions
4. **Study type**: 
   - For efficacy: RCTs, systematic reviews, meta-analyses
   - For mechanisms: Basic research papers
   - For methods: Protocol papers, method comparisons

### Step 4: Deep Dive into Key Papers

For important papers:
```python
# 1. Extract full text
content = extract_pdf_content("paper_url.pdf")

# 2. Get supplementary materials
log = fetch_supplementary_info_from_doi("10.xxxx/xxxxx", "./supplements")

# 3. Check references for additional papers
# 4. Check citations to find follow-up work
```

## Query Formulation Best Practices

### Do's:

1. **Use synonyms and alternative terms**
   ```python
   # Bad: Too narrow
   query_pubmed("heart attack treatment")
   
   # Good: Includes medical terms
   query_pubmed("(myocardial infarction OR heart attack) AND (treatment OR therapy)")
   ```

2. **Use controlled vocabulary (MeSH terms for PubMed)**
   ```python
   # Better recall with MeSH
   query_pubmed('"Myocardial Infarction"[MeSH] AND "Drug Therapy"[MeSH]')
   ```

3. **Specify study types when needed**
   ```python
   # Find clinical trials only
   query_pubmed("COVID-19 vaccine efficacy AND clinical trial[Publication Type]")
   ```

4. **Use phrase searching for exact matches**
   ```python
   # Use quotes for exact phrases
   query_pubmed('"single cell RNA sequencing" AND methods')
   ```

5. **Iterate based on results**
   - Too many results? Add more specific terms
   - Too few results? Broaden terms, add synonyms
   - Wrong focus? Adjust key concepts

### Don'ts:

1. **Don't use overly long queries**
   ```python
   # Bad: Too specific, may miss relevant papers
   query_pubmed("CRISPR Cas9 gene editing HEK293T cells 2024 efficiency optimization delivery")
   
   # Good: Core concepts only
   query_pubmed("CRISPR Cas9 gene editing optimization efficiency")
   ```

2. **Don't rely on a single database**
   - PubMed: Biomedical focus
   - arXiv: Preprints, computational methods
   - Google Scholar: Broader coverage

3. **Don't ignore publication dates**
   - Science moves fast; check for recent updates
   - But don't exclude foundational older papers

4. **Don't skip title/abstract review**
   - Not all "relevant" search results are truly relevant
   - Read abstracts before full papers

## Common Search Scenarios

### Scenario 1: Finding Methods/Protocols

```python
# Search for methodology papers
results = query_pubmed(
    '"Western Blotting"[MeSH] AND (protocol OR method OR technique)',
    max_papers=10
)

# Check web for step-by-step protocols
results = search_google("Western blot protocol for membrane proteins", num_results=5)
```

### Scenario 2: Understanding Disease Mechanism

```python
# Find review articles first
results = query_pubmed(
    '"Alzheimer Disease"[MeSH] AND pathophysiology AND review[Publication Type]',
    max_papers=10
)

# Then find specific mechanistic studies
results = query_pubmed(
    '"Alzheimer Disease"[MeSH] AND ("amyloid beta"[MeSH] OR tau) AND mechanism',
    max_papers=20
)
```

### Scenario 3: Finding Drug/Treatment Information

```python
# Clinical trials
results = query_pubmed(
    '"Drug Name"[Substance Name] AND "Condition"[MeSH] AND clinical trial[Publication Type]',
    max_papers=20
)

# Systematic reviews
results = query_pubmed(
    '"Drug Name" AND "Condition" AND (systematic review[Publication Type] OR meta-analysis[Publication Type])',
    max_papers=10
)
```

### Scenario 4: Latest Developments in a Field

```python
# Use AI-assisted search for synthesis
results = advanced_web_search_claude(
    "What are the most significant advances in CAR-T cell therapy in 2024?",
    max_searches=3
)

# Supplement with recent PubMed searches
results = query_pubmed(
    '"Chimeric Antigen Receptor T-Cell Therapy"[MeSH] AND "2024"[Date - Publication]',
    max_papers=20
)
```

### Scenario 5: Finding Specific Reagents/Materials

```python
# Search for validated reagents
results = advanced_web_search_claude(
    "validated antibodies for Western blot detection of p53 protein",
    max_searches=2
)

# Search databases
results = search_google("p53 antibody Western blot validated", num_results=5)
```

## Quality Assessment Checklist

When evaluating search results:

- [ ] **Source reliability**: Is it from a peer-reviewed journal?
- [ ] **Author credentials**: Are authors experts in the field?
- [ ] **Recency**: Is the information up-to-date?
- [ ] **Study design**: Appropriate for the question?
- [ ] **Sample size**: Adequate for conclusions drawn?
- [ ] **Reproducibility**: Methods described clearly?
- [ ] **Conflicts of interest**: Any declared conflicts?
- [ ] **Citation count**: Well-cited by others?

## Troubleshooting

### Issue: No Results Found

**Solutions**:
1. Broaden search terms
2. Remove restrictive filters (date, publication type)
3. Try alternative databases
4. Use synonyms and alternative terminology
5. Check spelling of scientific terms

### Issue: Too Many Results

**Solutions**:
1. Add more specific terms
2. Use MeSH terms instead of free text
3. Restrict to specific publication types
4. Limit date range
5. Add [Title] or [Title/Abstract] field tags

### Issue: Results Not Relevant

**Solutions**:
1. Use phrase searching ("exact phrase")
2. Use NOT to exclude irrelevant topics
3. Focus on MeSH major topic ([MeSH Major Topic])
4. Review and refine key concepts

### Issue: API Rate Limiting

**Solutions**:
1. Add delays between requests (`time.sleep(2)`)
2. Use different databases
3. Cache results for repeated searches
4. Reduce number of results requested

## Resources

### Databases:
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/
- **arXiv**: https://arxiv.org/
- **Google Scholar**: https://scholar.google.com/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **bioRxiv**: https://www.biorxiv.org/

### Reference Management:
- **Zotero**: Free, open-source
- **Mendeley**: Free with premium features
- **EndNote**: Commercial, institutional licenses

### Search Guides:
- **PubMed User Guide**: https://pubmed.ncbi.nlm.nih.gov/help/
- **MeSH Browser**: https://meshb.nlm.nih.gov/
- **arXiv Help**: https://arxiv.org/help/

