# Task: Kaplan-Meier Survival Analysis

Perform survival analysis to determine if gene expression level is associated with patient survival.

## Input Data

**Survival data**: `input_data/survival_data.tsv`
- Columns:
  - `sample_id`: Sample identifier
  - `OS_time`: Overall survival time (in months)
  - `OS_status`: Event status (1 = death, 0 = censored)
  - `gene_expression`: Log2-transformed expression value of the gene of interest

## Requirements

1. Load the survival data from `input_data/survival_data.tsv`
2. Divide samples into "High" and "Low" expression groups based on median expression
3. Create Kaplan-Meier survival curves for each group
4. Perform log-rank test to compare survival between groups
5. Report:
   - Number of samples in each group
   - Log-rank test p-value
   - Median survival time for each group (if applicable)
6. Save the survival plot as `survival_plot.png`

## Expected Output Format

Provide:
- Kaplan-Meier survival plot (PNG format)
- Summary statistics and log-rank test results
- Interpretation of whether gene expression is significantly associated with survival
