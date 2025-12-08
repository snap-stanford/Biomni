# Answer: Data Loading and Summary Statistics

## Data Loading Summary

### File Information
- **File format**: TSV (tab-separated values)
- **File path**: `input_data/sample_data.tsv`

### Data Dimensions
- **Number of rows (genes)**: 10
- **Number of columns**: 6 (1 index + 5 samples)

### Column Information
- **Index column**: Gene_ID (e.g., GENE001, GENE002, ...)
- **Data columns**: Sample_A, Sample_B, Sample_C, Sample_D, Sample_E

### Data Type
- **Values**: Floating point numbers (expression values)

## Summary Statistics

### Overall Statistics

|       | Sample_A | Sample_B | Sample_C | Sample_D | Sample_E |
|-------|----------|----------|----------|----------|----------|
| count | 10       | 10       | 10       | 10       | 10       |
| mean  | 9.33     | 9.54     | 9.22     | 9.65     | 9.32     |
| std   | 1.71     | 1.65     | 1.78     | 1.62     | 1.73     |
| min   | 6.50     | 6.80     | 6.20     | 6.90     | 6.40     |
| max   | 12.10    | 11.80    | 12.30    | 11.80    | 12.00    |

### Sample Data (First 5 rows)

| Gene_ID | Sample_A | Sample_B | Sample_C | Sample_D | Sample_E |
|---------|----------|----------|----------|----------|----------|
| GENE001 | 10.5     | 11.2     | 10.8     | 11.5     | 10.9     |
| GENE002 | 9.2      | 9.5      | 9.0      | 9.8      | 9.3      |
| GENE003 | 8.7      | 8.3      | 8.9      | 8.1      | 8.6      |
| GENE004 | 12.1     | 11.8     | 12.3     | 11.5     | 12.0     |
| GENE005 | 7.5      | 7.8      | 7.2      | 7.9      | 7.4      |

## Example Code

### Python (pandas)

```python
import pandas as pd

# Load TSV file
df = pd.read_csv('input_data/sample_data.tsv', sep='\t', index_col=0)

# Basic information
print(f"Dimensions: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

# Summary statistics
print(f"\nSummary statistics:")
print(df.describe())

# First few rows
print(f"\nFirst 5 rows:")
print(df.head())
```

### R

```r
# Load data
df <- read.table('input_data/sample_data.tsv', sep='\t', header=TRUE, row.names=1)

# Basic information
cat("Dimensions:", dim(df), "\n")
cat("Column names:", colnames(df), "\n")

# Summary statistics
summary(df)

# First few rows
head(df)
```

## Notes

- Always check data dimensions and structure before analysis
- Verify data types are appropriate for the analysis
- Check for missing values (NA/NaN)
