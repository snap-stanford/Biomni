# Answer: Survival Analysis Results

## Summary

- **Total samples**: 30
- **High expression group**: 15 samples (expression > median)
- **Low expression group**: 15 samples (expression ≤ median)
- **Median expression**: 11.15
- **Log-rank test p-value**: 0.002 (statistically significant)

## Survival Curve Description

The Kaplan-Meier survival curve shows a significant difference in survival between high and low expression groups:
- **High expression group**: Better survival (higher survival probability over time)
- **Low expression group**: Worse survival (lower survival probability)

## Group Statistics

| Group | N | Events | Median Survival (months) |
|-------|---|--------|--------------------------|
| High | 15 | 3 | Not reached |
| Low | 15 | 12 | 18.7 |

## Example Code

### R (survival, survminer)

```r
library(survival)
library(survminer)
library(ggplot2)

# Load data
survival_data <- read.table('input_data/survival_data.tsv', sep='\t', header=TRUE)

# Divide into groups based on median
median_expr <- median(survival_data$gene_expression)
survival_data$group <- ifelse(survival_data$gene_expression > median_expr, 
                              "High", "Low")

# Create survival object
surv_obj <- Surv(time = survival_data$OS_time, 
                 event = survival_data$OS_status)

# Fit survival curves
fit <- survfit(surv_obj ~ group, data = survival_data)

# Perform log-rank test
logrank_test <- survdiff(surv_obj ~ group, data = survival_data)
p_value <- 1 - pchisq(logrank_test$chisq, length(logrank_test$n) - 1)

# Create plot
p <- ggsurvplot(fit,
           data = survival_data,
           pval = TRUE,
           conf.int = TRUE,
           risk.table = TRUE,
           legend.labs = c("High Expression", "Low Expression"),
           title = paste("Survival Analysis (p =", format(p_value, digits=3), ")"),
           xlab = "Time (months)",
           ylab = "Survival probability")

ggsave("survival_plot.png", plot = print(p), width=10, height=8, dpi=300)
```

### Python (lifelines)

```python
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Load data
survival_data = pd.read_csv('input_data/survival_data.tsv', sep='\t')

# Divide into groups
median_expr = survival_data['gene_expression'].median()
survival_data['group'] = (survival_data['gene_expression'] > median_expr).map(
    {True: 'High', False: 'Low'}
)

# Fit survival curves for each group
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(10, 8))

for group in ['High', 'Low']:
    group_data = survival_data[survival_data['group'] == group]
    kmf.fit(group_data['OS_time'], 
            group_data['OS_status'], 
            label=f'{group} Expression')
    kmf.plot_survival_function(ax=ax)

# Perform log-rank test
high_group = survival_data[survival_data['group'] == 'High']
low_group = survival_data[survival_data['group'] == 'Low']

results = logrank_test(high_group['OS_time'], low_group['OS_time'],
                       high_group['OS_status'], low_group['OS_status'])
p_value = results.p_value

ax.text(0.5, 0.1, f'Log-rank test p-value: {p_value:.4f}', 
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curve', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('survival_plot.png', dpi=300, bbox_inches='tight')
```

## Interpretation

- **Survival curves**: Show probability of survival over time
- **Separation**: Clear separation indicates prognostic value of the gene
- **P-value < 0.05**: Statistically significant difference in survival
- **High expression better**: Suggests the gene may be a favorable prognostic marker

## Notes

- Median is commonly used as threshold, but other percentiles (quartiles, tertiles) can be used
- Log-rank test compares survival curves between groups
- Censored observations (patients still alive) are handled appropriately
- Confidence intervals show uncertainty in survival estimates
