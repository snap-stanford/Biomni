# Answer: t-test Results

## Descriptive Statistics

### Group 1
- **Mean**: 10.98
- **Standard deviation**: 0.38
- **Sample size**: 5

### Group 2
- **Mean**: 9.36
- **Standard deviation**: 0.31
- **Sample size**: 5

## t-test Results

- **t-statistic**: 7.89
- **p-value**: 0.00015
- **Degrees of freedom**: 7.6 (Welch's t-test, unequal variances)
- **Alternative hypothesis**: Two-sided (means are not equal)

## Interpretation

The t-test shows a **statistically significant difference** between the two groups (p < 0.05).

- Group 1 has a significantly higher mean (10.98) compared to Group 2 (9.36)
- The difference is 1.62 units
- With p = 0.00015 < 0.05, we reject the null hypothesis that the means are equal

## Example Code

### Python (scipy)

```python
from scipy.stats import ttest_ind
import numpy as np

group1 = np.array([10.5, 11.2, 10.8, 11.5, 10.9])
group2 = np.array([9.2, 9.5, 9.0, 9.8, 9.3])

# Perform t-test (Welch's t-test for unequal variances)
t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.5f}")
print(f"Group 1 mean: {group1.mean():.2f}")
print(f"Group 2 mean: {group2.mean():.2f}")
```

### R

```r
group1 <- c(10.5, 11.2, 10.8, 11.5, 10.9)
group2 <- c(9.2, 9.5, 9.0, 9.8, 9.3)

# Perform t-test
result <- t.test(group1, group2, var.equal = FALSE)

print(result)
```

## Notes

- **Welch's t-test** (unequal variances) is recommended when group variances differ
- **Equal variances assumption** can be tested using Levene's test or F-test
- For small sample sizes (< 30), consider non-parametric alternatives (Mann-Whitney U test)
