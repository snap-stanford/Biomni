# Answer: Histogram Visualization

## Distribution Summary

- **Mean**: 10.5
- **Median**: 10.6
- **Standard deviation**: 0.95
- **Minimum**: 9.0
- **Maximum**: 12.1
- **Sample size**: 15

## Plot Description

The histogram shows a roughly normal distribution of values centered around 10.5, with most values falling between 9.5 and 11.5.

## Example Code

### Python (matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

data = [10.5, 11.2, 10.8, 11.5, 10.9, 9.2, 9.5, 9.0, 9.8, 9.3, 
        12.1, 11.8, 10.2, 11.0, 10.6]

fig, ax = plt.subplots(figsize=(8, 6))

# Create histogram
n, bins, patches = ax.hist(data, bins=10, edgecolor='black', alpha=0.7)

# Add mean and median lines
mean_val = np.mean(data)
median_val = np.median(data)
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

# Labels and title
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Histogram of Data Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
```

### R (ggplot2)

```r
library(ggplot2)

data <- c(10.5, 11.2, 10.8, 11.5, 10.9, 9.2, 9.5, 9.0, 9.8, 9.3, 
          12.1, 11.8, 10.2, 11.0, 10.6)

df <- data.frame(value = data)

ggplot(df, aes(x=value)) +
  geom_histogram(bins=10, fill='steelblue', color='black', alpha=0.7) +
  geom_vline(aes(xintercept=mean(value)), color='red', linetype='dashed', size=1) +
  geom_vline(aes(xintercept=median(value)), color='blue', linetype='dashed', size=1) +
  labs(x='Value', y='Frequency', title='Histogram of Data Distribution') +
  theme_minimal()

ggsave("histogram.png", width=8, height=6, dpi=300)
```

## Interpretation

- **Distribution shape**: Approximately normal (bell-shaped)
- **Center**: Values cluster around 10.5
- **Spread**: Most values within 1-2 standard deviations of the mean
- **Outliers**: No obvious outliers in this dataset

## Notes

- Choose appropriate number of bins (too few = loss of detail, too many = noisy)
- Adding density curve helps visualize the distribution shape
- Mean and median lines help identify skewness
