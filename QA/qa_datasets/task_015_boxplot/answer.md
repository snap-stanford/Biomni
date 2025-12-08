# Answer: Boxplot for Group Comparison

## Summary Statistics by Group

| Group | Mean | Median | Std Dev | Min | Max |
|-------|------|--------|---------|-----|-----|
| Group A | 10.98 | 10.9 | 0.38 | 10.5 | 11.5 |
| Group B | 9.36 | 9.3 | 0.31 | 9.0 | 9.8 |
| Group C | 11.94 | 12.0 | 0.30 | 11.5 | 12.3 |

## Plot Description

The boxplot shows clear differences between the three groups:
- **Group C** has the highest median value (~12.0)
- **Group A** has intermediate values (~10.9)
- **Group B** has the lowest values (~9.3)

## Example Code

### Python (matplotlib, seaborn)

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare data
data = {
    'Group A': [10.5, 11.2, 10.8, 11.5, 10.9],
    'Group B': [9.2, 9.5, 9.0, 9.8, 9.3],
    'Group C': [12.1, 11.8, 12.3, 11.5, 12.0]
}

# Convert to long format
df = pd.DataFrame([(k, v) for k, vals in data.items() for v in vals],
                  columns=['Group', 'Value'])

# Create boxplot
fig, ax = plt.subplots(figsize=(8, 6))

box_plot = ax.boxplot([data[g] for g in data.keys()], 
                       labels=data.keys(),
                       patch_artist=True)

# Color boxes
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

# Optional: Add strip plot overlay
for i, (group, values) in enumerate(data.items(), 1):
    x_positions = np.random.normal(i, 0.04, size=len(values))
    ax.scatter(x_positions, values, alpha=0.5, s=30, color='black')

# Labels and title
ax.set_ylabel('Value', fontsize=12)
ax.set_xlabel('Group', fontsize=12)
ax.set_title('Boxplot: Group Comparison', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
```

### R (ggplot2)

```r
library(ggplot2)

# Prepare data
data <- data.frame(
  Group = rep(c("Group A", "Group B", "Group C"), 
              c(5, 5, 5)),
  Value = c(10.5, 11.2, 10.8, 11.5, 10.9,
            9.2, 9.5, 9.0, 9.8, 9.3,
            12.1, 11.8, 12.3, 11.5, 12.0)
)

# Create boxplot
ggplot(data, aes(x=Group, y=Value, fill=Group)) +
  geom_boxplot(alpha=0.7) +
  geom_jitter(width=0.2, alpha=0.5, size=2) +  # Add data points
  scale_fill_manual(values=c("Group A"="lightblue", 
                              "Group B"="lightgreen", 
                              "Group C"="lightcoral")) +
  labs(x="Group", y="Value", title="Boxplot: Group Comparison") +
  theme_minimal() +
  theme(legend.position="none")

ggsave("boxplot.png", width=8, height=6, dpi=300)
```

## Boxplot Elements Explained

- **Box**: Contains middle 50% of data (IQR: Q3 - Q1)
- **Median line**: Middle value (Q2)
- **Whiskers**: Extend to 1.5×IQR or min/max, whichever is closer
- **Outliers**: Points beyond whiskers (shown as individual points)

## Interpretation

- **Group differences**: Clear separation between groups indicates significant differences
- **Variability**: Box height shows within-group variability
- **Outliers**: Individual points beyond whiskers may indicate unusual values
- **Distribution shape**: Box position relative to whiskers indicates skewness

## Notes

- Boxplots are excellent for comparing distributions across groups
- Adding individual data points (strip plot) shows actual values
- Useful for identifying outliers and distribution shapes
