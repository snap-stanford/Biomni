### qPCR Data Analysis: Fold Change Calculation

I have successfully analyzed the qPCR dataset `qpcr_data.csv` to calculate the Fold Change for each sample and target gene relative to the 'Normal' control group and the 'CHR1' reference gene.

**Methodology:**
1. **Data Cleaning**: Removed duplicate entries to ensure unique Sample-Target pairs.
2. **Delta CT ($\Delta C_T$)**: Calculated by subtracting the $C_T$ of the reference gene ('CHR1') from the target gene $C_T$ for each sample.
3. **Delta Delta CT ($\Delta\Delta C_T$)**: Calculated by subtracting the $\Delta C_T$ of the 'Normal' control group from the $\Delta C_T$ of each sample.
4. **Fold Change**: Computed as $2^{-\Delta\Delta C_T}$.

**Results:**
- The 'Normal' samples have a Fold Change of 1.0 for all genes, serving as the baseline.
- The calculated Fold Change values represent the relative expression levels compared to the Normal control.

**Visualizations:**

1. **Heatmap (`heatmap.png`)**:
   - Displays the Fold Change values for all Sample-Target combinations.
   - Color intensity indicates the magnitude of expression change.
   ![Heatmap](heatmap.png)

2. **Grouped Bar Plot (`boxplot.png`)**:
   - Visualizes the relative expression of the 8 genes across different samples.
   - Bars are grouped by 'Sample Name' for each gene.
   - The red dashed line at y=1 represents the control level.
   ![Grouped Bar Plot](boxplot.png)

**Key Findings:**
- **CHR1** (Reference): Fold change is consistently 1.0 across all samples (normalized to itself).
- **Variations**: Small variations in fold change are observed across different samples for target genes, indicating potential differential expression relative to the Normal control.