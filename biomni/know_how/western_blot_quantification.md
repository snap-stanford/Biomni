# Western Blot Quantification and Analysis

---

## Metadata

**Short Description**: Comprehensive guide for quantifying and analyzing Western blot images with multiple experimental repetitions, including intensity measurement, normalization, statistical analysis, and visualization.

**Authors**: Biomni Team

**Version**: 1.0

**Last Updated**: December 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

---

## Overview

This guide provides a standardized workflow for analyzing Western blot images, particularly for experiments with multiple repetitions and conditions. The protocol covers band intensity detection, normalization procedures, statistical aggregation, and visualization best practices.

## Standard Workflow

### Step 1: Image Preprocessing and Band Detection

**Objective**: Identify ROIs and isolate individual bands in the Western blot image.

**Key Considerations**:
- Use analyze_pixel_distribution then find_roi_from_image functions
- If find_roi_from_image couldn't detect ROIs properly, retry with different lower_threshold and upper_threshold parameters
- If there are still undetected ROIs, you can manually infer the coordinates of undetected ROIs by using the correctly detected ROIs (IMPORTANT: You MUST preserve the coordinates of correctly detected ROIs)
- The final image with ROIs should also be saved as an image
- Detect all bands for target proteins
- Identify experimental repetitions (typically arranged in lanes)
- Recognize different conditions (e.g., control, treatment groups)
- Handle potential artifacts, background noise, and lane alignment issues

**Tools**: analyze_pixel_distribution, find_roi_from_image

### Step 2: Intensity Measurement

**Objective**: Quantify band intensities for all detected bands.

**Procedure**:
1. For each lane/repetition, measure the intensity of:
   - Target protein band (e.g., PSMAD2)
   - Loading control band (e.g., SMAD2, GAPDH)
   - Background intensity (for correction if needed)

2. Record measurements in a structured format:
   - Condition name (e.g., "control", "P144", "TGF-β1", "Tβ1Ab")
   - Repetition number (e.g., Rep1, Rep2, Rep3)
   - Protein target (e.g., PSMAD2, SMAD2, GAPDH)
   - Raw intensity value

**Best Practices**:
- Use consistent ROI (Region of Interest) sizes for all bands
- Apply background subtraction if necessary
- Verify band detection visually before proceeding

### Step 3: Normalization Procedure

**Objective**: Normalize target protein intensities to account for loading variations.

**Two-Step Normalization Process**:

#### Step A: Loading Control Normalization
Calculate the relative intensity of the loading control protein:

```
SMAD2_norm = Intensity_SMAD2 / Intensity_GAPDH
```

This accounts for variations in total protein loading across samples.

#### Step B: Target Protein Normalization
Calculate the final normalized target protein intensity:

```
Target_value = Intensity_PSMAD2 / SMAD2_norm
```

This provides the normalized PSMAD2 intensity that accounts for both loading control and relative protein levels.

**Alternative Normalization Methods**:
- **Single loading control**: If only GAPDH is available: `Target_norm = Intensity_Target / Intensity_GAPDH`
- **Total protein normalization**: If using total protein stain: `Target_norm = Intensity_Target / Intensity_TotalProtein`
- **Housekeeping gene**: Common controls include GAPDH, β-actin, α-tubulin

### Step 4: Relative Quantification (Fold Change Calculation)

**Objective**: Express results relative to a control condition.

**Procedure**:
1. For each experimental repetition, identify the control condition
2. Calculate fold change for each condition:

```
Fold_Change = Target_value_condition / Target_value_control
```

3. The control condition will have a fold change of 1.0 by definition
4. Treatment conditions will show fold changes relative to control (e.g., 1.5 = 50% increase, 0.7 = 30% decrease)

**Important Notes**:
- Always normalize within the same repetition before comparing across repetitions
- Ensure control condition is clearly identified
- Document which condition serves as the baseline

### Step 5: Statistical Aggregation

**Objective**: Combine data from multiple experimental repetitions.

**Procedure**:
1. Collect normalized values (or fold changes) from all repetitions
2. For each condition, calculate:
   - **Mean**: Average across repetitions
   - **Standard Deviation (SD)**: Measure of variability
   - **Standard Error (SE)**: SD / √n, where n = number of repetitions
   - **Sample size (n)**: Number of repetitions

**Statistical Considerations**:
- Minimum of 3 repetitions recommended for meaningful statistics
- Report both mean and error measure (SD or SE)
- Consider statistical tests (t-test, ANOVA) if comparing multiple conditions
- Document any excluded repetitions and reasons

### Step 6: Visualization

**Objective**: Create clear, publication-ready visualizations.

**Bar Graph Requirements**:
1. **X-axis**: Experimental conditions (e.g., Control, P144, TGF-β1, Tβ1Ab)
2. **Y-axis**: Normalized values or fold change (typically starting from 0)
3. **Bars**: Mean values for each condition
4. **Error bars**: Standard deviation or standard error
5. **Labels**: Clear condition names, units, sample size (n=X)

**Visualization Best Practices**:
- Use consistent colors for conditions across figures
- Include statistical significance indicators if applicable (e.g., *, **, ***)
- Add figure title and axis labels
- Save in high resolution (300 DPI for publications)

**Verification Images**:
- Create grid/overlay images showing detected ROIs
- Helps verify correct band detection
- Useful for troubleshooting and quality control
- Save as separate file (e.g., `wb_grid_verification.png`)

## Common Experimental Designs

### Design 1: Multiple Conditions with Replicates
- **Structure**: 3-4 conditions × 3 repetitions = 9-12 lanes
- **Example**: Control, Treatment A, Treatment B, Treatment C (each in triplicate)
- **Analysis**: Normalize within each repetition, then aggregate across repetitions

### Design 2: Time Course
- **Structure**: Multiple time points × conditions × repetitions
- **Example**: 0h, 6h, 12h, 24h for Control and Treatment
- **Analysis**: Normalize to time 0 control, then compare across time points

### Design 3: Dose Response
- **Structure**: Multiple concentrations × repetitions
- **Example**: 0, 1, 5, 10, 50 μM treatment
- **Analysis**: Normalize to 0 concentration, plot dose-response curve

## Troubleshooting

### Issue: Inconsistent Band Detection
**Problem**: Some bands not detected or incorrectly identified
**Solutions**:
- Adjust detection parameters (threshold, size filters)
- Manually verify and correct ROI placement
- Check image quality and contrast
- Use grid verification image to validate detection

### Issue: High Variability Between Repetitions
**Problem**: Large standard deviations or inconsistent results
**Solutions**:
- Verify loading control normalization is correct
- Check for technical artifacts (bubbles, uneven transfer)
- Ensure consistent sample preparation
- Consider excluding outliers if justified

### Issue: Unexpected Normalization Results
**Problem**: Normalized values don't match expected biological response
**Solutions**:
- Verify loading control bands are appropriate (not saturated)
- Check that control condition is correctly identified
- Ensure all calculations are performed in correct order
- Review raw intensity values for anomalies

### Issue: Background Issues
**Problem**: High background affecting intensity measurements
**Solutions**:
- Apply background subtraction
- Use local background measurement near each band
- Adjust image preprocessing (contrast, brightness)
- Consider re-imaging if background is too high

## Quality Control Checklist

Before finalizing analysis, verify:
- [ ] All bands detected correctly (verify with grid image)
- [ ] Loading control bands are present and measurable for all samples
- [ ] Normalization calculations are correct
- [ ] Control condition is clearly identified
- [ ] All repetitions included in statistical analysis
- [ ] Error bars represent appropriate measure (SD or SE)
- [ ] Visualization clearly shows experimental design
- [ ] Results are biologically plausible

## Output Files

**Required Outputs**:
1. **Quantification results**: CSV or Excel file with:
   - Raw intensities
   - Normalized values
   - Fold changes
   - Statistical summaries (mean, SD, SE)

2. **Visualization**: Bar graph image (e.g., `psmad2_quantification.png`)
   - High resolution (300 DPI minimum)
   - Clear labels and error bars
   - Publication-ready format

3. **Verification image** (optional but recommended): `wb_grid_verification.png`
   - Shows detected ROIs
   - Helps validate analysis

## Example Workflow Summary

For a typical experiment with 3 repetitions and 4 conditions:

1. **Detect bands**: Identify all PSMAD2, SMAD2, and GAPDH bands across 12 lanes
2. **Measure intensities**: Extract intensity values for each band
3. **Normalize**: 
   - Step A: SMAD2_norm = SMAD2 / GAPDH (for each sample)
   - Step B: Target_value = PSMAD2 / SMAD2_norm (for each sample)
4. **Calculate fold change**: Target_value_condition / Target_value_control (within each repetition)
5. **Aggregate**: Calculate mean ± SD across 3 repetitions for each condition
6. **Visualize**: Create bar graph with error bars
7. **Save**: Export quantification table and visualization images

## Key Formulas Reference

```
Loading Control Normalization:
  Loading_norm = Intensity_LoadingControl / Intensity_Housekeeping

Target Normalization:
  Target_norm = Intensity_Target / Loading_norm

Fold Change:
  Fold_Change = Target_norm_condition / Target_norm_control

Statistics:
  Mean = Σ(values) / n
  SD = √[Σ(value - mean)² / (n-1)]
  SE = SD / √n
```

## Best Practices

1. **Always normalize**: Never use raw intensities without normalization
2. **Use appropriate controls**: Choose loading controls that are stable across conditions
3. **Verify detection**: Always review grid/verification images
4. **Document exclusions**: Note any excluded samples and reasons
5. **Report statistics**: Include both mean and error measure
6. **Save intermediate data**: Keep raw intensities for potential re-analysis
7. **Visual validation**: Compare visualization with expected biological response
