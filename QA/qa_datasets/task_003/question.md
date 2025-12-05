# Task: Analyzee Western blot image

Please analyze the provided Western blot image, which consists of 3 experimental repetitions. Each repetition includes four conditions: control, P144, TGF-β1, and Tβ1Ab. The targets are PSMAD2, SMAD2, and GAPDH.

Execute the following steps to quantify and visualize the data:

1. **Measure Intensity**: Detect and measure the band intensities for PSMAD2, SMAD2, and GAPDH for all conditions across the 3 repetitions.

2. **Calculate Normalized Values**:
   For each sample, perform the following calculations:
   - **Step A**: Calculate the relative intensity of SMAD2 by dividing the SMAD2 intensity by the GAPDH intensity. (Formula: $SMAD2_{norm} = Intensity_{SMAD2} / Intensity_{GAPDH}$)
   - **Step B**: Calculate the final normalized PSMAD2 intensity by dividing the raw PSMAD2 intensity by the result of Step A. (Formula: $Target_{value} = Intensity_{PSMAD2} / SMAD2_{norm}$)

3. **Relative Quantification (Fold Change)**:
   - Normalize the values so that the 'control' condition is set to 1. To do this, divide the $Target_{value}$ of each condition by the $Target_{value}$ of the 'control' sample within the same repetition.

4. **Statistical Analysis**:
   - Aggregate the data from the 3 repetitions.
   - Calculate the Mean and Standard Deviation (or Standard Error) for each condition.

5. **Visualization**:
   - Generate a bar graph based on the calculated means.
   - Include error bars on the graph to represent the variation across the 3 repetitions.
   - Save the bar graph as `psmad2_quantification.png`
   - Optionally, save a grid verification image showing the detected ROIs as `wb_grid_verification.png`