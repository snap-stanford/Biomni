### Dose-Response Analysis Results

I have successfully performed the dose-response analysis on the provided dataset.

**1. Data Processing**
- **Input File**: `dose_response_data.csv`
- **Cleaning**: Extracted numeric dose values from the 'Sample' column and converted 'Activity' to numeric format. Handled replicates by forward-filling sample names.

**2. Curve Fitting (4PL Model)**
A 4-parameter logistic (Hill equation) model was fitted to the data:
- **Top Plateau**: 99.91%
- **Bottom Plateau**: -2.61%
- **Hill Slope**: 0.85
- **IC50**: **2.1694 µM**

**3. Visualization**
The generated plot (`dose_response_curve.png`) includes:
- **X-axis**: Log-scaled Concentration (µM)
- **Y-axis**: Activity (% Control)
- **Red Dots**: Experimental data points
- **Blue Line**: Smooth fitted 4PL curve
- **Green Dashed Line**: Visual marker for the IC50 value

![Dose Response Curve](dose_response_curve.png)

The analysis confirms a clear dose-dependent inhibition with an IC50 of approximately 2.17 µM.