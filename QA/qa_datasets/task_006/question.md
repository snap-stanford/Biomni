Using the provided dataset, perform a dose-response analysis and generate a smooth curve based on the 'Sample' (Dose) and 'Activity' (Response) columns.

Follow these detailed steps to achieve a result similar to the reference image (smooth curve with IC50):

1. **Data Assignment**: Use the 'Sample' column for the X-axis (Dose/Concentration) and the 'Activity' column for the Y-axis (% Control/Response).

2. **Curve Fitting**:
   - Perform a non-linear regression analysis using a **4-parameter logistic (4PL) model** (also known as the Hill equation) to fit a smooth curve to the experimental data.

3. **Calculate IC50**:
   - Based on the fitted 4PL model, calculate the **IC50 value** (the concentration at which the response is reduced by 50% between the top and bottom plateaus).

4. **Visualization**:
   - **X-axis Scale**: Apply a **logarithmic scale** to the X-axis.
   - **Experimental Data**: Plot the original data points as a scatter plot (e.g., red dots).
   - **Fitted Curve**: Overlay the **smooth fitted 4PL curve** onto the scatter plot (e.g., a blue line).
   - **IC50 Marker**: Visually mark the IC50 concentration on the graph (e.g., using a dashed vertical line) and explicitly display the calculated **IC50 value** in the graph legend or as a text annotation.
   - **Labels & Grid**: Add clear labels for the X and Y axes (including units if available), a descriptive title, and a background grid for better readability.

## Generated Files:
`dose_response_curve.png` - Shows the dose-response curve with log-scaled doses and activity.