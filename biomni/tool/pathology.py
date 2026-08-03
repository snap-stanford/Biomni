def analyze_aortic_diameter_and_geometry(image_path, output_dir="./output"):
    """Analyze aortic diameter and geometry from cardiovascular imaging data.

    This function processes cardiovascular imaging data (ultrasound or CT/MRI) to
    measure aortic root diameter, ascending aorta diameter, and calculate
    geometric parameters such as tortuosity and dilation indices.

    Parameters
    ----------
    image_path : str
        Path to the cardiovascular imaging data (DICOM, JPG, PNG)
    output_dir : str, optional
        Directory to save output files (default: "./output")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = []
    log.append(f"Aortic Diameter and Geometry Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Input image: {image_path}")

    # Step 1: Load the image
    log.append("\n1. Loading and preprocessing image")
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"Error: File {image_path} not found"

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return f"Error: Could not read image {image_path}"

        log.append(f"  - Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Basic preprocessing
        image = cv2.GaussianBlur(image, (5, 5), 0)
        log.append("  - Applied Gaussian blur for noise reduction")

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        log.append("  - Enhanced contrast using CLAHE")

    except Exception as e:
        return f"Error during image loading: {str(e)}"

    # Step 2: Segment the aorta
    log.append("\n2. Aorta segmentation")
    try:
        # Otsu thresholding for initial segmentation
        thresh_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        log.append(f"  - Applied Otsu thresholding (threshold value: {thresh_val})")

        # Morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        log.append("  - Applied morphological operations to refine segmentation")

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        log.append(f"  - Detected {len(contours)} contours")

        # Assuming the aorta is one of the largest contours
        if len(contours) == 0:
            return "Error: No contours detected in the image"

        # Sort contours by area and take the largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        aorta_contour = contours[0]  # Assuming the largest contour is the aorta
        log.append(f"  - Selected largest contour as aorta (area: {cv2.contourArea(aorta_contour)} px²)")

    except Exception as e:
        return f"Error during aorta segmentation: {str(e)}"

    # Step 3: Measure aortic diameters
    log.append("\n3. Measuring aortic diameters")
    try:
        # Create a mask of the aorta
        aorta_mask = np.zeros_like(image)
        cv2.drawContours(aorta_mask, [aorta_contour], 0, 255, -1)

        # Find the centerline of the aorta (simplified approach)
        # This is a simplified approach - in a real implementation, skeletonization would be better
        M = cv2.moments(aorta_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            log.append(f"  - Aorta centroid: ({cx}, {cy})")
        else:
            cx, cy = 0, 0
            log.append("  - Warning: Could not calculate aorta centroid")

        # Measure aortic root diameter (assuming it's at the bottom of the contour)
        aortic_root_points = sorted(aorta_contour.reshape(-1, 2), key=lambda p: p[1], reverse=True)[:20]
        aortic_root_diameter = np.max([p[0] for p in aortic_root_points]) - np.min([p[0] for p in aortic_root_points])
        log.append(f"  - Aortic root diameter: {aortic_root_diameter:.2f} pixels")

        # Measure ascending aorta diameter (assuming it's at the middle of the contour)
        y_mid = (
            np.min([p[1] for p in aorta_contour.reshape(-1, 2)]) + np.max([p[1] for p in aorta_contour.reshape(-1, 2)])
        ) // 2
        ascending_points = [p for p in aorta_contour.reshape(-1, 2) if abs(p[1] - y_mid) < 10]
        if ascending_points:
            ascending_diameter = np.max([p[0] for p in ascending_points]) - np.min([p[0] for p in ascending_points])
            log.append(f"  - Ascending aorta diameter: {ascending_diameter:.2f} pixels")
        else:
            ascending_diameter = 0
            log.append("  - Warning: Could not measure ascending aorta diameter")

    except Exception as e:
        return f"Error during diameter measurements: {str(e)}"

    # Step 4: Calculate geometric parameters
    log.append("\n4. Calculating aortic geometry parameters")
    try:
        # Calculate tortuosity (simplified as the ratio of contour length to end-to-end distance)
        contour_length = cv2.arcLength(aorta_contour, closed=True)

        # Find the two most distant points (approximation for end-to-end distance)
        hull = cv2.convexHull(aorta_contour)
        hull_points = hull.reshape(-1, 2)
        max_dist = 0
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                dist = np.sqrt(
                    (hull_points[i][0] - hull_points[j][0]) ** 2 + (hull_points[i][1] - hull_points[j][1]) ** 2
                )
                max_dist = max(max_dist, dist)

        if max_dist > 0:
            tortuosity = contour_length / max_dist
            log.append(f"  - Aortic tortuosity index: {tortuosity:.2f}")
        else:
            tortuosity = 0
            log.append("  - Warning: Could not calculate tortuosity")

        # Calculate dilation index (ratio of max diameter to min diameter)
        min_diameter = min(aortic_root_diameter, ascending_diameter)
        max_diameter = max(aortic_root_diameter, ascending_diameter)

        if min_diameter > 0:
            dilation_index = max_diameter / min_diameter
            log.append(f"  - Aortic dilation index: {dilation_index:.2f}")
        else:
            dilation_index = 0
            log.append("  - Warning: Could not calculate dilation index")

    except Exception as e:
        return f"Error during geometry calculations: {str(e)}"

    # Step 5: Save results
    log.append("\n5. Saving results")
    try:
        # Create a visual output
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, [aorta_contour], 0, (0, 255, 0), 2)

        # Mark diameters
        cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)

        # Save the output image
        output_filename = os.path.join(output_dir, f"aorta_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(output_filename, output_image)
        log.append(f"  - Saved annotated image to: {output_filename}")

        # Save measurements to a text file
        measurements_filename = os.path.join(
            output_dir,
            f"aorta_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        with open(measurements_filename, "w") as f:
            f.write(f"Aortic Root Diameter: {aortic_root_diameter:.2f} pixels\n")
            f.write(f"Ascending Aorta Diameter: {ascending_diameter:.2f} pixels\n")
            f.write(f"Tortuosity Index: {tortuosity:.2f}\n")
            f.write(f"Dilation Index: {dilation_index:.2f}\n")
        log.append(f"  - Saved measurements to: {measurements_filename}")

    except Exception as e:
        return f"Error during saving results: {str(e)}"

    # Step 6: Summarize findings
    log.append("\n6. Summary of findings")
    log.append(f"  - Aortic root diameter: {aortic_root_diameter:.2f} pixels")
    log.append(f"  - Ascending aorta diameter: {ascending_diameter:.2f} pixels")
    log.append(f"  - Tortuosity index: {tortuosity:.2f}")
    log.append(f"  - Dilation index: {dilation_index:.2f}")

    return "\n".join(log)


def analyze_atp_luminescence_assay(
    data_file,
    standard_curve_file,
    normalization_method="cell_count",
    normalization_data=None,
):
    """Analyze luminescence-based ATP assay data to determine intracellular ATP concentration.

    Parameters
    ----------
    data_file : str
        Path to CSV file containing luminescence readings from samples.
        Expected format: columns for 'Sample_ID' and 'Luminescence_Value'.

    standard_curve_file : str
        Path to CSV file containing standard curve data.
        Expected format: columns for 'ATP_Concentration' (in nM) and 'Luminescence_Value'.

    normalization_method : str, optional
        Method used to normalize ATP values, either 'cell_count' or 'protein_content'.
        Default is 'cell_count'.

    normalization_data : str or dict, optional
        Either path to CSV file with normalization data or dictionary with sample IDs as keys.
        For 'cell_count': values should be cell counts per sample.
        For 'protein_content': values should be protein concentration (μg/mL).

    Returns
    -------
    str
        Research log summarizing the ATP content measurement process and results.

    """
    from datetime import datetime

    import numpy as np
    import pandas as pd

    log = []
    log.append(f"ATP Content Measurement Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 50)

    # Step 1: Load the raw luminescence data
    log.append("Step 1: Loading raw luminescence data")
    try:
        sample_data = pd.read_csv(data_file)
        log.append(f"Successfully loaded data from {data_file}")
        log.append(f"Number of samples: {len(sample_data)}")
    except Exception as e:
        log.append(f"Error loading sample data: {str(e)}")
        return "\n".join(log)

    # Step 2: Load the standard curve data
    log.append("\nStep 2: Loading ATP standard curve data")
    try:
        std_curve_data = pd.read_csv(standard_curve_file)
        log.append(f"Successfully loaded standard curve from {standard_curve_file}")
    except Exception as e:
        log.append(f"Error loading standard curve data: {str(e)}")
        return "\n".join(log)

    # Step 3: Generate standard curve equation (linear regression)
    log.append("\nStep 3: Generating ATP standard curve")
    try:
        x = std_curve_data["Luminescence_Value"]
        y = std_curve_data["ATP_Concentration"]
        slope, intercept = np.polyfit(x, y, 1)
        log.append(f"Standard curve equation: ATP_Concentration = {slope:.6f} × Luminescence + {intercept:.6f}")
    except Exception as e:
        log.append(f"Error generating standard curve: {str(e)}")
        return "\n".join(log)

    # Step 4: Calculate ATP concentrations for samples
    log.append("\nStep 4: Calculating ATP concentrations for samples")
    try:
        sample_data["ATP_Concentration_nM"] = slope * sample_data["Luminescence_Value"] + intercept
        log.append("ATP concentrations calculated for all samples")
    except Exception as e:
        log.append(f"Error calculating ATP concentrations: {str(e)}")
        return "\n".join(log)

    # Step 5: Normalize ATP concentrations
    log.append(f"\nStep 5: Normalizing ATP concentrations by {normalization_method}")

    # Load normalization data
    if normalization_data:
        try:
            if isinstance(normalization_data, str):
                norm_data = pd.read_csv(normalization_data)
                norm_dict = dict(zip(norm_data["Sample_ID"], norm_data[normalization_method], strict=False))
            else:
                norm_dict = normalization_data

            # Apply normalization
            for idx, row in sample_data.iterrows():
                sample_id = row["Sample_ID"]
                if sample_id in norm_dict:
                    if normalization_method == "cell_count":
                        sample_data.at[idx, "ATP_pmol_per_million_cells"] = (
                            row["ATP_Concentration_nM"] / norm_dict[sample_id] * 1000
                        )
                    elif normalization_method == "protein_content":
                        sample_data.at[idx, "ATP_nmol_per_mg_protein"] = (
                            row["ATP_Concentration_nM"] / norm_dict[sample_id]
                        )

            log.append(f"Successfully normalized ATP concentrations by {normalization_method}")
        except Exception as e:
            log.append(f"Error during normalization: {str(e)}")
            log.append("Continuing with unnormalized ATP concentrations")
    else:
        log.append("No normalization data provided. Reporting unnormalized ATP concentrations.")

    # Step 6: Save results
    output_file = "atp_measurement_results.csv"
    try:
        sample_data.to_csv(output_file, index=False)
        log.append(f"\nStep 6: Results saved to {output_file}")
    except Exception as e:
        log.append(f"\nStep 6: Error saving results: {str(e)}")

    # Step 7: Summary statistics
    log.append("\nStep 7: Summary of ATP measurement results")
    try:
        log.append("ATP Concentration Summary (nM):")
        log.append(f"  Mean: {sample_data['ATP_Concentration_nM'].mean():.2f}")
        log.append(f"  Median: {sample_data['ATP_Concentration_nM'].median():.2f}")
        log.append(f"  Min: {sample_data['ATP_Concentration_nM'].min():.2f}")
        log.append(f"  Max: {sample_data['ATP_Concentration_nM'].max():.2f}")

        if normalization_method == "cell_count" and "ATP_pmol_per_million_cells" in sample_data.columns:
            log.append("\nATP per Million Cells (pmol):")
            log.append(f"  Mean: {sample_data['ATP_pmol_per_million_cells'].mean():.2f}")
            log.append(f"  Median: {sample_data['ATP_pmol_per_million_cells'].median():.2f}")
        elif normalization_method == "protein_content" and "ATP_nmol_per_mg_protein" in sample_data.columns:
            log.append("\nATP per mg Protein (nmol):")
            log.append(f"  Mean: {sample_data['ATP_nmol_per_mg_protein'].mean():.2f}")
            log.append(f"  Median: {sample_data['ATP_nmol_per_mg_protein'].median():.2f}")
    except Exception as e:
        log.append(f"Error generating summary statistics: {str(e)}")

    log.append("\nATP Content Measurement Analysis Complete")

    return "\n".join(log)


def analyze_thrombus_histology(image_path, output_dir="./output"):
    """Analyze histological images of thrombus samples stained with H&E to identify and quantify
    different thrombus components (fresh, cellular lysis, endothelialization, fibroblastic reaction).

    Parameters
    ----------
    image_path : str
        Path to the histological image of thrombus sample stained with H&E
    output_dir : str, optional
        Directory to save output files (default: "./output")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np
    from skimage import color

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize research log
    log = f"# Thrombus Component Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += f"Analyzing image: {image_path}\n\n"

    # Step 1: Load and preprocess the image
    log += "## Step 1: Image Loading and Preprocessing\n"
    original_image = cv2.imread(image_path)
    if original_image is None:
        return f"Error: Could not load image from {image_path}"

    # Convert to RGB (from BGR)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    log += f"- Loaded image with dimensions: {rgb_image.shape[1]}x{rgb_image.shape[0]} pixels\n"

    # Convert to LAB color space for better color segmentation
    lab_image = color.rgb2lab(rgb_image)
    log += "- Converted image to LAB color space for improved color-based segmentation\n\n"

    # Step 2: Segment thrombus components based on color characteristics
    log += "## Step 2: Thrombus Component Segmentation\n"

    # Create masks for different components based on color thresholds
    # These thresholds are approximate and may need adjustment for specific staining protocols

    # Fresh thrombus: typically red (erythrocytes) with blue nuclei (leukocytes)
    # Usually appears as bright red/pink areas with scattered dark blue/purple nuclei
    fresh_mask = (
        (lab_image[:, :, 0] > 50)  # Lightness
        & (lab_image[:, :, 1] > 15)  # a* channel (red component)
    )

    # Cellular lysis: degraded cellular components, less intense staining
    lysis_mask = (
        (lab_image[:, :, 0] > 60)  # Lightness
        & (lab_image[:, :, 1] < 15)  # Less red
        & (lab_image[:, :, 1] > -5)
        & (lab_image[:, :, 2] < 10)  # Less blue
    )

    # Endothelialization: endothelial cells lining the thrombus
    # Typically appears as thin, organized cellular layers
    endothel_mask = (
        (lab_image[:, :, 0] > 40) & (lab_image[:, :, 0] < 70) & (lab_image[:, :, 2] < -5)  # More blue component
    )

    # Fibroblastic reaction: fibroblasts and collagen deposition
    # Usually appears as pale pink/purple fibrous structures
    fibro_mask = (
        (lab_image[:, :, 0] > 70)  # Lighter areas
        & (lab_image[:, :, 1] < 5)  # Less red
        & (lab_image[:, :, 1] > -10)
        & (lab_image[:, :, 2] > 0)  # Less blue
    )

    log += "- Created masks for each thrombus component based on color characteristics\n"

    # Step 3: Quantify the components
    log += "\n## Step 3: Component Quantification\n"

    # Calculate total pixel count (excluding background)
    total_pixels = fresh_mask.sum() + lysis_mask.sum() + endothel_mask.sum() + fibro_mask.sum()

    # Calculate percentages
    if total_pixels > 0:
        fresh_percent = (fresh_mask.sum() / total_pixels) * 100
        lysis_percent = (lysis_mask.sum() / total_pixels) * 100
        endothel_percent = (endothel_mask.sum() / total_pixels) * 100
        fibro_percent = (fibro_mask.sum() / total_pixels) * 100
    else:
        fresh_percent = lysis_percent = endothel_percent = fibro_percent = 0

    log += f"- Fresh thrombus: {fresh_percent:.2f}%\n"
    log += f"- Cellular lysis: {lysis_percent:.2f}%\n"
    log += f"- Endothelialization: {endothel_percent:.2f}%\n"
    log += f"- Fibroblastic reaction: {fibro_percent:.2f}%\n\n"

    # Step 4: Visualize the results
    log += "## Step 4: Visualization\n"

    # Create a visualization image
    visualization = np.zeros_like(rgb_image)

    # Assign colors to different components
    visualization[fresh_mask] = [255, 0, 0]  # Red for fresh thrombus
    visualization[lysis_mask] = [0, 255, 0]  # Green for cellular lysis
    visualization[endothel_mask] = [0, 0, 255]  # Blue for endothelialization
    visualization[fibro_mask] = [255, 255, 0]  # Yellow for fibroblastic reaction

    # Save the visualization
    output_filename = os.path.join(output_dir, f"thrombus_components_{os.path.basename(image_path)}")
    cv2.imwrite(output_filename, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    log += f"- Visualization saved as: {output_filename}\n"
    log += "- Color legend:\n"
    log += "  - Red: Fresh thrombus\n"
    log += "  - Green: Cellular lysis\n"
    log += "  - Blue: Endothelialization\n"
    log += "  - Yellow: Fibroblastic reaction\n\n"

    # Step 5: Summary
    log += "## Summary\n"
    log += "The histological analysis of the thrombus sample revealed the following composition:\n\n"
    log += f"1. Fresh thrombus: {fresh_percent:.2f}%\n"
    log += f"2. Cellular lysis: {lysis_percent:.2f}%\n"
    log += f"3. Endothelialization: {endothel_percent:.2f}%\n"
    log += f"4. Fibroblastic reaction: {fibro_percent:.2f}%\n\n"

    # Save results to CSV
    import csv

    csv_filename = os.path.join(
        output_dir,
        f"thrombus_analysis_{os.path.basename(image_path).split('.')[0]}.csv",
    )
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Component", "Percentage"])
        writer.writerow(["Fresh thrombus", f"{fresh_percent:.2f}"])
        writer.writerow(["Cellular lysis", f"{lysis_percent:.2f}"])
        writer.writerow(["Endothelialization", f"{endothel_percent:.2f}"])
        writer.writerow(["Fibroblastic reaction", f"{fibro_percent:.2f}"])

    log += f"Quantitative results saved to: {csv_filename}\n"

    return log


def analyze_intracellular_calcium_with_rhod2(
    background_image_path, control_image_path, sample_image_path, output_dir="./output"
):
    """Analyzes intracellular calcium concentration using Rhod-2 fluorescent indicator from microscopy images.

    Parameters
    ----------
    background_image_path : str
        Path to the background image (no cells, just media)
    control_image_path : str
        Path to the control image (cells without calcium stimulus)
    sample_image_path : str
        Path to the sample image (cells with calcium stimulus)
    output_dir : str, optional
        Directory to save output files, default is "./output"

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load images
    background = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    control = cv2.imread(control_image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    sample = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE).astype(float)

    # Step 2: Background subtraction
    control_corrected = np.maximum(control - background, 0)
    sample_corrected = np.maximum(sample - background, 0)

    # Step 3: Calculate fluorescence intensity (mean values)
    control_intensity = np.mean(control_corrected)
    sample_intensity = np.mean(sample_corrected)

    # Step 4: Convert to calcium concentration using a standard calibration equation
    # [Ca2+] = Kd * (F - Fmin) / (Fmax - F)
    # For Rhod-2, Kd is approximately 570 nM
    # We'll use control as Fmin and estimate Fmax as 2.5x the sample value (typical for max calcium response)
    kd = 570  # nM
    f_min = control_intensity
    f_max = 2.5 * sample_intensity  # Estimated maximum
    f = sample_intensity

    # Guard against division by zero
    calcium_concentration = float("inf") if f_max == f else kd * (f - f_min) / (f_max - f)

    # Step 5: Generate heatmap visualization of calcium concentration
    plt.figure(figsize=(10, 8))
    calcium_map = (sample_corrected - control_corrected) / (f_max - control_corrected + 1e-10) * kd
    plt.imshow(calcium_map, cmap="hot")
    plt.colorbar(label="[Ca²⁺] (nM)")
    plt.title("Intracellular Calcium Concentration Map")

    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{output_dir}/calcium_map_{timestamp}.png"
    plt.savefig(result_filename)
    plt.close()

    # Generate research log
    log = f"""
Intracellular Calcium Imaging Analysis Log - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
------------------------------------------------------------
1. Loaded microscopy images:
   - Background image: {background_image_path}
   - Control image: {control_image_path}
   - Sample image: {sample_image_path}

2. Performed background subtraction to correct for autofluorescence

3. Calculated fluorescence intensity:
   - Control intensity: {control_intensity:.2f} a.u.
   - Sample intensity: {sample_intensity:.2f} a.u.

4. Converted fluorescence to calcium concentration using Rhod-2 calibration:
   - Kd value: {kd} nM
   - Mean intracellular [Ca²⁺]: {calcium_concentration:.2f} nM

5. Generated calcium concentration heatmap saved to:
   - {result_filename}

Analysis complete. The sample shows {calcium_concentration:.2f} nM intracellular calcium concentration.
"""

    return log


def quantify_corneal_nerve_fibers(image_path, marker_type, output_dir="./output", threshold_method="otsu"):
    """Quantify the volume/density of immunofluorescence-labeled corneal nerve fibers.

    Parameters
    ----------
    image_path : str
        Path to the immunofluorescence microscopy image file
    marker_type : str
        Type of nerve fiber marker (e.g., 'βIII-tubulin', 'SP', 'L1CAM')
    output_dir : str, optional
        Directory to save output files (default: './output')
    threshold_method : str, optional
        Method for thresholding ('otsu', 'adaptive', 'manual'), default is 'otsu'

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import numpy as np
    from skimage import filters, io, measure, morphology, util
    from skimage.color import rgb2gray

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{marker_type.replace(' ', '_')}_{timestamp}"

    # Step 1: Load the image
    try:
        image = io.imread(image_path)
        # Convert to grayscale if RGB
        if len(image.shape) > 2:
            image = rgb2gray(image)
    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Step 2: Preprocess the image
    # Normalize to 0-1 range
    image_normalized = image.astype(float)
    image_normalized = (image_normalized - image_normalized.min()) / (image_normalized.max() - image_normalized.min())

    # Step 3: Segment the nerve fibers
    if threshold_method == "otsu":
        threshold_value = filters.threshold_otsu(image_normalized)
    elif threshold_method == "adaptive":
        threshold_value = filters.threshold_local(image_normalized, block_size=35)
    else:  # manual
        threshold_value = 0.5  # Default manual threshold

    # Create binary mask
    binary_mask = image_normalized > threshold_value

    # Step 4: Clean up with morphological operations
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=50)
    # Fill holes
    cleaned_mask = morphology.closing(cleaned_mask, morphology.disk(2))

    # Step 5: Quantify the nerve fibers
    # Calculate total area of nerve fibers (in pixels)
    fiber_area = np.sum(cleaned_mask)
    # Calculate total image area
    total_area = cleaned_mask.size
    # Calculate fiber density (percentage of area covered by fibers)
    fiber_density = (fiber_area / total_area) * 100

    # Calculate properties of individual fiber segments
    labeled_fibers = measure.label(cleaned_mask)
    fiber_props = measure.regionprops(labeled_fibers)

    # Calculate average fiber length and width
    total_length = 0
    total_width = 0
    fiber_count = len(fiber_props)

    if fiber_count > 0:
        for prop in fiber_props:
            # Approximating length as major axis length
            total_length += prop.major_axis_length
            # Approximating width as minor axis length
            total_width += prop.minor_axis_length

        avg_length = total_length / fiber_count
        avg_width = total_width / fiber_count
    else:
        avg_length = 0
        avg_width = 0

    # Step 6: Save results
    # Save segmented image
    segmented_image_path = os.path.join(output_dir, f"{base_filename}_segmented.png")
    io.imsave(segmented_image_path, util.img_as_ubyte(cleaned_mask))

    # Save measurements to CSV
    measurements_path = os.path.join(output_dir, f"{base_filename}_measurements.csv")
    with open(measurements_path, "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Marker Type,{marker_type}\n")
        f.write(f"Fiber Area (pixels),{fiber_area}\n")
        f.write(f"Total Image Area (pixels),{total_area}\n")
        f.write(f"Fiber Density (%),{fiber_density:.2f}\n")
        f.write(f"Fiber Count,{fiber_count}\n")
        f.write(f"Average Fiber Length (pixels),{avg_length:.2f}\n")
        f.write(f"Average Fiber Width (pixels),{avg_width:.2f}\n")

    # Step 7: Generate research log
    log = f"""
Research Log: Quantification of {marker_type} Labeled Corneal Nerve Fibers
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {os.path.basename(image_path)}

Analysis Steps:
1. Loaded and preprocessed the immunofluorescence microscopy image
2. Applied {threshold_method} thresholding for nerve fiber segmentation
3. Performed morphological operations to clean the segmentation
4. Quantified nerve fiber metrics

Results:
- Fiber Area: {fiber_area} pixels
- Total Image Area: {total_area} pixels
- Fiber Density: {fiber_density:.2f}%
- Number of Fiber Segments: {fiber_count}
- Average Fiber Length: {avg_length:.2f} pixels
- Average Fiber Width: {avg_width:.2f} pixels

Output Files:
- Segmented Image: {os.path.basename(segmented_image_path)}
- Measurements: {os.path.basename(measurements_path)}
"""

    return log


def segment_and_quantify_cells_in_multiplexed_images(
    image_path, markers_list, nuclear_channel_index=0, output_dir="./output"
):
    """Segment cells and quantify protein expression levels from multichannel tissue images.

    Parameters
    ----------
    image_path : str
        Path to the multichannel image file (tiff stack or similar format)
    markers_list : list of str
        List of marker names corresponding to each channel in the image
    nuclear_channel_index : int, optional
        Index of the nuclear marker channel (default: 0, typically DAPI)
    output_dir : str, optional
        Directory to save output files (default: "./output")

    Returns
    -------
    str
        Research log summarizing the steps performed and output file locations

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from scipy import ndimage  # Import ndimage for distance transform
    from skimage import filters, io, measure, morphology, segmentation

    # Initialize research log
    log = []
    log.append(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting cell segmentation and protein quantification"
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.append(f"Created output directory: {output_dir}")

    # Load the multichannel image
    try:
        log.append(f"Loading multichannel image from {image_path}")
        image = io.imread(image_path)
        log.append(f"Image loaded successfully. Shape: {image.shape}")

        # Validate image dimensions
        if len(image.shape) < 3:
            return (
                "Error: Image should be multichannel with shape (channels, height, width) or (height, width, channels)"
            )

        # Determine image format
        if image.shape[0] == len(markers_list):
            # Format is (channels, height, width)
            num_channels, height, width = image.shape
            # Extract nuclear channel
            nuclear_channel = image[nuclear_channel_index]
        elif image.shape[-1] == len(markers_list):
            # Format is (height, width, channels)
            height, width, num_channels = image.shape
            # Extract nuclear channel
            nuclear_channel = image[:, :, nuclear_channel_index]
        else:
            return f"Error: Number of channels in image ({image.shape}) doesn't match markers list length ({len(markers_list)})"

        log.append(f"Processing image with {num_channels} channels, dimensions: {height}x{width}")

    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Segment nuclei
    log.append("Performing nuclear segmentation")
    # Apply threshold to nuclear channel
    threshold = filters.threshold_otsu(nuclear_channel)
    binary_nuclei = nuclear_channel > threshold

    # Clean up binary image
    binary_nuclei = morphology.remove_small_objects(binary_nuclei, min_size=50)
    binary_nuclei = morphology.binary_closing(binary_nuclei, morphology.disk(2))

    # Label nuclei
    labeled_nuclei = measure.label(binary_nuclei)
    log.append(f"Identified {np.max(labeled_nuclei)} potential nuclei")

    # Create cell masks by expanding nuclei
    log.append("Expanding nuclear masks to approximate cell boundaries")
    cell_masks = segmentation.watershed(
        -ndimage.distance_transform_edt(~binary_nuclei),  # Use ndimage for distance transform
        labeled_nuclei,
        mask=morphology.binary_dilation(binary_nuclei, morphology.disk(10)),
    )

    # Get region properties for nuclei
    regions = measure.regionprops(cell_masks)
    log.append(f"Segmented {len(regions)} cells")

    # Create a dataframe to store cell features
    log.append("Extracting features and quantifying marker intensities")
    cell_data = []

    for i, region in enumerate(regions):
        cell_features = {
            "cell_id": i + 1,
            "centroid_y": region.centroid[0],
            "centroid_x": region.centroid[1],
            "area": region.area,
        }

        # Extract intensity for each marker
        for channel_idx, marker_name in enumerate(markers_list):
            if image.shape[0] == len(markers_list):
                # Format is (channels, height, width)
                marker_image = image[channel_idx]
            else:
                # Format is (height, width, channels)
                marker_image = image[:, :, channel_idx]

            # Create mask for current cell
            cell_mask = cell_masks == region.label

            # Calculate mean intensity within the cell mask
            mean_intensity = np.mean(marker_image[cell_mask])
            cell_features[f"{marker_name}_mean_intensity"] = mean_intensity

        cell_data.append(cell_features)

    # Create dataframe
    cell_df = pd.DataFrame(cell_data)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"cell_features_{timestamp}.csv")
    cell_df.to_csv(results_file, index=False)
    log.append(f"Saved spatial feature table to {results_file}")

    # Save segmentation mask for visualization
    mask_file = os.path.join(output_dir, f"cell_segmentation_mask_{timestamp}.tiff")
    io.imsave(mask_file, cell_masks.astype(np.uint16))
    log.append(f"Saved cell segmentation mask to {mask_file}")

    # Summarize results
    log.append(f"Analysis complete. Processed {num_channels} markers across {len(regions)} cells.")
    log.append("Output files:")
    log.append(f"  - Spatial feature table: {results_file}")
    log.append(f"  - Cell segmentation mask: {mask_file}")

    return "\n".join(log)


def analyze_bone_microct_morphometry(input_file_path, output_dir="./results", threshold_value=None):
    """Analyze bone microarchitecture parameters from 3D micro-CT images.

    Performs quantitative analysis of bone microstructure to calculate bone mineral density (BMD),
    bone volume (BV), trabecular number (Tb.N), trabecular thickness (Tb.Th), and
    trabecular separation (Tb.S) from micro-CT data.

    Parameters
    ----------
    input_file_path : str
        Path to the micro-CT scan data file (TIFF stack or similar 3D format)
    output_dir : str, optional
        Directory to save output files, default is "./results"
    threshold_value : float, optional
        Threshold value for bone segmentation. If None, Otsu's method will be used

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import json
    import os
    from datetime import datetime

    import numpy as np
    from scipy import ndimage
    from skimage import filters, io

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Start research log
    log = [
        "# Micro-CT Bone Morphometry Analysis",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {input_file_path}",
    ]

    # Step 1: Load the micro-CT data
    log.append("\n## 1. Data Loading")
    try:
        log.append(f"Loading 3D micro-CT data from {input_file_path}")
        # Load the 3D image data
        image_data = io.imread(input_file_path)
        if image_data.ndim < 3:
            log.append("Warning: Input appears to be 2D. Converting to 3D.")
            image_data = np.expand_dims(image_data, axis=0)

        log.append(f"Data loaded successfully. Dimensions: {image_data.shape}")
        voxel_count = image_data.size
        log.append(f"Total voxels: {voxel_count}")
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        log.append(error_msg)
        return "\n".join(log)

    # Step 2: Preprocess the data
    log.append("\n## 2. Preprocessing")
    log.append("Applying median filter to reduce noise")
    filtered_data = ndimage.median_filter(image_data, size=2)

    # Step 3: Segmentation of bone tissue
    log.append("\n## 3. Bone Segmentation")
    if threshold_value is None:
        log.append("Calculating optimal threshold using Otsu's method")
        threshold_value = filters.threshold_otsu(filtered_data)

    log.append(f"Segmenting bone using threshold value: {threshold_value}")
    binary_image = filtered_data > threshold_value

    # Step 4: Calculate bone morphometry parameters
    log.append("\n## 4. Morphometry Analysis")

    # 4.1 Calculate BMD (simplified as mean intensity in bone regions)
    bmd = np.mean(image_data[binary_image])
    log.append(f"Bone Mineral Density (BMD): {bmd:.2f} arbitrary units")

    # 4.2 Calculate BV (bone volume)
    bone_volume = np.sum(binary_image)
    total_volume = binary_image.size
    bv_tv_ratio = bone_volume / total_volume
    log.append(f"Bone Volume (BV): {bone_volume} voxels")
    log.append(f"BV/TV Ratio: {bv_tv_ratio:.4f}")

    # 4.3 Calculate trabecular thickness (Tb.Th)
    # Using distance transform method
    log.append("Calculating trabecular thickness (Tb.Th)")
    distance_map = ndimage.distance_transform_edt(binary_image)
    tb_th_mean = np.mean(distance_map[binary_image]) * 2  # Multiply by 2 for diameter
    log.append(f"Trabecular Thickness (Tb.Th): {tb_th_mean:.2f} voxels")

    # 4.4 Calculate trabecular separation (Tb.S)
    log.append("Calculating trabecular separation (Tb.S)")
    distance_map_inv = ndimage.distance_transform_edt(~binary_image)
    tb_s_mean = np.mean(distance_map_inv[~binary_image])
    log.append(f"Trabecular Separation (Tb.S): {tb_s_mean:.2f} voxels")

    # 4.5 Calculate trabecular number (Tb.N)
    # Simplified calculation: Tb.N = (BV/TV) / Tb.Th
    log.append("Calculating trabecular number (Tb.N)")
    tb_n = bv_tv_ratio / tb_th_mean if tb_th_mean > 0 else 0
    log.append(f"Trabecular Number (Tb.N): {tb_n:.4f} 1/voxel")

    # Step 5: Save results
    log.append("\n## 5. Saving Results")

    # Save binary segmentation as a sample slice
    middle_slice = binary_image.shape[0] // 2
    segmentation_file = os.path.join(output_dir, "bone_segmentation_slice.tif")
    io.imsave(segmentation_file, binary_image[middle_slice].astype(np.uint8) * 255)
    log.append(f"Sample segmentation slice saved to: {segmentation_file}")

    # Save numerical results as JSON
    results = {
        "BMD": float(bmd),
        "BV": int(bone_volume),
        "BV/TV": float(bv_tv_ratio),
        "Tb.Th": float(tb_th_mean),
        "Tb.S": float(tb_s_mean),
        "Tb.N": float(tb_n),
    }

    results_file = os.path.join(output_dir, "bone_morphometry_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    log.append(f"Numerical results saved to: {results_file}")

    # Summary of results
    log.append("\n## 6. Summary of Bone Morphometry Results")
    log.append(f"- Bone Mineral Density (BMD): {bmd:.2f} arbitrary units")
    log.append(f"- Bone Volume (BV): {bone_volume} voxels")
    log.append(f"- BV/TV Ratio: {bv_tv_ratio:.4f}")
    log.append(f"- Trabecular Thickness (Tb.Th): {tb_th_mean:.2f} voxels")
    log.append(f"- Trabecular Separation (Tb.S): {tb_s_mean:.2f} voxels")
    log.append(f"- Trabecular Number (Tb.N): {tb_n:.4f} 1/voxel")

    return "\n".join(log)


def predict_spatial_gene_expression_from_histology(
    image_path,
    gene_symbols,
    gene_embedding_source="scgpt",
    model_id="ratschlab/DeepSpotM",
    device="auto",
    output_dir="./output",
):
    """Predict spatial gene expression for a single H&E histology tile with DeepSpot-M.

    DeepSpot-M is a multimodal foundation model for transcriptome-wide virtual spatial
    transcriptomics from histology (https://doi.org/10.64898/2026.06.19.26356060, posted
    22 June 2026). It tokenises a 224x224 H&E tile with a LoRA-adapted pathology
    foundation backbone (Midnight); a cross-attention gene decoder lets each gene query
    attend to the patch tokens, and a gene router hypernetwork generates gene-specific
    output projections from frozen biological embeddings (Evo 2, Orthrus, ProtT5, scGPT,
    Apertus). Because genes are queryable embeddings rather than fixed output units, the
    model spans the protein-coding transcriptome, including genes that were not seen
    during training. Applied to TCGA it produced a virtual spatial transcriptomics atlas
    of 28,664 slides across 32 cancer types.

    IMPORTANT: the returned values are PREDICTED from tissue morphology, never measured.
    They are not a substitute for a spatial transcriptomics assay and must not be
    reported as experimental measurements. Research use only, not for clinical or
    diagnostic use. DeepSpot-M code is licensed PolyForm Noncommercial 1.0.0 and its
    weights CC-BY-NC-SA-4.0, i.e. non-commercial research use only.

    This tool depends on the optional `deepspotm` package and on gated model weights.
    Biomni installs and runs without them; if either is missing the tool returns a
    message explaining how to enable it rather than raising.

    Parameters
    ----------
    image_path : str
        Path to a single H&E histology tile that is exactly 224x224 pixels, cut from the
        slide at roughly 20x magnification (~0.5 microns per pixel). The physical scale
        cannot be recovered from the image file, so a tile extracted at a different
        magnification is silently out of distribution and degrades the predictions:
        re-extract the tile at ~0.5 microns per pixel rather than resizing it.
    gene_symbols : list[str]
        Explicit list of HGNC gene symbols to predict, e.g. ["EPCAM", "CD3D", "PTPRC"].
        Required: there is no default panel, so request only the genes relevant to the
        question instead of the full protein-coding transcriptome.
    gene_embedding_source : str, optional
        Frozen biological embedding used by the gene router to build the gene queries.
        One of "evo2", "orthrus", "prott5", "scgpt", "apertus" (default: "scgpt")
    model_id : str, optional
        Hugging Face repository holding the weights (default: "ratschlab/DeepSpotM")
    device : str, optional
        Torch device for inference: "auto" uses CUDA when available, otherwise pass an
        explicit device such as "cpu" or "cuda" (default: "auto")
    output_dir : str, optional
        Directory to save output files (default: "./output")

    Returns
    -------
    str
        Research log summarizing the prediction steps, the predicted log1p-CPM value for
        each requested gene, and the path of the saved CSV file

    """
    import csv
    import os
    from datetime import datetime

    valid_sources = ("evo2", "orthrus", "prott5", "scgpt", "apertus")

    # Step 0: validate the inputs before touching the model or the filesystem
    if isinstance(gene_symbols, str):
        gene_symbols = [gene_symbols]
    if not isinstance(gene_symbols, (list, tuple)) or len(gene_symbols) == 0:
        return (
            "Error: `gene_symbols` must be a non-empty list of HGNC gene symbols, for example "
            '["EPCAM", "CD3D", "PTPRC"]. There is no default gene panel: request only the genes '
            "that are relevant to the question."
        )
    gene_symbols = [str(gene).strip() for gene in gene_symbols]
    if any(not gene for gene in gene_symbols):
        return "Error: `gene_symbols` contains an empty gene symbol."

    if gene_embedding_source not in valid_sources:
        return (
            f"Error: `gene_embedding_source` must be one of {', '.join(valid_sources)}, got '{gene_embedding_source}'."
        )

    if not os.path.exists(image_path):
        return f"Error: File {image_path} not found"

    # Step 0b: optional dependencies are imported lazily so that Biomni stays installable
    # and testable without DeepSpot-M and without network access.
    try:
        from deepspotm import DeepSpotM
    except ImportError:
        return (
            "Error: the optional `deepspotm` package is not installed, so spatial gene expression "
            "cannot be predicted from histology. To enable this tool:\n"
            "  1. Install the package: pip install deepspotm\n"
            "  2. Request access to the gated weights at https://huggingface.co/ratschlab/DeepSpotM\n"
            "  3. Authenticate: huggingface-cli login\n"
            "The code (PolyForm Noncommercial 1.0.0) and the weights (CC-BY-NC-SA-4.0) are licensed "
            "for non-commercial research use only."
        )

    try:
        import torch
        from PIL import Image
    except ImportError as import_error:
        return f"Error: DeepSpot-M needs torch and pillow, but they could not be imported: {import_error}"

    os.makedirs(output_dir, exist_ok=True)

    log = f"# Virtual Spatial Transcriptomics with DeepSpot-M - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += f"Input tile: {image_path}\n"
    log += f"Requested genes ({len(gene_symbols)}): {', '.join(gene_symbols)}\n\n"

    # Step 1: load the tile and enforce the expected input geometry
    log += "## Step 1: Tile Loading and Validation\n"
    try:
        tile_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error: Could not read image {image_path}: {e}"

    width, height = tile_image.size
    if (width, height) != (224, 224):
        return (
            f"Error: DeepSpot-M expects a tile of exactly 224x224 pixels, but {image_path} is "
            f"{width}x{height} pixels. Tiles must be cut from the slide at roughly 20x magnification "
            "(~0.5 microns per pixel). Do not resize a tile that was taken at another magnification: "
            "the physical scale cannot be recovered from the image file and a mismatched scale "
            "silently degrades the predictions. Re-extract the tile at ~0.5 microns per pixel instead."
        )
    log += "- Loaded H&E tile of 224x224 pixels (expected magnification ~20x, ~0.5 microns per pixel)\n\n"

    # Step 2: load the gated weights
    log += "## Step 2: Model Loading\n"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log += f"- Inference device: {device}\n"

    try:
        model, image_processor = DeepSpotM.from_pretrained(model_id, source=gene_embedding_source)
    except Exception as e:
        return (
            f"Error: could not load the DeepSpot-M weights from '{model_id}' with gene embedding "
            f"source '{gene_embedding_source}': {e}\n"
            "These weights are gated: request access at https://huggingface.co/ratschlab/DeepSpotM "
            "and then authenticate with `huggingface-cli login`."
        )

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    log += f"- Loaded DeepSpot-M from '{model_id}' using the '{gene_embedding_source}' gene embeddings\n\n"

    # Step 3: predict the requested genes
    log += "## Step 3: Gene Expression Prediction\n"
    try:
        tile_tensor = image_processor(tile_image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model.predict_genes(tile_tensor, gene_symbols)
    except Exception as e:
        return f"Error during DeepSpot-M inference: {e}"

    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu()
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    values = list(predictions)
    # predict_genes returns one row per tile and this tool scores a single tile
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]

    if len(values) != len(gene_symbols):
        return (
            f"Error: DeepSpot-M returned {len(values)} values for {len(gene_symbols)} requested gene "
            "symbols. Check that every entry of `gene_symbols` is a valid HGNC gene symbol."
        )
    log += f"- Predicted log1p-CPM expression for {len(gene_symbols)} gene(s)\n\n"

    # Step 4: report and persist the predictions
    log += "## Step 4: Predicted Expression\n"
    log += "| Gene | Predicted log1p-CPM |\n"
    log += "| --- | --- |\n"
    for gene, value in zip(gene_symbols, values, strict=True):
        log += f"| {gene} | {float(value):.4f} |\n"

    csv_filename = os.path.join(
        output_dir,
        f"deepspotm_predicted_expression_{os.path.splitext(os.path.basename(image_path))[0]}.csv",
    )
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gene_symbol", "predicted_log1p_cpm"])
        for gene, value in zip(gene_symbols, values, strict=True):
            writer.writerow([gene, f"{float(value):.6f}"])
    log += f"\nPredictions saved to: {csv_filename}\n\n"

    log += "## Notes\n"
    log += "- These values are PREDICTED from H&E morphology by DeepSpot-M, not measured by a spatial "
    log += "transcriptomics assay, and must not be reported as experimental measurements.\n"
    log += "- Units are log1p-CPM.\n"
    log += "- Research use only, not for clinical or diagnostic use.\n"
    log += "- DeepSpot-M code is licensed PolyForm Noncommercial 1.0.0 and its weights CC-BY-NC-SA-4.0, "
    log += "i.e. non-commercial research use only.\n"
    log += "- Reference: Nonchev K, Dawo S, Silina K, Koelzer VH, Raetsch G. DeepSpot-M: a multimodal "
    log += "foundation model for transcriptome-wide virtual spatial transcriptomics from histology. "
    log += "https://doi.org/10.64898/2026.06.19.26356060\n"

    return log
