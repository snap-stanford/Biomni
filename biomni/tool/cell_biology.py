def quantify_cell_cycle_phases_from_microscopy(image_paths: list[str], output_dir: str = "./results") -> str:
    """Quantify the percentage of cells in each cell cycle phase using Calcofluor white stained microscopy images.

    This function processes microscopy images where cell walls/septa are stained with Calcofluor white,
    segments individual cells, extracts features, and classifies each cell into G1, S, or G2/M phase.

    Parameters
    ----------
    image_paths : list of str
        List of file paths to microscopy images of cells stained with Calcofluor white
    output_dir : str, optional
        Directory to save results (default: './results')

    Returns
    -------
    str
        Research log summarizing the analysis process and results

    """
    import os

    import numpy as np
    import pandas as pd
    from skimage import filters, io, measure, morphology

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log
    log = "# Cell Cycle Phase Quantification Analysis Log\n\n"
    log += f"Processing {len(image_paths)} microscopy images with Calcofluor white staining.\n\n"

    # Initialize data structures to store results
    all_cells = []
    cell_features = []

    # Step 1: Process images and segment cells
    log += "## Image Processing and Cell Segmentation\n\n"

    for i, img_path in enumerate(image_paths):
        log += f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}\n"

        # Load image
        img = io.imread(img_path)

        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = np.mean(img, axis=2).astype(np.uint8)

        # Preprocessing
        filtered_img = filters.gaussian(img, sigma=1.0)

        # Thresholding to identify cell walls
        threshold = filters.threshold_otsu(filtered_img)
        binary = filtered_img > threshold

        # Clean up binary image
        binary = morphology.remove_small_objects(binary, min_size=30)
        binary = morphology.binary_closing(binary)

        # Segment cells
        labeled_cells = measure.label(binary)
        cell_props = measure.regionprops(labeled_cells, intensity_image=img)

        log += f"  - Found {len(cell_props)} cells\n"
        all_cells.extend(cell_props)

        # Extract features for each cell
        for cell in cell_props:
            # Basic morphological features
            area = cell.area
            perimeter = cell.perimeter
            eccentricity = cell.eccentricity
            mean_intensity = cell.mean_intensity

            # Cell shape features
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            # Check for septation (indicative of S or G2/M phase)
            # Calcofluor white stains the septum more intensely
            has_septum = False
            if area > 50:  # Only check larger cells
                intensity_profile = cell.intensity_image[cell.intensity_image > 0]
                if intensity_profile.size > 0:
                    intensity_std = np.std(intensity_profile)
                    intensity_max = np.max(intensity_profile)
                    # High standard deviation and max intensity suggests septum presence
                    has_septum = intensity_std > 0.2 * np.mean(intensity_profile) and intensity_max > 1.5 * np.mean(
                        intensity_profile
                    )

            # Store features
            cell_features.append(
                [
                    area,
                    perimeter,
                    eccentricity,
                    mean_intensity,
                    circularity,
                    has_septum,
                    i,  # include image index
                ]
            )

    log += f"\nTotal cells detected across all images: {len(all_cells)}\n\n"

    # Step 2: Feature engineering and classification
    log += "## Cell Cycle Phase Classification\n\n"

    # Convert to numpy array for easier processing
    X = np.array(cell_features)

    # Simple rule-based classification
    # This is a simplified approach - in practice, you would train a classifier on labeled data
    # G1: Smaller cells, no septum
    # S: Medium-sized cells, early septum formation
    # G2/M: Larger cells, clear septum, often elongated

    # Define classification rules based on features
    phases = []
    for i in range(X.shape[0]):
        area = X[i, 0]
        perimeter = X[i, 1]
        eccentricity = X[i, 2]
        has_septum = X[i, 5]

        if has_septum and area > np.median(X[:, 0]) * 1.2:
            # Larger cells with septum are likely in G2/M
            phases.append("G2/M")
        elif has_septum or (area > np.median(X[:, 0]) and eccentricity > 0.5):
            # Cells with septum or larger elongated cells are likely in S
            phases.append("S")
        else:
            # Smaller cells without septum are likely in G1
            phases.append("G1")

    # Calculate percentages
    unique_phases, counts = np.unique(phases, return_counts=True)
    percentages = (counts / len(phases)) * 100

    # Create results table
    results_df = pd.DataFrame({"Phase": unique_phases, "Count": counts, "Percentage": percentages})

    # Save results
    results_path = os.path.join(output_dir, "cell_cycle_phases.csv")
    results_df.to_csv(results_path, index=False)

    # Log results
    log += "### Cell Cycle Phase Distribution\n\n"
    for phase, count, percentage in zip(unique_phases, counts, percentages, strict=False):
        log += f"- {phase}: {count} cells ({percentage:.2f}%)\n"

    log += f"\nResults saved to: {results_path}\n"
    log += "\nNote: This analysis uses simplified morphological features for classification. "
    log += (
        "For more accurate results, a supervised machine learning approach with labeled training data is recommended.\n"
    )

    return log


def quantify_and_cluster_cell_motility(
    image_sequence_path: str, output_dir: str = "./results", num_clusters: int = 3
) -> str:
    """Quantify cell motility features from time-lapse microscopy images and cluster cells based on motility patterns.

    Parameters
    ----------
    image_sequence_path : str
        Path to directory containing time-lapse microscopy images in sequential order
    output_dir : str
        Directory to save output files (default: "./results")
    num_clusters : int
        Number of motility pattern clusters to identify (default: 3)

    Returns
    -------
    str
        Research log summarizing the analysis process and results

    """
    import os

    import cv2
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load image sequence
    image_files = sorted(
        [f for f in os.listdir(image_sequence_path) if f.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    )

    if len(image_files) < 2:
        return "Error: Insufficient images found. At least 2 time points are required."

    # Step 2: Initialize cell tracking
    first_image = cv2.imread(os.path.join(image_sequence_path, image_files[0]), cv2.IMREAD_GRAYSCALE)

    # Simple cell detection using thresholding and contour detection
    _, binary = cv2.threshold(first_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract cell centroids from first frame
    cells = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 50:  # Filter small noise
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cells.append({"id": i, "positions": [(cx, cy)], "frame_indices": [0]})

    # Step 3: Track cells across frames
    for frame_idx, img_file in enumerate(image_files[1:], 1):
        img = cv2.imread(os.path.join(image_sequence_path, img_file), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get current frame centroids
        current_centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centroids.append((cx, cy))

        # Match cells with nearest centroids in current frame
        for cell in cells:
            if len(cell["positions"]) == frame_idx:  # Only track cells found in previous frame
                prev_pos = cell["positions"][-1]

                if current_centroids:
                    # Calculate distances to all current centroids
                    distances = [
                        np.sqrt((prev_pos[0] - c[0]) ** 2 + (prev_pos[1] - c[1]) ** 2) for c in current_centroids
                    ]
                    min_idx = np.argmin(distances)

                    # Only match if distance is below threshold (prevents incorrect matching)
                    if distances[min_idx] < 50:  # Threshold distance in pixels
                        cell["positions"].append(current_centroids[min_idx])
                        cell["frame_indices"].append(frame_idx)
                        current_centroids.pop(min_idx)  # Remove matched centroid

    # Step 4: Calculate motility features for each cell
    cell_features = []

    for cell in cells:
        # Only analyze cells tracked for at least 3 frames
        if len(cell["positions"]) < 3:
            continue

        # Calculate displacements between consecutive positions
        displacements = []
        for i in range(1, len(cell["positions"])):
            prev_pos = cell["positions"][i - 1]
            curr_pos = cell["positions"][i]
            displacement = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
            displacements.append(displacement)

        # Calculate speed (pixels per frame)
        avg_speed = np.mean(displacements)

        # Calculate directionality (displacement from start to end / total path length)
        start_pos = cell["positions"][0]
        end_pos = cell["positions"][-1]
        net_displacement = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
        total_path_length = np.sum(displacements)
        directionality = net_displacement / total_path_length if total_path_length > 0 else 0

        # Calculate mean squared displacement
        msd = np.mean(
            [
                np.sqrt((cell["positions"][i][0] - start_pos[0]) ** 2 + (cell["positions"][i][1] - start_pos[1]) ** 2)
                for i in range(1, len(cell["positions"]))
            ]
        )

        # Calculate track duration
        duration = cell["frame_indices"][-1] - cell["frame_indices"][0]

        cell_features.append(
            {
                "cell_id": cell["id"],
                "avg_speed": avg_speed,
                "directionality": directionality,
                "msd": msd,
                "track_duration": duration,
                "track_length": len(cell["positions"]),
            }
        )

    # Step 5: Cluster cells based on motility features
    if len(cell_features) < num_clusters:
        return f"Error: Not enough cells ({len(cell_features)}) to form {num_clusters} clusters. Try reducing num_clusters."

    # Create feature matrix for clustering
    feature_df = pd.DataFrame(cell_features)
    feature_matrix = feature_df[["avg_speed", "directionality", "msd"]].values

    # Normalize features
    feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / np.std(feature_matrix, axis=0)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(feature_matrix)

    # Add cluster assignments to dataframe
    feature_df["cluster"] = clusters

    # Save results to CSV
    results_file = os.path.join(output_dir, "cell_motility_features.csv")
    feature_df.to_csv(results_file, index=False)

    # Calculate cluster statistics
    cluster_stats = feature_df.groupby("cluster").agg(
        {
            "avg_speed": ["mean", "std"],
            "directionality": ["mean", "std"],
            "msd": ["mean", "std"],
            "cell_id": "count",
        }
    )

    # Save cluster statistics
    stats_file = os.path.join(output_dir, "cluster_statistics.csv")
    cluster_stats.to_csv(stats_file)

    # Create research log
    log = f"""
Cell Motility Quantification and Clustering Analysis
===================================================

Analysis Steps:
1. Loaded {len(image_files)} time-lapse microscopy images
2. Detected and tracked {len(cells)} initial cells across time frames
3. Calculated motility features for {len(cell_features)} cells with sufficient tracking data
4. Performed k-means clustering to identify {num_clusters} distinct motility patterns

Results Summary:
- Detected {len(cells)} cells in the first frame
- Successfully tracked {len(cell_features)} cells across multiple frames
- Clustered cells into {num_clusters} motility pattern groups

Cluster Statistics:
{cluster_stats.to_string()}

Output Files:
- Cell motility features saved to: {results_file}
- Cluster statistics saved to: {stats_file}

Analysis Notes:
- Features used for clustering: average speed, directionality, and mean squared displacement
- Cells were required to be tracked for at least 3 frames to be included in the analysis
"""

    # Save research log
    log_file = os.path.join(output_dir, "research_log.txt")
    with open(log_file, "w") as f:
        f.write(log)

    return log


def perform_facs_cell_sorting(
    cell_suspension_data,
    fluorescence_parameter,
    threshold_min=None,
    threshold_max=None,
    output_file="sorted_cells.csv",
):
    """Performs Fluorescence-Activated Cell Sorting (FACS) to enrich cell populations based on fluorescence characteristics.

    Parameters
    ----------
    cell_suspension_data : str
        Path to the FCS file containing flow cytometry data
    fluorescence_parameter : str
        The fluorescence parameter to use for sorting (e.g., 'GFP', 'FITC', 'PE')
    threshold_min : float, optional
        Minimum threshold for the fluorescence parameter. Cells below this value will be excluded
    threshold_max : float, optional
        Maximum threshold for the fluorescence parameter. Cells above this value will be excluded
    output_file : str, optional
        Filename to save the sorted cell population data

    Returns
    -------
    str
        Research log detailing the FACS cell sorting process

    """
    import os

    import pandas as pd

    # Initialize research log
    log = "# FACS-based Cell Sorting and Enrichment Research Log\n\n"

    try:
        # Load cell suspension data
        log += "## Loading Cell Suspension Data\n"
        if isinstance(cell_suspension_data, str):
            # If data is provided as a file path
            if cell_suspension_data.lower().endswith(".fcs"):
                try:
                    import flowkit as fk

                    fcs_data = fk.Sample(cell_suspension_data)
                    cell_df = pd.DataFrame(fcs_data.get_dataframe())
                    log += f"Successfully loaded FCS file containing {len(cell_df)} cells\n"
                except ImportError:
                    # Fallback if flowkit is not available
                    log += "FlowKit not available, attempting to load as CSV\n"
                    cell_df = pd.read_csv(cell_suspension_data)
            else:
                # Assume CSV or other tabular format
                cell_df = pd.read_csv(cell_suspension_data)
                log += f"Loaded data file containing {len(cell_df)} cells\n"
        else:
            # Assume data is already a DataFrame
            cell_df = cell_suspension_data.copy()
            log += f"Using provided DataFrame containing {len(cell_df)} cells\n"

        # Validate fluorescence parameter exists in data
        if fluorescence_parameter not in cell_df.columns:
            raise ValueError(f"Fluorescence parameter '{fluorescence_parameter}' not found in data")

        # Apply gating strategy based on fluorescence thresholds
        log += f"\n## Applying Gating Strategy for {fluorescence_parameter}\n"
        original_count = len(cell_df)

        # Filter cells based on min/max thresholds
        if threshold_min is not None:
            cell_df = cell_df[cell_df[fluorescence_parameter] >= threshold_min]
            log += f"Applied minimum threshold of {threshold_min}: {len(cell_df)} cells remaining\n"

        if threshold_max is not None:
            cell_df = cell_df[cell_df[fluorescence_parameter] <= threshold_max]
            log += f"Applied maximum threshold of {threshold_max}: {len(cell_df)} cells remaining\n"

        # Calculate enrichment statistics
        enrichment_percent = (len(cell_df) / original_count) * 100 if original_count > 0 else 0
        log += "\n## Enrichment Results\n"
        log += f"Original population: {original_count} cells\n"
        log += f"Sorted population: {len(cell_df)} cells\n"
        log += f"Enrichment percentage: {enrichment_percent:.2f}%\n"

        # Calculate statistics of sorted population
        mean_fluorescence = cell_df[fluorescence_parameter].mean()
        median_fluorescence = cell_df[fluorescence_parameter].median()

        log += f"Mean {fluorescence_parameter}: {mean_fluorescence:.2f}\n"
        log += f"Median {fluorescence_parameter}: {median_fluorescence:.2f}\n"

        # Save sorted population to file
        cell_df.to_csv(output_file, index=False)
        log += "\n## Output\n"
        log += f"Sorted cell population saved to: {os.path.abspath(output_file)}\n"

        return log

    except Exception as e:
        log += "\n## Error\n"
        log += f"An error occurred during cell sorting: {str(e)}\n"
        return log


def analyze_flow_cytometry_immunophenotyping(
    fcs_file_path, gating_strategy, compensation_matrix=None, output_dir="./results"
):
    """Analyze flow cytometry data to identify and quantify specific cell populations based on surface markers.

    Parameters
    ----------
    fcs_file_path : str
        Path to the FCS file containing flow cytometry data
    gating_strategy : dict
        Dictionary defining the gating strategy. Each key is a population name, and each value is a list of tuples
        (marker, operator, threshold). For example: {'HSCs': [('Lin', '<', 100), ('Sca1', '>', 1000), ...]}
    compensation_matrix : numpy.ndarray, optional
        Spillover/compensation matrix to correct for fluorescence overlap
    output_dir : str, optional
        Directory to save the results

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os

    import pandas as pd
    from FlowCytometryTools import FCMeasurement

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the FCS file
    sample = FCMeasurement(ID="Sample", datafile=fcs_file_path)

    # Apply compensation if provided
    if compensation_matrix is not None:
        sample = sample.compensate(compensation_matrix)

    # Initialize log
    log = "Flow Cytometry Analysis Log\n"
    log += "==========================\n"
    log += f"File analyzed: {os.path.basename(fcs_file_path)}\n"
    log += f"Total events: {len(sample)}\n\n"

    # Create a dictionary to store population counts
    populations = {}

    # Apply gating strategy to identify cell populations
    for pop_name, gates in gating_strategy.items():
        # Start with all events
        current_population = sample.copy()

        log += f"Identifying {pop_name} population:\n"

        # Apply each gate sequentially
        for marker, operator, threshold in gates:
            initial_count = len(current_population)

            if operator == ">":
                current_population = current_population.gate(f"{marker} > {threshold}")
            elif operator == "<":
                current_population = current_population.gate(f"{marker} < {threshold}")
            elif operator == "between":
                # For 'between', threshold should be a tuple (lower, upper)
                lower, upper = threshold
                current_population = current_population.gate(f"{marker} > {lower} and {marker} < {upper}")

            final_count = len(current_population)
            log += f"  Gate {marker} {operator} {threshold}: {initial_count} → {final_count} events\n"

        # Store the final population
        populations[pop_name] = current_population

        # Calculate percentage of original sample
        percentage = (len(current_population) / len(sample)) * 100
        log += f"  Final {pop_name} count: {len(current_population)} events ({percentage:.2f}% of total)\n\n"

    # Create a summary dataframe
    summary_data = {"Population": [], "Count": [], "Percentage": []}

    for pop_name, population in populations.items():
        summary_data["Population"].append(pop_name)
        summary_data["Count"].append(len(population))
        summary_data["Percentage"].append((len(population) / len(sample)) * 100)

    summary_df = pd.DataFrame(summary_data)

    # Save the summary to a CSV file
    summary_file = os.path.join(output_dir, "population_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    log += "Summary of identified populations:\n"
    for _, row in summary_df.iterrows():
        log += f"  {row['Population']}: {row['Count']} events ({row['Percentage']:.2f}%)\n"

    log += f"\nDetailed results saved to: {summary_file}\n"

    return log


def analyze_mitochondrial_morphology_and_potential(
    morphology_image_path: str, potential_image_path: str, output_dir: str = "./output"
) -> str:
    """Quantifies metrics of mitochondrial morphology and membrane potential from fluorescence microscopy images.

    Parameters
    ----------
    morphology_image_path : str
        Path to the fluorescence microscopy image showing mitochondrial morphology (e.g., MTS-GFP)
    potential_image_path : str
        Path to the fluorescence microscopy image showing mitochondrial membrane potential (e.g., TMRE staining)
    output_dir : str, optional
        Directory to save output files, default is "./output"

    Returns
    -------
    str
        A research log summarizing the analysis steps and results

    """
    import datetime
    import os

    import cv2
    import numpy as np
    from scipy import ndimage
    from skimage import filters, io, morphology, util

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    log = []
    log.append(f"Mitochondrial Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 50)

    # Load images
    try:
        morph_img = io.imread(morphology_image_path)
        pot_img = io.imread(potential_image_path)

        log.append(f"Successfully loaded morphology image: {morphology_image_path}")
        log.append(f"Successfully loaded potential image: {potential_image_path}")
        log.append(f"Morphology image shape: {morph_img.shape}")
        log.append(f"Potential image shape: {pot_img.shape}")
    except Exception as e:
        return f"Error loading images: {str(e)}"

    # Convert to grayscale if needed
    if len(morph_img.shape) > 2:
        morph_img = cv2.cvtColor(morph_img, cv2.COLOR_RGB2GRAY)
        log.append("Converted morphology image to grayscale")
    if len(pot_img.shape) > 2:
        pot_img = cv2.cvtColor(pot_img, cv2.COLOR_RGB2GRAY)
        log.append("Converted potential image to grayscale")

    # Normalize images
    morph_img = util.img_as_float(morph_img)
    pot_img = util.img_as_float(pot_img)

    log.append("\nMorphology Analysis:")
    log.append("-" * 30)

    # Denoise morphology image
    morph_img_denoised = filters.gaussian(morph_img, sigma=1.0)

    # Threshold to segment mitochondria
    threshold_value = filters.threshold_otsu(morph_img_denoised)
    binary_img = morph_img_denoised > threshold_value

    # Remove small objects
    binary_img = morphology.remove_small_objects(binary_img, min_size=20)

    # Save binary image
    binary_path = os.path.join(output_dir, "binary_mitochondria.png")
    io.imsave(binary_path, util.img_as_ubyte(binary_img))
    log.append(f"Binary segmentation saved to: {binary_path}")

    # Skeletonize for network analysis
    skeleton = morphology.skeletonize(binary_img)
    skeleton_path = os.path.join(output_dir, "skeleton.png")
    io.imsave(skeleton_path, util.img_as_ubyte(skeleton))
    log.append(f"Skeleton image saved to: {skeleton_path}")

    # Calculate morphology metrics
    # 1. Count branches and junctions
    # Label the skeleton
    labeled_skeleton, num_branches = ndimage.label(skeleton)

    # Find branch points (pixels with more than 2 neighbors)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Don't count the center pixel
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel)
    junction_points = (neighbor_count > 2) & skeleton
    num_junctions = np.sum(junction_points)

    # 2. Calculate fragmentation metrics
    labeled_objects, num_objects = ndimage.label(binary_img)
    object_sizes = ndimage.sum(binary_img, labeled_objects, range(1, num_objects + 1))
    mean_size = np.mean(object_sizes) if len(object_sizes) > 0 else 0

    # Calculate network connectivity (ratio of junctions to branches)
    connectivity = num_junctions / num_branches if num_branches > 0 else 0

    # Calculate fragmentation index (inverse of mean object size, normalized)
    fragmentation = 1 / (mean_size / np.max(object_sizes)) if mean_size > 0 and np.max(object_sizes) > 0 else 0

    log.append(f"Number of mitochondrial fragments: {num_objects}")
    log.append(f"Number of branches: {num_branches}")
    log.append(f"Number of junction points: {num_junctions}")
    log.append(f"Network connectivity index: {connectivity:.4f}")
    log.append(f"Fragmentation index: {fragmentation:.4f}")
    log.append(f"Mean fragment size: {mean_size:.2f} pixels")

    # Membrane potential analysis
    log.append("\nMembrane Potential Analysis:")
    log.append("-" * 30)

    # Create mask from morphology image to analyze only mitochondrial regions
    mito_mask = binary_img

    # Calculate potential metrics within mitochondrial regions
    pot_intensity_raw = np.mean(pot_img)
    pot_intensity_in_mito = np.mean(pot_img[mito_mask]) if np.sum(mito_mask) > 0 else 0

    # Calculate potential heterogeneity (standard deviation of intensity)
    pot_heterogeneity = np.std(pot_img[mito_mask]) if np.sum(mito_mask) > 0 else 0

    # Calculate potential distribution
    if np.sum(mito_mask) > 0:
        percentiles = [10, 25, 50, 75, 90]
        pot_percentiles = np.percentile(pot_img[mito_mask], percentiles)
        percentile_str = ", ".join([f"{p}th: {v:.4f}" for p, v in zip(percentiles, pot_percentiles, strict=False)])
    else:
        percentile_str = "No mitochondrial regions detected"

    log.append(f"Overall mean TMRE intensity: {pot_intensity_raw:.4f}")
    log.append(f"Mean TMRE intensity in mitochondria: {pot_intensity_in_mito:.4f}")
    log.append(f"TMRE intensity heterogeneity (std): {pot_heterogeneity:.4f}")
    log.append(f"TMRE intensity percentiles: {percentile_str}")

    # Create a combined results image
    # Overlay potential intensity on morphology
    overlay = np.zeros((morph_img.shape[0], morph_img.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = morph_img  # Red channel - morphology
    overlay[..., 2] = pot_img  # Blue channel - potential
    overlay = np.clip(overlay, 0, 1)

    overlay_path = os.path.join(output_dir, "morphology_potential_overlay.png")
    io.imsave(overlay_path, util.img_as_ubyte(overlay))
    log.append(f"Overlay image saved to: {overlay_path}")

    # Save quantitative results to CSV
    import csv

    results_path = os.path.join(output_dir, "mitochondrial_analysis_results.csv")
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Number of fragments", num_objects])
        writer.writerow(["Number of branches", num_branches])
        writer.writerow(["Number of junctions", num_junctions])
        writer.writerow(["Network connectivity", connectivity])
        writer.writerow(["Fragmentation index", fragmentation])
        writer.writerow(["Mean fragment size", mean_size])
        writer.writerow(["Mean TMRE intensity", pot_intensity_in_mito])
        writer.writerow(["TMRE heterogeneity", pot_heterogeneity])

    log.append(f"Quantitative results saved to: {results_path}")
    log.append("\nAnalysis complete.")

    return "\n".join(log)
