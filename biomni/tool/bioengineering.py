def analyze_cell_migration_metrics(
    image_sequence_path,
    pixel_size_um=1.0,
    time_interval_min=1.0,
    min_track_length=10,
    output_dir="./",
):
    """Analyze cell migration metrics from time-lapse microscopy images.

    Parameters
    ----------
    image_sequence_path : str
        Path to the directory containing time-lapse images or path to a multi-frame TIFF file
    pixel_size_um : float
        Conversion factor from pixels to micrometers (default: 1.0)
    time_interval_min : float
        Time interval between consecutive frames in minutes (default: 1.0)
    min_track_length : int
        Minimum number of frames a cell must be tracked to be included in analysis (default: 10)
    output_dir : str
        Directory to save output files (default: "./")

    Returns
    -------
    str
        Research log summarizing the cell migration analysis process and results

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import trackpy as tp
    from skimage import io

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image sequence
    if os.path.isdir(image_sequence_path):
        # Load from directory of images
        image_files = sorted(
            [f for f in os.listdir(image_sequence_path) if f.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
        )
        frames = [io.imread(os.path.join(image_sequence_path, f)) for f in image_files]
    else:
        # Load multi-frame TIFF
        frames = io.imread(image_sequence_path)
        if frames.ndim == 3:  # Check if it's a time series
            pass
        else:
            return "Error: Input is not a valid time-lapse image sequence"

    # Step 1: Detect cells in each frame
    features_list = []
    for i, frame in enumerate(frames):
        # Detect cells (particles) in the frame
        features = tp.locate(frame, diameter=15, minmass=100)
        if features is not None and not features.empty:
            features["frame"] = i
            features_list.append(features)

    if not features_list:
        return "Error: No cells detected in the image sequence"

    # Combine all features
    all_features = pd.concat(features_list)

    # Save raw detections to output directory
    raw_detections_file = os.path.join(output_dir, "raw_cell_detections.csv")
    all_features.to_csv(raw_detections_file, index=False)

    # Step 2: Link features into cell trajectories
    trajectories = tp.link_df(all_features, search_range=10, memory=3)

    # Save linked trajectories before filtering
    # Reset index to ensure frame is a column, not an index level
    trajectories_reset = trajectories.reset_index(drop=True)
    all_trajectories_file = os.path.join(output_dir, "all_trajectories.csv")
    trajectories_reset.to_csv(all_trajectories_file, index=False)

    # Step 3: Filter trajectories to get only the ones that appear in enough frames
    trajectories = tp.filter_stubs(trajectories, threshold=min_track_length)

    if trajectories.empty:
        return "No complete cell tracks found. Try adjusting parameters."

    # Save filtered trajectories
    filtered_trajectories_file = os.path.join(output_dir, "filtered_trajectories.csv")
    # Reset index again to be safe
    trajectories = trajectories.reset_index(drop=True)
    trajectories.to_csv(filtered_trajectories_file, index=False)

    # Step 4: Calculate migration metrics for each cell
    cell_ids = trajectories["particle"].unique()
    metrics = []

    for cell_id in cell_ids:
        cell_track = trajectories[trajectories["particle"] == cell_id].sort_values("frame")

        # Convert pixel positions to micrometers
        cell_track["x_um"] = cell_track["x"] * pixel_size_um
        cell_track["y_um"] = cell_track["y"] * pixel_size_um

        # Calculate displacements between consecutive frames
        dx = np.diff(cell_track["x_um"])
        dy = np.diff(cell_track["y_um"])

        # Calculate step distances
        step_distances = np.sqrt(dx**2 + dy**2)

        # Calculate total path length
        path_length = np.sum(step_distances)

        # Calculate net displacement (straight-line distance from start to end)
        start_x, start_y = cell_track.iloc[0][["x_um", "y_um"]]
        end_x, end_y = cell_track.iloc[-1][["x_um", "y_um"]]
        net_displacement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

        # Calculate directionality ratio (net displacement / path length)
        directionality = net_displacement / path_length if path_length > 0 else 0

        # Calculate speed (μm/min)
        time_tracked = (cell_track["frame"].max() - cell_track["frame"].min()) * time_interval_min
        speed = path_length / time_tracked if time_tracked > 0 else 0

        metrics.append(
            {
                "cell_id": cell_id,
                "frames_tracked": len(cell_track),
                "speed_um_per_min": speed,
                "directionality": directionality,
                "displacement_um": net_displacement,
                "path_length_um": path_length,
            }
        )

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, "cell_migration_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)

    # Calculate summary statistics
    summary = {
        "num_cells_tracked": len(metrics_df),
        "avg_speed": metrics_df["speed_um_per_min"].mean(),
        "std_speed": metrics_df["speed_um_per_min"].std(),
        "avg_directionality": metrics_df["directionality"].mean(),
        "std_directionality": metrics_df["directionality"].std(),
        "avg_displacement": metrics_df["displacement_um"].mean(),
        "std_displacement": metrics_df["displacement_um"].std(),
    }

    # Save summary statistics
    summary_file = os.path.join(output_dir, "migration_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_file, index=False)

    # Save trajectories visualization
    fig, ax = plt.figure(figsize=(8, 8)), plt.gca()
    tp.plot_traj(trajectories, ax=ax)
    plt.title("Cell Migration Trajectories")
    plt.xlabel("x position (pixels)")
    plt.ylabel("y position (pixels)")

    trajectories_file = os.path.join(output_dir, "cell_trajectories.png")
    plt.savefig(trajectories_file)
    plt.close()

    # Create a rose plot to show migration directionality
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Calculate angles for each cell (end position relative to start)
    angles = []
    for cell_id in cell_ids:
        cell_track = trajectories[trajectories["particle"] == cell_id].sort_values("frame")
        start_x, start_y = cell_track.iloc[0][["x", "y"]]
        end_x, end_y = cell_track.iloc[-1][["x", "y"]]
        dx, dy = end_x - start_x, end_y - start_y
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # Plot the histogram
    bins = np.linspace(-np.pi, np.pi, 16)
    ax.hist(angles, bins=bins)
    ax.set_title("Cell Migration Directionality")

    # Save the rose plot
    rose_plot_file = os.path.join(output_dir, "rose_plot.png")
    plt.savefig(rose_plot_file)
    plt.close()

    # Create a displacement plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics_df)), metrics_df["displacement_um"])
    plt.xlabel("Cell ID")
    plt.ylabel("Displacement (μm)")
    plt.title("Cell Displacements")

    # Save the displacement plot
    displacement_plot_file = os.path.join(output_dir, "track_displacement_plot.png")
    plt.savefig(displacement_plot_file)
    plt.close()

    # Generate research log
    log = f"""
Cell Migration Analysis Research Log:

1. Analyzed time-lapse sequence with {len(frames)} frames
2. Detected and tracked {len(cell_ids)} cells that persisted for at least {min_track_length} frames
3. Calculated key migration metrics:
   - Average speed: {summary["avg_speed"]:.2f} ± {summary["std_speed"]:.2f} μm/min
   - Average directionality ratio: {summary["avg_directionality"]:.2f} ± {summary["std_directionality"]:.2f}
   - Average displacement: {summary["avg_displacement"]:.2f} ± {summary["std_displacement"]:.2f} μm

4. Files saved:
   - Raw cell detections: {raw_detections_file}
   - All cell trajectories: {all_trajectories_file}
   - Filtered trajectories: {filtered_trajectories_file}
   - Detailed cell metrics: {metrics_file}
   - Summary statistics: {summary_file}
   - Cell trajectories visualization: {trajectories_file}
   - Direction rose plot: {rose_plot_file}
   - Cell displacement plot: {displacement_plot_file}

Note: Analysis used pixel size of {pixel_size_um} μm and time interval of {time_interval_min} min between frames.
    """

    return log.strip()


def perform_crispr_cas9_genome_editing(guide_rna_sequences, target_genomic_loci, cell_tissue_type):
    """Simulates CRISPR-Cas9 genome editing process including guide RNA design, delivery, and analysis.

    Parameters
    ----------
    guide_rna_sequences : list of str
        List of guide RNA sequences (20 nucleotides each) targeting the genomic region of interest

    target_genomic_loci : str
        Target genomic sequence to be edited (should be longer than guide RNA and contain the target sites)

    cell_tissue_type : str
        Type of cell or tissue being edited (affects delivery efficiency and editing outcomes)

    Returns
    -------
    str
        Research log detailing the CRISPR-Cas9 editing process, including steps taken and results

    """
    import os
    import random
    from datetime import datetime

    # Initialize research log
    log = "CRISPR-Cas9 Genome Editing Research Log\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Cell/Tissue Type: {cell_tissue_type}\n\n"

    # Step 1: Validate guide RNA sequences
    log += "STEP 1: Guide RNA Validation\n"
    valid_guides = []

    for i, guide in enumerate(guide_rna_sequences):
        if len(guide) != 20:
            log += f"  Guide {i + 1}: INVALID - Guide RNA must be 20 nucleotides (current length: {len(guide)})\n"
            continue

        if not all(n in "ATGC" for n in guide.upper()):
            log += f"  Guide {i + 1}: INVALID - Guide RNA contains invalid nucleotides\n"
            continue

        # Calculate GC content (affects guide efficiency)
        gc_content = (guide.upper().count("G") + guide.upper().count("C")) / len(guide) * 100
        efficiency_score = 0

        if 40 <= gc_content <= 60:
            efficiency_score += 1
            gc_quality = "Optimal"
        else:
            gc_quality = "Suboptimal"

        log += f"  Guide {i + 1}: VALID - {guide} (GC content: {gc_content:.1f}% - {gc_quality})\n"
        valid_guides.append((guide, efficiency_score))

    if not valid_guides:
        log += "\nNo valid guide RNAs found. Genome editing cannot proceed.\n"
        return log

    # Step 2: Target site identification
    log += "\nSTEP 2: Target Site Identification\n"

    target_seq = target_genomic_loci.upper()
    target_matches = []

    for i, (guide, score) in enumerate(valid_guides):
        # Find guide RNA target in genomic sequence (including PAM site NGG)
        guide.upper() + "NGG"

        # Check if guide sequence is in target (simplified)
        if guide.upper() in target_seq:
            position = target_seq.find(guide.upper())
            # Check if there's a PAM sequence (NGG) after the guide
            if position + len(guide) + 2 <= len(target_seq):
                potential_pam = target_seq[position + len(guide) : position + len(guide) + 3]
                if potential_pam[1:3] == "GG":
                    pam_quality = "Found"
                    score += 2
                else:
                    pam_quality = "Not found"
            else:
                pam_quality = "Out of bounds"

            log += f"  Guide {i + 1}: Found at position {position} (PAM: {pam_quality})\n"
            target_matches.append((guide, position, score))
        else:
            log += f"  Guide {i + 1}: No match found in target sequence\n"

    if not target_matches:
        log += "\nNo matching target sites found. Genome editing cannot proceed.\n"
        return log

    # Step 3: Simulate CRISPR-Cas9 delivery
    log += "\nSTEP 3: CRISPR-Cas9 Delivery Simulation\n"

    # Cell-specific delivery efficiencies (simplified model)
    delivery_efficiencies = {
        "hek293": 0.85,
        "hela": 0.75,
        "ipsc": 0.60,
        "primary_neuron": 0.40,
        "hematopoietic_stem_cell": 0.55,
        "mouse_embryo": 0.70,
        "plant_cell": 0.30,
    }

    # Get delivery efficiency based on cell type (default to 0.5 if unknown)
    cell_type_key = cell_tissue_type.lower().replace(" ", "_")
    delivery_efficiency = delivery_efficiencies.get(cell_type_key, 0.5)

    log += f"  Delivery method: Lipofection for {cell_tissue_type}\n"
    log += f"  Estimated delivery efficiency: {delivery_efficiency * 100:.1f}%\n"

    # Step 4: Simulate genome editing
    log += "\nSTEP 4: Genome Editing Simulation\n"

    # Select best guide based on score
    best_guide, best_position, best_score = sorted(target_matches, key=lambda x: x[2], reverse=True)[0]

    log += f"  Selected guide RNA: {best_guide} (highest efficiency score)\n"
    log += f"  Target position: {best_position} to {best_position + len(best_guide) - 1}\n"

    # Simulate editing outcome
    edit_success_rate = delivery_efficiency * (0.5 + (best_score * 0.1))  # Between 50-90% based on guide quality

    # Cut site (typically 3 bases upstream of PAM)
    cut_position = best_position + len(best_guide) - 3
    log += f"  Predicted cut site: Between positions {cut_position} and {cut_position + 1}\n"

    # Simulate editing outcomes
    indel_size = random.randint(1, 5)  # Random indel size between 1-5 bp

    # Create modified sequence (simulate a deletion for simplicity)
    modified_sequence = target_seq[:cut_position] + target_seq[cut_position + indel_size :]

    log += f"  Simulated edit: {indel_size}bp deletion at cut site\n"
    log += f"  Predicted editing efficiency: {edit_success_rate * 100:.1f}%\n"

    # Step 5: Analysis of editing outcomes
    log += "\nSTEP 5: Editing Outcome Analysis\n"

    # Calculate basic stats
    log += f"  Original sequence length: {len(target_seq)} bp\n"
    log += f"  Modified sequence length: {len(modified_sequence)} bp\n"

    # Save sequences to files
    os.makedirs("crispr_results", exist_ok=True)

    original_file = "crispr_results/original_sequence.txt"
    with open(original_file, "w") as f:
        f.write(f">Original_Sequence\n{target_seq}\n")

    modified_file = "crispr_results/modified_sequence.txt"
    with open(modified_file, "w") as f:
        f.write(f">Modified_Sequence\n{modified_sequence}\n")

    log += f"  Original sequence saved to: {original_file}\n"
    log += f"  Modified sequence saved to: {modified_file}\n"

    # Summary
    log += "\nSUMMARY:\n"
    log += f"  CRISPR-Cas9 editing successfully simulated for {cell_tissue_type}\n"
    log += f"  {indel_size}bp deletion introduced at position {cut_position}\n"
    log += f"  Expected success rate in cell population: {edit_success_rate * 100:.1f}%\n"

    return log


def analyze_calcium_imaging_data(image_stack_path, output_dir="./"):
    """Analyze calcium imaging data to quantify neuronal activity metrics.

    This function processes fluorescence microscopy images of GCaMP-labeled neurons
    to extract quantitative metrics of neuronal activity, including cell counts,
    event rates, decay times, and signal-to-noise ratios.

    Parameters
    ----------
    image_stack_path : str
        Path to the time-series stack of fluorescence microscopy images (TIFF format)
    output_dir : str, optional
        Directory to save output files (default: "./")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os

    import numpy as np
    import pandas as pd
    from scipy import ndimage, signal
    from scipy.optimize import curve_fit
    from skimage import feature, filters, io, measure, segmentation

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the image stack
    log = "CALCIUM IMAGING ANALYSIS LOG\n"
    log += "===========================\n\n"
    log += f"Loading image stack from: {image_stack_path}\n"

    try:
        image_stack = io.imread(image_stack_path)
        num_frames, height, width = image_stack.shape
        log += f"Successfully loaded {num_frames} frames of size {height}x{width}\n\n"
    except Exception as e:
        return f"Error loading image stack: {str(e)}"

    # Step 2: Calculate mean image for segmentation
    log += "Step 1: Preprocessing and neuron segmentation\n"
    mean_image = np.mean(image_stack, axis=0)

    # Apply Gaussian filter to reduce noise
    smooth_mean = filters.gaussian(mean_image, sigma=2)

    # Step 3: Segment neurons using watershed
    # Find local maxima (potential cell centers)
    distance = ndimage.distance_transform_edt(smooth_mean)
    # Create a mask for local maxima instead of using indices=False parameter
    coordinates = feature.peak_local_max(distance, min_distance=10)

    # Handle case where no local maxima are found
    if len(coordinates) == 0:
        log += "No local maxima detected. Using simple thresholding instead.\n"
        # Create a simple binary mask using thresholding as fallback
        binary_mask = smooth_mean > filters.threshold_otsu(smooth_mean)
        markers = measure.label(binary_mask)
    else:
        local_max = np.zeros_like(distance, dtype=bool)
        for coord in coordinates:
            local_max[coord[0], coord[1]] = True
        markers = measure.label(local_max)

    # Watershed segmentation
    segmented = segmentation.watershed(-smooth_mean, markers, mask=smooth_mean > filters.threshold_otsu(smooth_mean))

    # Get region properties
    regions = measure.regionprops(segmented)
    cell_count = len(regions)
    log += f"Detected {cell_count} neurons in the field of view\n\n"

    # Step 4: Extract time-series data for each neuron
    log += "Step 2: Extracting fluorescence time-series for each neuron\n"
    time_series_data = []

    for _i, region in enumerate(regions):
        mask = segmented == region.label
        cell_time_series = []

        for frame in range(num_frames):
            intensity = np.mean(image_stack[frame][mask])
            cell_time_series.append(intensity)

        time_series_data.append(cell_time_series)

    time_series_array = np.array(time_series_data)

    # Step 5: Detect calcium events and calculate metrics
    log += "Step 3: Calculating neuronal activity metrics\n"

    # Function to fit exponential decay
    def exp_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c

    event_rates = []
    decay_times = []
    snr_values = []

    for i, ts in enumerate(time_series_data):
        # Normalize time series
        baseline = np.percentile(ts, 20)
        ts_norm = [(x - baseline) / baseline for x in ts]

        # Simple event detection (threshold crossing)
        threshold = np.std(ts_norm) * 2
        events = []
        in_event = False

        for j, val in enumerate(ts_norm):
            if not in_event and val > threshold:
                events.append(j)
                in_event = True
            elif in_event and val < threshold:
                in_event = False

        # Calculate event rate (events per minute, assuming 10 Hz acquisition)
        acquisition_rate = 10  # Hz (assumption)
        recording_time_minutes = num_frames / acquisition_rate / 60
        event_rate = len(events) / recording_time_minutes
        event_rates.append(event_rate)

        # Calculate decay times for detected events
        cell_decay_times = []
        for event_start in events:
            if event_start + 30 < len(ts_norm):  # Ensure enough frames after event
                event_window = ts_norm[event_start : event_start + 30]
                peak_idx = np.argmax(event_window)
                decay_segment = event_window[peak_idx:]

                try:
                    # Fit exponential decay
                    x_data = np.arange(len(decay_segment))
                    popt, _ = curve_fit(
                        exp_decay,
                        x_data,
                        decay_segment,
                        p0=[decay_segment[0], 5, decay_segment[-1]],
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    )
                    tau = popt[1]  # Decay time constant
                    cell_decay_times.append(tau / acquisition_rate)  # Convert to seconds
                except Exception:
                    # Skip if curve fitting fails
                    pass

        if cell_decay_times:
            decay_times.append(np.mean(cell_decay_times))
        else:
            decay_times.append(np.nan)

        # Calculate signal-to-noise ratio
        signal = np.mean([ts_norm[e] for e in events]) if events else 0  # noqa: F811
        noise = np.std([ts_norm[i] for i in range(len(ts_norm)) if all(abs(i - e) > 5 for e in events)])
        snr = signal / noise if noise > 0 else 0
        snr_values.append(snr)

    # Step 6: Compile and save results
    cell_metrics = pd.DataFrame(
        {
            "Cell_ID": range(1, cell_count + 1),
            "Event_Rate_per_min": event_rates,
            "Decay_Time_sec": decay_times,
            "SNR": snr_values,
        }
    )

    metrics_file = os.path.join(output_dir, "neuron_activity_metrics.csv")
    cell_metrics.to_csv(metrics_file, index=False)

    # Save time series data
    time_series_file = os.path.join(output_dir, "neuron_time_series.csv")
    time_series_df = pd.DataFrame(time_series_array.T)
    time_series_df.columns = [f"Cell_{i + 1}" for i in range(cell_count)]
    time_series_df.to_csv(time_series_file, index=False)

    # Step 7: Summarize results
    log += f"Cell count: {cell_count}\n"
    log += f"Average event rate: {np.nanmean(event_rates):.2f} events/min\n"
    log += f"Average decay time: {np.nanmean(decay_times):.2f} seconds\n"
    log += f"Average SNR: {np.nanmean(snr_values):.2f}\n\n"

    log += "Step 4: Results saved to files\n"
    log += f"Neuron activity metrics saved to: {metrics_file}\n"
    log += f"Time series data saved to: {time_series_file}\n"

    return log


def analyze_in_vitro_drug_release_kinetics(
    time_points,
    concentration_data,
    drug_name="Drug",
    total_drug_loaded=None,
    output_dir="./",
):
    """Analyzes in vitro drug release kinetics from biomaterial formulations.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points at which drug concentrations were measured (in hours)
    concentration_data : list or numpy.ndarray
        Measured drug concentration at each time point
    drug_name : str, optional
        Name of the drug being analyzed (default: "Drug")
    total_drug_loaded : float, optional
        Total amount of drug initially loaded in the formulation.
        If None, the maximum concentration is used as 100% (default: None)
    output_dir : str, optional
        Directory to save output files (default: "./")

    Returns
    -------
    str
        Research log summarizing the analysis steps, results, and saved file locations

    """
    import os
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to numpy arrays
    time_points = np.array(time_points)
    concentration_data = np.array(concentration_data)

    # Calculate cumulative release percentage
    if total_drug_loaded is None:
        total_drug_loaded = np.max(concentration_data)

    cumulative_release = (concentration_data / total_drug_loaded) * 100

    # Create a DataFrame for easier manipulation
    release_df = pd.DataFrame(
        {
            "Time (hours)": time_points,
            "Concentration": concentration_data,
            "Cumulative Release (%)": cumulative_release,
        }
    )

    # Calculate release rate (simple approximation using differences)
    release_df["Release Rate"] = np.gradient(release_df["Cumulative Release (%)"], release_df["Time (hours)"])

    # Define kinetic models
    def zero_order(t, k):
        return k * t

    def first_order(t, k):
        return 100 * (1 - np.exp(-k * t))

    def higuchi(t, k):
        return k * np.sqrt(t)

    def korsmeyer_peppas(t, k, n):
        return 100 * (k * t) ** n

    # Fit data to different kinetic models
    models = {}
    r2_values = {}

    # Zero-order kinetics
    try:
        params, _ = curve_fit(zero_order, time_points, cumulative_release)
        y_pred = zero_order(time_points, *params)
        ss_total = np.sum((cumulative_release - np.mean(cumulative_release)) ** 2)
        ss_residual = np.sum((cumulative_release - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        models["Zero-order"] = {
            "params": params,
            "equation": f"Release = {params[0]:.4f} * t",
            "pred": y_pred,
        }
        r2_values["Zero-order"] = r2
    except Exception:
        models["Zero-order"] = {
            "params": None,
            "equation": "Fitting failed",
            "pred": None,
        }
        r2_values["Zero-order"] = 0

    # First-order kinetics
    try:
        params, _ = curve_fit(first_order, time_points, cumulative_release, bounds=(0, [1]))
        y_pred = first_order(time_points, *params)
        ss_total = np.sum((cumulative_release - np.mean(cumulative_release)) ** 2)
        ss_residual = np.sum((cumulative_release - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        models["First-order"] = {
            "params": params,
            "equation": f"Release = 100 * (1 - exp(-{params[0]:.4f} * t))",
            "pred": y_pred,
        }
        r2_values["First-order"] = r2
    except Exception:
        models["First-order"] = {
            "params": None,
            "equation": "Fitting failed",
            "pred": None,
        }
        r2_values["First-order"] = 0

    # Higuchi model
    try:
        params, _ = curve_fit(higuchi, time_points, cumulative_release)
        y_pred = higuchi(time_points, *params)
        ss_total = np.sum((cumulative_release - np.mean(cumulative_release)) ** 2)
        ss_residual = np.sum((cumulative_release - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        models["Higuchi"] = {
            "params": params,
            "equation": f"Release = {params[0]:.4f} * sqrt(t)",
            "pred": y_pred,
        }
        r2_values["Higuchi"] = r2
    except Exception:
        models["Higuchi"] = {"params": None, "equation": "Fitting failed", "pred": None}
        r2_values["Higuchi"] = 0

    # Korsmeyer-Peppas model
    try:
        # Only use the first 60% of release data for Korsmeyer-Peppas model
        mask = cumulative_release <= 60
        if sum(mask) >= 3:  # Need at least 3 points for fitting
            params, _ = curve_fit(
                korsmeyer_peppas,
                time_points[mask],
                cumulative_release[mask],
                bounds=([0, 0], [1, 1]),
            )
            y_pred = korsmeyer_peppas(time_points, *params)
            ss_total = np.sum((cumulative_release - np.mean(cumulative_release)) ** 2)
            ss_residual = np.sum((cumulative_release - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            models["Korsmeyer-Peppas"] = {
                "params": params,
                "equation": f"Release = 100 * ({params[0]:.4f} * t)^{params[1]:.4f}",
                "pred": y_pred,
            }
            r2_values["Korsmeyer-Peppas"] = r2
        else:
            models["Korsmeyer-Peppas"] = {
                "params": None,
                "equation": "Insufficient data points",
                "pred": None,
            }
            r2_values["Korsmeyer-Peppas"] = 0
    except Exception:
        models["Korsmeyer-Peppas"] = {
            "params": None,
            "equation": "Fitting failed",
            "pred": None,
        }
        r2_values["Korsmeyer-Peppas"] = 0

    # Determine best model based on R² value
    best_model = max(r2_values, key=r2_values.get)

    # Calculate half-life (time to 50% release)
    try:
        # Use best model to calculate half-life
        if best_model == "Zero-order":
            k = models[best_model]["params"][0]
            half_life = 50 / k if k > 0 else float("inf")
        elif best_model == "First-order":
            k = models[best_model]["params"][0]
            half_life = -np.log(0.5) / k if k > 0 else float("inf")
        elif best_model == "Higuchi":
            k = models[best_model]["params"][0]
            half_life = (50 / k) ** 2 if k > 0 else float("inf")
        elif best_model == "Korsmeyer-Peppas":
            k, n = models[best_model]["params"]
            half_life = (0.5 ** (1 / n)) / k if k > 0 else float("inf")
        else:
            # Interpolate from data if model fitting failed
            from scipy.interpolate import interp1d

            if np.max(cumulative_release) >= 50:
                f = interp1d(cumulative_release, time_points)
                half_life = float(f(50))
            else:
                half_life = "Not reached"
    except Exception:
        half_life = "Could not calculate"

    # Create plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Cumulative release plot with model fits
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, cumulative_release, "o-", label="Experimental data")

    for model_name, model_data in models.items():
        if model_data["pred"] is not None:
            plt.plot(
                time_points,
                model_data["pred"],
                "--",
                label=f"{model_name} (R² = {r2_values[model_name]:.4f})",
            )

    plt.xlabel("Time (hours)")
    plt.ylabel("Cumulative Release (%)")
    plt.title(f"In Vitro Release Profile of {drug_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    cumulative_plot_path = os.path.join(output_dir, f"cumulative_release_{timestamp}.png")
    plt.savefig(cumulative_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Release rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, release_df["Release Rate"], "o-")
    plt.xlabel("Time (hours)")
    plt.ylabel("Release Rate (%/hour)")
    plt.title(f"Release Rate of {drug_name}")
    plt.grid(True, linestyle="--", alpha=0.7)
    rate_plot_path = os.path.join(output_dir, f"release_rate_{timestamp}.png")
    plt.savefig(rate_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save data to CSV
    csv_path = os.path.join(output_dir, f"drug_release_data_{timestamp}.csv")
    release_df.to_csv(csv_path, index=False)

    # Generate research log
    log = f"""
# In Vitro Drug Release Kinetics Analysis for {drug_name}

## Analysis Summary
- **Date/Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Drug Analyzed:** {drug_name}
- **Time Range:** {min(time_points)} to {max(time_points)} hours
- **Number of Data Points:** {len(time_points)}
- **Maximum Release Achieved:** {max(cumulative_release):.2f}%

## Kinetic Models Analysis
The release data was fitted to four standard kinetic models:

1. **Zero-order Model:** {models["Zero-order"]["equation"]} (R² = {r2_values["Zero-order"]:.4f})
2. **First-order Model:** {models["First-order"]["equation"]} (R² = {r2_values["First-order"]:.4f})
3. **Higuchi Model:** {models["Higuchi"]["equation"]} (R² = {r2_values["Higuchi"]:.4f})
4. **Korsmeyer-Peppas Model:** {models["Korsmeyer-Peppas"]["equation"]} (R² = {r2_values["Korsmeyer-Peppas"]:.4f})

**Best-fitting Model:** {best_model} (R² = {r2_values[best_model]:.4f})

## Release Metrics
- **Half-life (t50%):** {half_life if isinstance(half_life, str) else f"{half_life:.2f} hours"}
- **Initial Release Rate:** {release_df["Release Rate"].iloc[0]:.4f} %/hour
- **Average Release Rate:** {np.mean(release_df["Release Rate"]):.4f} %/hour

## Files Generated
1. Cumulative Release Plot: {cumulative_plot_path}
2. Release Rate Plot: {rate_plot_path}
3. Data CSV: {csv_path}

## Interpretation
The drug release profile of {drug_name} best follows a {
        best_model
    } kinetic model, which suggests that the release mechanism is primarily driven by {
        "diffusion through a porous matrix"
        if best_model == "Higuchi"
        else "diffusion with erosion"
        if best_model == "Korsmeyer-Peppas" and 0.43 <= models[best_model]["params"][1] <= 0.85
        else "Fickian diffusion"
        if best_model == "Korsmeyer-Peppas" and models[best_model]["params"][1] < 0.43
        else "case-II transport"
        if best_model == "Korsmeyer-Peppas" and models[best_model]["params"][1] > 0.85
        else "concentration-dependent diffusion"
        if best_model == "First-order"
        else "constant release rate independent of concentration"
        if best_model == "Zero-order"
        else "complex mechanisms"
    }.
"""

    return log.strip()


def analyze_myofiber_morphology(
    image_path,
    nuclei_channel=2,
    myofiber_channel=1,
    threshold_method="otsu",
    output_dir="./",
):
    """Quantifies morphological properties of myofibers in microscopy images of tissue sections.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image file (typically a multichannel image with nuclei and myofiber staining)
    nuclei_channel : int, default=2
        Channel index containing nuclei staining (DAPI, Hoechst, etc.)
    myofiber_channel : int, default=1
        Channel index containing myofiber staining (α-Actinin, etc.)
    threshold_method : str, default='otsu'
        Method for thresholding ('otsu', 'adaptive', or 'manual')
    output_dir : str, default='./'
        Directory to save output files

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from skimage import exposure, filters, io, measure, morphology
    from skimage.color import label2rgb

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    image = io.imread(image_path)

    # Extract channels (assuming multichannel image)
    if len(image.shape) > 2:
        if len(image.shape) == 3:
            # RGB image
            nuclei_img = image[:, :, nuclei_channel] if nuclei_channel < image.shape[2] else image[:, :, 0]
            myofiber_img = image[:, :, myofiber_channel] if myofiber_channel < image.shape[2] else image[:, :, 1]
        else:
            # Multichannel image (e.g., from confocal)
            nuclei_img = image[nuclei_channel, :, :] if nuclei_channel < image.shape[0] else image[0, :, :]
            myofiber_img = image[myofiber_channel, :, :] if myofiber_channel < image.shape[0] else image[1, :, :]
    else:
        # Single channel image - can't separate nuclei and myofibers
        return "Error: Input image must be multichannel to separate nuclei and myofibers"

    # Enhance contrast
    nuclei_img = exposure.equalize_adapthist(nuclei_img)
    myofiber_img = exposure.equalize_adapthist(myofiber_img)

    # Segment nuclei
    if threshold_method == "otsu":
        nuclei_thresh = filters.threshold_otsu(nuclei_img)
    elif threshold_method == "adaptive":
        nuclei_thresh = filters.threshold_local(nuclei_img, block_size=35)
    else:  # manual
        nuclei_thresh = np.mean(nuclei_img) * 1.5

    nuclei_binary = nuclei_img > nuclei_thresh
    nuclei_binary = morphology.remove_small_objects(nuclei_binary, min_size=30)
    nuclei_binary = morphology.binary_closing(nuclei_binary)

    # Label nuclei
    nuclei_labels = measure.label(nuclei_binary)
    nuclei_props = measure.regionprops(nuclei_labels)

    # Segment myofibers
    if threshold_method == "otsu":
        myofiber_thresh = filters.threshold_otsu(myofiber_img)
    elif threshold_method == "adaptive":
        myofiber_thresh = filters.threshold_local(myofiber_img, block_size=101)
    else:  # manual
        myofiber_thresh = np.mean(myofiber_img) * 1.2

    myofiber_binary = myofiber_img > myofiber_thresh
    myofiber_binary = morphology.remove_small_objects(myofiber_binary, min_size=500)
    myofiber_binary = morphology.binary_closing(myofiber_binary, morphology.disk(3))

    # Label myofibers
    myofiber_labels = measure.label(myofiber_binary)
    myofiber_props = measure.regionprops(myofiber_labels)

    # Count nuclei inside myofibers
    nuclei_inside = 0
    nuclei_total = len(nuclei_props)

    for nucleus in nuclei_props:
        y, x = nucleus.centroid
        y, x = int(y), int(x)
        if myofiber_binary[y, x]:
            nuclei_inside += 1

    percent_inside = nuclei_inside / nuclei_total * 100 if nuclei_total > 0 else 0

    # Calculate myofiber morphological properties
    myofiber_data = []
    for fiber in myofiber_props:
        myofiber_data.append(
            {
                "Area": fiber.area,
                "Perimeter": fiber.perimeter,
                "Eccentricity": fiber.eccentricity,
                "Solidity": fiber.solidity,
                "Orientation": fiber.orientation,
            }
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/myofiber_analysis_{timestamp}.csv"

    if myofiber_data:
        df = pd.DataFrame(myofiber_data)
        df.to_csv(results_file, index=False)

        # Calculate summary statistics
        mean_area = df["Area"].mean()
        mean_perimeter = df["Perimeter"].mean()
        mean_eccentricity = df["Eccentricity"].mean()
    else:
        mean_area = mean_perimeter = mean_eccentricity = 0

    # Save labeled image
    labeled_image = label2rgb(myofiber_labels, image=myofiber_img)
    labeled_image_path = f"{output_dir}/labeled_myofibers_{timestamp}.png"
    io.imsave(labeled_image_path, (labeled_image * 255).astype(np.uint8))

    # Create research log
    log = f"""
MYOFIBER MORPHOLOGICAL ANALYSIS REPORT
======================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {image_path}

ANALYSIS STEPS:
1. Loaded multichannel microscopy image
2. Extracted nuclei (channel {nuclei_channel}) and myofiber (channel {myofiber_channel}) signals
3. Enhanced contrast using adaptive histogram equalization
4. Segmented nuclei using {threshold_method} thresholding
5. Segmented myofibers using {threshold_method} thresholding
6. Performed morphological operations to refine segmentation
7. Identified and measured individual myofibers and nuclei

RESULTS:
- Total myofibers detected: {len(myofiber_props)}
- Total nuclei detected: {nuclei_total}
- Nuclei inside myofibers: {nuclei_inside} ({percent_inside:.2f}%)
- Mean myofiber area: {mean_area:.2f} pixels
- Mean myofiber perimeter: {mean_perimeter:.2f} pixels
- Mean myofiber eccentricity: {mean_eccentricity:.2f}

FILES GENERATED:
- Morphological measurements: {results_file}
- Labeled myofiber image: {labeled_image_path}
"""

    return log


def decode_behavior_from_neural_trajectories(neural_data, behavioral_data, n_components=10, output_dir="./"):
    """Model neural activity trajectories and decode behavioral variables.

    Parameters
    ----------
    neural_data : numpy.ndarray
        Neural spiking activity data, shape (n_timepoints, n_neurons)
    behavioral_data : numpy.ndarray
        Behavioral data, shape (n_timepoints, n_behavioral_variables)
    n_components : int, optional
        Number of principal components to use for dimensionality reduction, default is 10
    output_dir : str, optional
        Directory to save output files, default is "./"

    Returns
    -------
    str
        Research log summarizing the steps taken and results

    """
    import os
    import pickle

    import matplotlib.pyplot as plt
    import numpy as np
    from pykalman import KalmanFilter
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize research log
    log = "# Neural Trajectory Modeling and Decoding Research Log\n\n"

    # Step 1: Preprocess the data
    log += "## Step 1: Data Preprocessing\n"
    log += f"- Neural data shape: {neural_data.shape}\n"
    log += f"- Behavioral data shape: {behavioral_data.shape}\n"

    # Check for NaN values and replace with zeros
    neural_data = np.nan_to_num(neural_data)
    behavioral_data = np.nan_to_num(behavioral_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(neural_data, behavioral_data, test_size=0.2, random_state=42)
    log += f"- Training set size: {X_train.shape[0]} samples\n"
    log += f"- Testing set size: {X_test.shape[0]} samples\n\n"

    # Step 2: Dimensionality reduction with PCA
    log += "## Step 2: Dimensionality Reduction\n"
    log += f"- Reducing neural data from {neural_data.shape[1]} dimensions to {n_components} components\n"

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    log += f"- Total variance explained: {explained_variance:.2f}%\n\n"

    # Save PCA components visualization
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Components Explained Variance")
        plt.xticks(range(1, n_components + 1))
        plt.tight_layout()

        pca_plot_path = os.path.join(output_dir, "pca_explained_variance.png")
        plt.savefig(pca_plot_path, dpi=300)
        plt.close()

        log += f"- PCA components visualization saved to: {pca_plot_path}\n\n"
    except Exception as e:
        log += f"- Error creating PCA visualization: {str(e)}\n\n"

    # Step 3: Train a Kalman filter for decoding
    log += "## Step 3: Trajectory Modeling and Decoding\n"
    log += "- Training Kalman filter to decode behavioral variables from neural trajectories\n"

    # Initialize and train Kalman filter
    kf = KalmanFilter(initial_state_mean=np.zeros(y_train.shape[1]), n_dim_obs=X_train_pca.shape[1])

    # Fit the Kalman filter to the data
    kf.em(X_train_pca, y_train)

    # Step 4: Decode behavioral variables
    log += "## Step 4: Decoding Behavioral Variables\n"

    # Use the Kalman filter to predict behavioral variables
    y_pred, _ = kf.filter(X_test_pca)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    log += f"- Mean squared error on test set: {mse:.4f}\n\n"

    # Save the decoded trajectories visualization
    try:
        if y_test.shape[1] >= 2:
            # Create visualization of true vs. predicted trajectories (first 2 dimensions)
            plt.figure(figsize=(12, 6))

            # First behavioral variable
            plt.subplot(1, 2, 1)
            plt.plot(y_test[:, 0], label="True")
            plt.plot(y_pred[:, 0], label="Predicted")
            plt.xlabel("Time steps")
            plt.ylabel("Behavioral Variable 1")
            plt.title("Decoding Performance - Variable 1")
            plt.legend()

            # Second behavioral variable
            plt.subplot(1, 2, 2)
            plt.plot(y_test[:, 1], label="True")
            plt.plot(y_pred[:, 1], label="Predicted")
            plt.xlabel("Time steps")
            plt.ylabel("Behavioral Variable 2")
            plt.title("Decoding Performance - Variable 2")
            plt.legend()

            plt.tight_layout()

            trajectory_plot_path = os.path.join(output_dir, "decoded_trajectories.png")
            plt.savefig(trajectory_plot_path, dpi=300)
            plt.close()

            log += f"- Decoded trajectories visualization saved to: {trajectory_plot_path}\n"
    except Exception as e:
        log += f"- Error creating trajectory visualization: {str(e)}\n"

    # Save the results as a pickle file
    results = {
        "true_behavior": y_test,
        "predicted_behavior": y_pred,
        "pca_model": pca,
        "kalman_filter": kf,
        "mse": mse,
    }

    results_file = os.path.join(output_dir, "neural_decoding_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    # Also save a CSV with the first few predicted vs. actual values for easier inspection
    try:
        import pandas as pd

        n_samples = min(100, y_test.shape[0])
        n_vars = y_test.shape[1]

        results_data = {}
        for i in range(n_vars):
            results_data[f"true_var{i + 1}"] = y_test[:n_samples, i]
            results_data[f"pred_var{i + 1}"] = y_pred[:n_samples, i]

        results_df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, "decoding_results_sample.csv")
        results_df.to_csv(csv_path, index=False)

        log += f"- Sample of decoding results saved to: {csv_path}\n"
    except Exception as e:
        log += f"- Error creating CSV results: {str(e)}\n"

    log += "\n## Results\n"
    log += f"- Full decoded behavioral trajectories saved to: {results_file}\n"
    log += f"- Decoder performance (MSE): {mse:.4f}\n"

    # Save the log to a file
    log_file = os.path.join(output_dir, "neural_decoding_log.txt")
    with open(log_file, "w") as f:
        f.write(log)

    log += f"- Analysis log saved to: {log_file}\n"

    return log


def simulate_whole_cell_ode_model(
    initial_conditions,
    parameters,
    ode_function=None,
    time_span=(0, 100),
    time_points=1000,
    method="LSODA",
):
    """Simulate a whole-cell model represented as a system of ordinary differential equations (ODEs).

    Parameters
    ----------
    initial_conditions : dict or array-like
        Initial values for each state variable in the model. If dict, keys are variable names
        and values are initial concentrations/values. If array-like, order must match the
        order expected by the ODE function.
    parameters : dict
        Model parameters required by the ODE function. Keys are parameter names and
        values are parameter values.
    ode_function : callable, optional
        Function defining the system of ODEs. Should take arguments ``(t, y, *args)`` where
        ``t`` is time, ``y`` is the state vector, and ``args`` contains additional parameters.
        If None, a simple example whole-cell model will be used.
    time_span : tuple, default=(0, 100)
        Tuple of (start_time, end_time) for the simulation.
    time_points : int, default=1000
        Number of time points to evaluate.
    method : str, default='LSODA'
        Numerical integration method to use (e.g., 'RK45', 'LSODA', 'BDF').

    Returns
    -------
    str
        Research log summarizing the simulation steps and results. Results are saved
        to a CSV file and the filename is included in the log.

    """
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Define a default ODE function if none is provided
    if ode_function is None:

        def default_whole_cell_model(t, y, params):
            # Unpack state variables
            # Simple model with:
            # - mRNA (y[0])
            # - Protein (y[1])
            # - Metabolite (y[2])
            # - ATP (y[3])
            mRNA, protein, metabolite, atp = y

            # Unpack parameters
            k_transcription = params["k_transcription"]  # mRNA synthesis rate
            k_translation = params["k_translation"]  # Protein synthesis rate
            k_mrna_deg = params["k_mrna_deg"]  # mRNA degradation rate
            k_protein_deg = params["k_protein_deg"]  # Protein degradation rate
            k_metabolism = params["k_metabolism"]  # Metabolite production rate
            k_atp_production = params["k_atp_production"]  # ATP production rate
            k_atp_consumption = params["k_atp_consumption"]  # ATP consumption rate

            # ODEs
            dmRNA_dt = k_transcription - k_mrna_deg * mRNA
            dprotein_dt = k_translation * mRNA * atp - k_protein_deg * protein
            dmetabolite_dt = k_metabolism * protein - k_atp_production * metabolite
            datp_dt = k_atp_production * metabolite - k_atp_consumption * atp - k_translation * mRNA * atp

            return [dmRNA_dt, dprotein_dt, dmetabolite_dt, datp_dt]

        ode_function = default_whole_cell_model

    # Prepare initial conditions as array
    if isinstance(initial_conditions, dict):
        y0_values = list(initial_conditions.values())
        variable_names = list(initial_conditions.keys())
    else:
        y0_values = initial_conditions
        variable_names = [f"Variable_{i}" for i in range(len(initial_conditions))]

    # Set up time points
    t_eval = np.linspace(time_span[0], time_span[1], time_points)

    # Start research log
    log = []
    log.append("# Whole-Cell ODE Model Simulation")
    log.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("\n## Simulation Setup")
    log.append(f"- Integration method: {method}")
    log.append(f"- Time span: {time_span[0]} to {time_span[1]} time units")
    log.append(f"- Number of time points: {time_points}")
    log.append(f"- Number of state variables: {len(y0_values)}")
    log.append("\n## Initial Conditions")
    for _i, (name, value) in enumerate(zip(variable_names, y0_values, strict=False)):
        log.append(f"- {name}: {value}")

    log.append("\n## Model Parameters")
    for param, value in parameters.items():
        log.append(f"- {param}: {value}")

    # Solve the ODE system
    log.append("\n## Running Simulation")
    try:
        solution = solve_ivp(
            lambda t, y: ode_function(t, y, parameters),
            time_span,
            y0_values,
            method=method,
            t_eval=t_eval,
        )

        # Check if simulation was successful
        if solution.success:
            log.append("Simulation completed successfully.")
            log.append(f"- Number of function evaluations: {solution.nfev}")
            log.append(f"- Number of Jacobian evaluations: {solution.njev}")
            log.append(f"- Number of steps: {len(solution.t)}")

            # Create DataFrame with results
            results_df = pd.DataFrame(solution.y.T, columns=variable_names)
            results_df.insert(0, "Time", solution.t)

            # Save results to CSV
            filename = f"whole_cell_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(filename, index=False)

            log.append("\n## Results Summary")
            log.append(f"Simulation results saved to: {filename}")

            # Calculate some basic statistics
            final_state = results_df.iloc[-1].drop("Time").to_dict()
            log.append("\n## Final State")
            for var, value in final_state.items():
                log.append(f"- {var}: {value:.6f}")

        else:
            log.append(f"Simulation failed with message: {solution.message}")

    except Exception as e:
        log.append(f"Error during simulation: {str(e)}")

    # Return the research log
    return "\n".join(log)
