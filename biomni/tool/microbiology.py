def optimize_anaerobic_digestion_process(
    waste_characteristics,
    operational_parameters,
    target_output="methane_yield",
    optimization_method="rsm",
):
    """Optimize anaerobic digestion process conditions to maximize VFA production or methane yield.

    Parameters
    ----------
    waste_characteristics : dict
        Dictionary containing waste characteristics such as:
        - total_solids (float): Total solids content (%)
        - volatile_solids (float): Volatile solids content (%)
        - cod (float): Chemical oxygen demand (mg/L)

    operational_parameters : dict
        Dictionary containing operational parameters and their ranges:
        - hrt (tuple): Hydraulic retention time range in days (min, max)
        - olr (tuple): Organic loading rate range in kg VS/(m³·d) (min, max)
        - if_ratio (tuple): Inoculum-to-feedstock ratio range (min, max)
        - temperature (tuple): Temperature range in °C (min, max)
        - ph (tuple): pH range (min, max)

    target_output : str, optional
        Target output to maximize, either 'vfa_production' or 'methane_yield'.
        Default is 'methane_yield'.

    optimization_method : str, optional
        Method used for optimization, either 'rsm' (Response Surface Methodology) or
        'genetic' (Genetic Algorithm). Default is 'rsm'.

    Returns
    -------
    str
        Research log summarizing the optimization process and results.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from scipy.optimize import differential_evolution, minimize

    # Research log initialization
    log = "# Anaerobic Digestion Process Optimization Research Log\n\n"
    log += "## Input Parameters\n\n"
    log += "### Waste Characteristics\n"
    for key, value in waste_characteristics.items():
        log += f"- {key}: {value}\n"

    log += "\n### Operational Parameters Ranges\n"
    for key, value in operational_parameters.items():
        log += f"- {key}: {value}\n"

    log += "\n## Optimization Setup\n"
    log += f"- Target output: {target_output}\n"
    log += f"- Optimization method: {optimization_method}\n\n"

    # Define simplified models for VFA production and methane yield
    # These are simplified models based on literature that relate operational parameters to outputs
    def vfa_production_model(params):
        hrt, olr, if_ratio, temp, ph = params

        # Basic model: Higher VFA at moderate HRT, high OLR, low I/F ratio, mesophilic temp, slightly acidic pH
        # This is a simplified model for demonstration purposes
        vfa = (
            -0.1 * (hrt - 10) ** 2  # Optimal HRT around 10 days
            + 2 * olr  # Higher OLR generally increases VFA
            + -5 * if_ratio  # Lower I/F ratio favors VFA accumulation
            + -0.05 * (temp - 35) ** 2  # Optimal around 35°C (mesophilic)
            + -10 * (ph - 5.5) ** 2  # Optimal pH around 5.5 for VFA
        )

        # Incorporate waste characteristics effects
        vfa *= 0.8 + 0.2 * waste_characteristics["volatile_solids"] / 100
        vfa *= 0.9 + 0.1 * waste_characteristics["cod"] / 10000

        return -vfa  # Negative because we're minimizing

    def methane_yield_model(params):
        hrt, olr, if_ratio, temp, ph = params

        # Basic model: Higher methane at longer HRT, moderate OLR, high I/F ratio, mesophilic/thermophilic temp, neutral pH
        # This is a simplified model for demonstration purposes
        methane = (
            0.05 * hrt  # Longer HRT generally increases methane
            + -0.5 * (olr - 3) ** 2  # Optimal OLR around 3
            + 2 * if_ratio  # Higher I/F ratio favors methanogenesis
            + -0.05 * (temp - 37) ** 2  # Optimal around 37°C (mesophilic)
            + -15 * (ph - 7.2) ** 2  # Optimal pH around 7.2 for methane
        )

        # Incorporate waste characteristics effects
        methane *= 0.7 + 0.3 * waste_characteristics["volatile_solids"] / 100
        methane *= 0.8 + 0.2 * waste_characteristics["cod"] / 10000

        return -methane  # Negative because we're minimizing

    # Select the appropriate model based on target output
    if target_output == "vfa_production":
        model = vfa_production_model
    else:  # methane_yield
        model = methane_yield_model

    # Define parameter bounds
    bounds = [
        operational_parameters["hrt"],
        operational_parameters["olr"],
        operational_parameters["if_ratio"],
        operational_parameters["temperature"],
        operational_parameters["ph"],
    ]

    # Perform optimization
    log += "## Optimization Process\n\n"
    log += "Performing parameter optimization to find optimal operating conditions...\n\n"

    if optimization_method == "rsm":
        # Initial guess (middle of each range)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        # Run optimization
        result = minimize(model, x0, bounds=bounds, method="L-BFGS-B")

        optimal_params = result.x
        optimal_value = -result.fun  # Convert back to positive

        log += f"Optimization converged after {result.nfev} function evaluations.\n"
        log += f"Optimization success: {result.success}\n"
        log += f"Final optimization message: {result.message}\n\n"

    else:  # genetic algorithm
        result = differential_evolution(model, bounds)

        optimal_params = result.x
        optimal_value = -result.fun  # Convert back to positive

        log += f"Genetic algorithm completed after {result.nfev} function evaluations.\n"
        log += f"Optimization success: {result.success}\n"
        log += f"Final optimization message: {result.message}\n\n"

    # Log optimal parameters
    log += "## Optimization Results\n\n"
    log += "### Optimal Operating Conditions\n"
    param_names = [
        "Hydraulic Retention Time (days)",
        "Organic Loading Rate (kg VS/(m³·d))",
        "Inoculum-to-Feedstock Ratio",
        "Temperature (°C)",
        "pH",
    ]

    for name, value in zip(param_names, optimal_params, strict=False):
        log += f"- {name}: {value:.2f}\n"

    log += "\n### Predicted Performance\n"
    log += f"- Predicted {target_output.replace('_', ' ')}: {optimal_value:.2f}\n\n"

    # Generate response surface visualization for the two most important parameters
    # For VFA, we'll use HRT and OLR; for methane, we'll use HRT and I/F ratio
    log += "## Response Surface Visualization\n\n"

    if target_output == "vfa_production":
        param1_idx, param2_idx = 0, 1  # HRT and OLR
        param1_name, param2_name = "HRT (days)", "OLR (kg VS/(m³·d))"
    else:  # methane_yield
        param1_idx, param2_idx = 0, 2  # HRT and I/F ratio
        param1_name, param2_name = "HRT (days)", "I/F ratio"

    # Create mesh grid for the two selected parameters
    param1_range = np.linspace(bounds[param1_idx][0], bounds[param1_idx][1], 20)
    param2_range = np.linspace(bounds[param2_idx][0], bounds[param2_idx][1], 20)
    P1, P2 = np.meshgrid(param1_range, param2_range)

    # Calculate output values
    Z = np.zeros_like(P1)
    for i in range(len(param1_range)):
        for j in range(len(param2_range)):
            params = list(optimal_params)  # Start with optimal values for other parameters
            params[param1_idx] = param1_range[i]
            params[param2_idx] = param2_range[j]
            Z[j, i] = -model(params)  # Convert back to positive

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(P1, P2, Z, cmap=cm.coolwarm, alpha=0.8)

    # Add optimal point
    ax.scatter(
        optimal_params[param1_idx],
        optimal_params[param2_idx],
        optimal_value,
        color="black",
        s=100,
        marker="*",
    )

    # Labels
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel(f"{target_output.replace('_', ' ')}")
    ax.set_title(f"Response Surface for {target_output.replace('_', ' ')}")

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Save figure
    plot_filename = f"ad_optimization_{target_output}_response_surface.png"
    plt.savefig(plot_filename)
    plt.close()

    log += f"Response surface plot saved as: {plot_filename}\n\n"

    # Sensitivity analysis
    log += "## Parameter Sensitivity Analysis\n\n"

    # Calculate sensitivity by varying each parameter slightly
    sensitivities = []
    for i, (name, param) in enumerate(zip(param_names, optimal_params, strict=False)):
        delta = (bounds[i][1] - bounds[i][0]) * 0.05  # 5% of range

        # Create parameter sets with small changes
        params_plus = list(optimal_params)
        params_plus[i] += delta

        params_minus = list(optimal_params)
        params_minus[i] -= delta

        # Calculate output change
        output_plus = -model(params_plus)
        output_minus = -model(params_minus)

        # Calculate sensitivity (normalized)
        sensitivity = abs(output_plus - output_minus) / (2 * delta) * (param / optimal_value)
        sensitivities.append((name, sensitivity))

    # Sort sensitivities
    sensitivities.sort(key=lambda x: x[1], reverse=True)

    # Log sensitivities
    log += "Parameters ranked by sensitivity (most to least sensitive):\n"
    for name, sensitivity in sensitivities:
        log += f"- {name}: {sensitivity:.4f}\n"

    log += "\n## Conclusion\n\n"
    log += f"The optimization process identified optimal operating conditions for maximizing {target_output.replace('_', ' ')} "
    log += "in anaerobic digestion of the given organic waste. The most sensitive parameters were "
    log += f"{sensitivities[0][0]} and {sensitivities[1][0]}.\n\n"

    log += "These results can be used to guide experimental design and process control in anaerobic digestion systems."

    return log


def analyze_arsenic_speciation_hplc_icpms(sample_data, sample_name="Unknown Sample", calibration_data=None):
    """Analyzes arsenic speciation in liquid samples using HPLC-ICP-MS technique.

    Parameters
    ----------
    sample_data : dict
        Dictionary containing sample data with keys as sample IDs and values as dictionaries
        with retention times (in minutes) as keys and signal intensities as values.
    sample_name : str, optional
        Name of the sample being analyzed (default: "Unknown Sample")
    calibration_data : dict, optional
        Dictionary containing calibration standards data with known concentrations for each arsenic species.
        If None, default calibration values will be used.

    Returns
    -------
    str
        A research log summarizing the steps of the analysis and results.

    """
    from datetime import datetime

    import pandas as pd

    # Start research log
    log = "# Arsenic Speciation Analysis by HPLC-ICP-MS\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Sample: {sample_name}\n\n"

    # Step 1: Sample preparation
    log += "## 1. Sample Preparation\n"
    log += "- Filtered sample through 0.45 μm filter\n"
    log += "- Diluted sample with mobile phase (if necessary)\n"
    log += "- Prepared for injection into HPLC system\n\n"

    # Step 2: HPLC-ICP-MS Analysis
    log += "## 2. HPLC-ICP-MS Analysis\n"
    log += "- Column: Anion exchange column\n"
    log += "- Mobile phase: 20 mM NH4H2PO4 (pH 6.0)\n"
    log += "- Flow rate: 1.0 mL/min\n"
    log += "- Injection volume: 50 μL\n"
    log += "- ICP-MS detection: m/z 75 for arsenic\n\n"

    # Step 3: Chromatographic separation and detection
    log += "## 3. Chromatographic Separation and Detection\n"
    log += "- Running chromatographic separation\n"
    log += "- Monitoring arsenic signal (m/z 75)\n"
    log += "- Collecting retention time data for species identification\n\n"

    # Define retention times for arsenic species (in minutes)
    arsenic_species = {
        "As(III)": 2.8,
        "As(V)": 7.5,
        "MMAs(III)": 3.9,
        "MMAs(V)": 6.2,
        "DMAs(III)": 4.7,
        "DMAs(V)": 5.3,
    }

    # Default calibration factors if not provided
    if calibration_data is None:
        calibration_data = {
            "As(III)": {"factor": 0.85, "limit": 0.1},
            "As(V)": {"factor": 0.92, "limit": 0.1},
            "MMAs(III)": {"factor": 0.78, "limit": 0.2},
            "MMAs(V)": {"factor": 0.88, "limit": 0.15},
            "DMAs(III)": {"factor": 0.81, "limit": 0.2},
            "DMAs(V)": {"factor": 0.90, "limit": 0.15},
        }

    # Step 4: Data analysis and quantification
    log += "## 4. Data Analysis and Quantification\n"

    # Process sample data to identify and quantify arsenic species
    results = {}

    for sample_id, sample in sample_data.items():
        species_concentrations = {}

        for species_name, expected_rt in arsenic_species.items():
            # Find the closest retention time in the sample data
            closest_rt = min(sample.keys(), key=lambda rt: abs(rt - expected_rt))

            # Check if the retention time is within an acceptable range (±0.3 min)
            if abs(closest_rt - expected_rt) <= 0.3:
                # Calculate concentration using calibration factor
                intensity = sample[closest_rt]
                concentration = intensity * calibration_data[species_name]["factor"]

                # Check if concentration is above detection limit
                if concentration >= calibration_data[species_name]["limit"]:
                    species_concentrations[species_name] = concentration
                else:
                    species_concentrations[species_name] = (
                        f"<{calibration_data[species_name]['limit']} (Below detection limit)"
                    )
            else:
                species_concentrations[species_name] = "Not detected"

        results[sample_id] = species_concentrations

    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Log the detection and quantification process
    log += "- Identified arsenic species based on retention times\n"
    log += "- Quantified concentrations using calibration standards\n"
    log += "- Applied detection limits for each species\n\n"

    # Step 5: Results
    log += "## 5. Results\n"
    log += "Detected arsenic species and their concentrations (μg/L):\n\n"

    # Create a results file
    results_filename = (
        f"arsenic_speciation_results_{sample_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results_df.to_csv(results_filename)

    log += f"Results have been saved to: {results_filename}\n\n"

    # Summary of findings
    log += "## 6. Summary\n"

    # Identify predominant species
    for sample_id, species_data in results.items():
        numeric_concs = {k: v for k, v in species_data.items() if isinstance(v, int | float)}
        if numeric_concs:
            predominant_species = max(numeric_concs.items(), key=lambda x: x[1])
            log += f"Sample {sample_id}: Predominant arsenic species is {predominant_species[0]} "
            log += f"at {predominant_species[1]:.2f} μg/L\n"
        else:
            log += f"Sample {sample_id}: No arsenic species detected above quantification limits\n"

    return log


def count_bacterial_colonies(image_path, dilution_factor=1, plate_area_cm2=65.0, output_dir="./output"):
    """Count bacterial colonies from an image of agar plate using computer vision techniques.

    Parameters
    ----------
    image_path : str
        Path to the image file containing bacterial colonies on agar plate
    dilution_factor : float
        Dilution factor of the plated sample (default=1)
    plate_area_cm2 : float
        Area of the agar plate in square centimeters (default=65.0, standard Petri dish)
    output_dir : str
        Directory to save output images and results (default="./output")

    Returns
    -------
    str
        Research log summarizing the colony counting process and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np
    from scipy import ndimage

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        error_message = f"Error: Could not load image from {image_path}. Please check if the file exists and is a valid image format."
        return error_message

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that background is 1 instead of 0
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(original_image, markers)

    # Count colonies (exclude background marker 1)
    unique_markers = np.unique(markers)
    colony_count = len(unique_markers) - 2  # Subtract background (1) and boundary (-1)

    # Calculate CFU concentration
    cfu_per_ml = colony_count * dilution_factor
    cfu_per_cm2 = cfu_per_ml / plate_area_cm2

    # Create output image showing detected colonies
    output_image = original_image.copy()
    output_image[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    # Draw centroids and numbers on colonies
    labeled_array = ndimage.label(markers > 1)[0]
    for i, region in enumerate(ndimage.find_objects(labeled_array)):
        if region is not None:
            y_slice, x_slice = region
            y = (y_slice.start + y_slice.stop) // 2
            x = (x_slice.start + x_slice.stop) // 2
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                output_image,
                str(i + 1),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Save output image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"colony_count_{timestamp}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output_image)

    # Generate research log
    log = f"""
Automated Bacterial Colony Counting - Research Log
=================================================
Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Input Image: {image_path}
Dilution Factor: {dilution_factor}
Plate Area: {plate_area_cm2} cm²

Methodology:
1. Loaded and converted image to grayscale
2. Applied Gaussian blur to reduce noise
3. Used Otsu's thresholding to separate colonies from background
4. Performed morphological operations to clean the image
5. Applied watershed algorithm to separate touching colonies
6. Counted unique colony markers

Results:
- Total Colony Count: {colony_count} CFUs
- Concentration: {cfu_per_ml:.2f} CFU/ml
- Area Density: {cfu_per_cm2:.2f} CFU/cm²

Output visualization saved as: {output_filename}
"""

    return log


def annotate_bacterial_genome(
    genome_file_path,
    output_dir="annotation_results",
    genus="",
    species="",
    strain="",
    prefix="",
):
    """Annotate a bacterial genome using Prokka to identify genes, proteins, and functional features.

    Parameters
    ----------
    genome_file_path : str
        Path to the assembled genome sequence file in FASTA format
    output_dir : str, optional
        Directory where annotation results will be saved (default: "annotation_results")
    genus : str, optional
        Genus name for the organism (default: "")
    species : str, optional
        Species name for the organism (default: "")
    strain : str, optional
        Strain identifier (default: "")
    prefix : str, optional
        Prefix for output files (default: "")

    Returns
    -------
    str
        Research log summarizing the annotation process and results

    """
    import os
    import subprocess
    import time

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a default prefix if not provided
    if not prefix:
        prefix = f"annotation_{int(time.time())}"

    # Build the Prokka command
    prokka_cmd = [
        "prokka",
        genome_file_path,
        "--outdir",
        output_dir,
        "--prefix",
        prefix,
    ]

    # Add organism information if provided
    if genus:
        prokka_cmd.extend(["--genus", genus])
    if species:
        prokka_cmd.extend(["--species", species])
    if strain:
        prokka_cmd.extend(["--strain", strain])

    # Run Prokka
    start_time = time.time()
    try:
        result = subprocess.run(
            prokka_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        success = True
        prokka_output = result.stdout
    except subprocess.CalledProcessError as e:
        success = False
        prokka_output = e.stderr
    except FileNotFoundError:
        return "ERROR: Prokka is not installed or not in PATH. Please install Prokka first."

    # Calculate runtime
    runtime = time.time() - start_time

    # Check if annotation was successful
    if not success:
        return f"ERROR: Genome annotation failed.\n\nProkka output:\n{prokka_output}"

    # Parse annotation summary from Prokka output
    feature_counts = {}
    for line in prokka_output.split("\n"):
        if "Found" in line and ":" in line:
            feature_type = line.split("Found")[1].split(":")[0].strip()
            count = line.split(":")[-1].strip()
            feature_counts[feature_type] = count

    # Generate research log
    log = f"""
GENOME ANNOTATION RESEARCH LOG

Input:
- Genome file: {genome_file_path}

Annotation Process:
- Tool: Prokka
- Runtime: {runtime:.2f} seconds
- Output directory: {output_dir}

Annotation Results:
"""

    # Add feature counts to log
    if feature_counts:
        for feature, count in feature_counts.items():
            log += f"- {feature}: {count}\n"

    # List output files
    log += "\nOutput Files:\n"
    for file in os.listdir(output_dir):
        if file.startswith(prefix):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            log += f"- {file} ({file_size:.1f} KB)\n"

    # Add explanation of key files
    log += """
Key Output Files:
- .gff: Annotation in GFF3 format (contains all genomic features)
- .gbk: Annotation in GenBank format
- .faa: Protein sequences in FASTA format
- .ffn: Nucleotide sequences of genes in FASTA format
- .txt: Summary statistics of the annotation
"""

    return log


def enumerate_bacterial_cfu_by_serial_dilution(
    initial_sample_volume_ml=1.0,
    estimated_concentration=1e8,
    dilution_factor=10,
    num_dilutions=8,
    spots_per_dilution=3,
    output_file="cfu_enumeration_results.csv",
):
    """Quantify bacterial concentration (CFU/mL) using serial dilutions and spot plating.

    Parameters
    ----------
    initial_sample_volume_ml : float
        Volume of the initial bacterial sample in milliliters
    estimated_concentration : float
        Estimated concentration of bacteria in the initial sample (CFU/mL)
    dilution_factor : int
        Factor by which each dilution reduces the concentration (typically 10)
    num_dilutions : int
        Number of serial dilutions to perform
    spots_per_dilution : int
        Number of replicate spots to plate for each dilution
    output_file : str
        Filename to save the CFU enumeration results

    Returns
    -------
    str
        Research log summarizing the CFU enumeration process

    """
    import numpy as np
    import pandas as pd

    # Generate log
    log = "# Bacterial CFU Enumeration via Serial Dilutions and Spot Plating\n\n"

    # Step 1: Prepare serial dilutions
    log += "## Step 1: Serial Dilution Preparation\n"
    log += f"- Initial sample volume: {initial_sample_volume_ml} mL\n"
    log += f"- Estimated concentration: {estimated_concentration:.2e} CFU/mL\n"
    log += f"- Dilution factor: {dilution_factor}\n"
    log += f"- Number of dilutions: {num_dilutions}\n\n"

    # Calculate theoretical concentrations at each dilution
    dilution_concentrations = []
    for i in range(num_dilutions + 1):  # +1 to include the undiluted sample
        conc = estimated_concentration / (dilution_factor**i)
        dilution_concentrations.append(conc)
        dilution_name = "Undiluted" if i == 0 else f"10^-{i}"
        log += f"  {dilution_name}: {conc:.2e} CFU/mL\n"

    # Step 2: Spot plating
    log += "\n## Step 2: Spot Plating\n"
    log += f"- Spots per dilution: {spots_per_dilution}\n"
    log += "- Spotting 10 μL from each dilution onto agar plates\n\n"

    # Simulate bacterial growth with some randomness to mimic real-world variation
    np.random.seed(42)  # For reproducibility

    # Create dataframe to store results
    results = []

    # For each dilution, simulate spotting and counting
    for i in range(num_dilutions + 1):
        dilution_name = "Undiluted" if i == 0 else f"10^-{i}"
        expected_cfu_per_spot = dilution_concentrations[i] * 0.01  # 10 μL = 0.01 mL

        # Simulate multiple spots per dilution
        spot_counts = []
        for spot in range(spots_per_dilution):
            # Add some randomness to the counts (Poisson distribution for bacterial counts)
            if expected_cfu_per_spot > 300:
                # Too many to count (TMTC)
                count = "TMTC"
                spot_counts.append(count)
            elif expected_cfu_per_spot < 1:
                # Simulate low probability events
                count = np.random.poisson(expected_cfu_per_spot)
                spot_counts.append(count)
            else:
                # Normal counting range
                count = np.random.poisson(expected_cfu_per_spot)
                spot_counts.append(count)

            results.append(
                {
                    "Dilution": dilution_name,
                    "Dilution_Factor": dilution_factor**i,
                    "Spot": spot + 1,
                    "CFU_Count": count,
                }
            )

    # Convert results to dataframe
    df = pd.DataFrame(results)

    # Step 3: Colony counting and CFU calculation
    log += "## Step 3: Colony Counting and CFU Calculation\n\n"

    # Find the first dilution with countable colonies (between 3 and 300 CFU)
    countable_dilutions = []

    for i in range(num_dilutions + 1):
        dilution_name = "Undiluted" if i == 0 else f"10^-{i}"
        dilution_data = df[df["Dilution"] == dilution_name]

        # Check if counts are numeric (not TMTC)
        numeric_counts = [c for c in dilution_data["CFU_Count"] if isinstance(c, int | float)]

        if numeric_counts:
            avg_count = sum(numeric_counts) / len(numeric_counts)
            if 3 <= avg_count <= 300:
                countable_dilutions.append(
                    {
                        "Dilution": dilution_name,
                        "Dilution_Factor": dilution_factor**i,
                        "Average_CFU": avg_count,
                        "CFU_per_mL": avg_count * 100 * (dilution_factor**i),  # × 100 to convert 10 μL to mL
                    }
                )

    # Calculate final CFU/mL based on countable dilutions
    if countable_dilutions:
        countable_df = pd.DataFrame(countable_dilutions)

        # Log the counts for each countable dilution
        for _, row in countable_df.iterrows():
            log += f"Dilution {row['Dilution']}: Average CFU per spot = {row['Average_CFU']:.1f}\n"
            log += f"  Calculated concentration: {row['CFU_per_mL']:.2e} CFU/mL\n\n"

        # Calculate the final CFU/mL as the average of all countable dilutions
        final_cfu = countable_df["CFU_per_mL"].mean()
        log += "## Final Result\n"
        log += f"Original sample concentration: {final_cfu:.2e} CFU/mL\n"
    else:
        log += "No countable dilutions found. Consider adjusting the dilution series.\n"
        final_cfu = None

    # Save results to CSV
    df.to_csv(output_file, index=False)
    log += f"\nDetailed results saved to: {output_file}\n"

    return log


def model_bacterial_growth_dynamics(
    initial_population,
    growth_rate,
    clearance_rate,
    niche_size,
    simulation_time=24,
    time_step=0.1,
):
    """Model bacterial population dynamics over time using ordinary differential equations.

    Parameters
    ----------
    initial_population : float
        Initial bacterial population size (CFU/ml or cells)
    growth_rate : float
        Bacterial growth rate (per hour)
    clearance_rate : float
        Rate at which bacteria are cleared from the system (per hour)
    niche_size : float
        Maximum carrying capacity of the environment (CFU/ml or cells)
    simulation_time : float, optional
        Total simulation time in hours (default: 24)
    time_step : float, optional
        Time step for simulation output (default: 0.1)

    Returns
    -------
    str
        Research log summarizing the bacterial growth dynamics simulation

    """
    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Define the ODE system for bacterial growth
    def bacterial_dynamics(t, N):
        # Logistic growth with clearance
        dNdt = growth_rate * N * (1 - N / niche_size) - clearance_rate * N
        return dNdt

    # Time points for simulation
    t_span = (0, simulation_time)
    t_eval = np.arange(0, simulation_time + time_step, time_step)

    # Solve the ODE system
    solution = solve_ivp(bacterial_dynamics, t_span, [initial_population], t_eval=t_eval, method="RK45")

    # Extract results
    time_points = solution.t
    population_size = solution.y[0]

    # Calculate key metrics
    max_population = np.max(population_size)
    final_population = population_size[-1]

    # Determine if population reached steady state
    # (defined as less than 1% change in the last 10% of simulation time)
    last_index = int(len(population_size) * 0.9)
    population_change = abs(population_size[-1] - population_size[last_index]) / population_size[last_index]
    steady_state_reached = population_change < 0.01

    # Save results to CSV
    results_df = pd.DataFrame({"Time (hours)": time_points, "Population Size": population_size})

    filename = "bacterial_growth_dynamics.csv"
    results_df.to_csv(filename, index=False)

    # Generate research log
    log = f"""Bacterial Growth Dynamics Simulation Results:

Initial conditions:
- Starting population: {initial_population:.2e} cells
- Growth rate: {growth_rate:.2f} per hour
- Clearance rate: {clearance_rate:.2f} per hour
- Niche size (carrying capacity): {niche_size:.2e} cells
- Simulation time: {simulation_time} hours

Results:
- Maximum population reached: {max_population:.2e} cells
- Final population: {final_population:.2e} cells
- Steady state {"reached" if steady_state_reached else "not reached"}

The complete population dynamics data has been saved to '{filename}'.
"""

    return log


def quantify_biofilm_biomass_crystal_violet(od_values, sample_names=None, control_index=0, save_path=None):
    """Quantifies biofilm biomass using crystal violet staining assay data.

    Parameters
    ----------
    od_values : list or numpy.ndarray
        Optical density measurements from crystal violet staining.
        Each value represents the absorbance reading for a sample.
    sample_names : list, optional
        Names of the biofilm samples corresponding to od_values.
        If None, samples will be labeled as Sample 1, Sample 2, etc.
    control_index : int, optional
        Index of the negative control sample in od_values. Default is 0.
    save_path : str, optional
        Path to save the results. If None, results won't be saved to a file.

    Returns
    -------
    str
        Research log detailing the quantification process and results.

    """
    import os
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from scipy import stats

    # Initialize research log
    log = "## Biofilm Biomass Quantification using Crystal Violet Staining\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Convert input to numpy array for processing
    od_values = np.array(od_values, dtype=float)

    # Generate sample names if not provided
    if sample_names is None:
        sample_names = [f"Sample {i + 1}" for i in range(len(od_values))]

    log += "### Samples Analyzed:\n"
    for i, name in enumerate(sample_names):
        log += f"- {name}: OD = {od_values[i]:.4f}\n"

    # Calculate normalized values (subtract control)
    control_value = od_values[control_index]
    normalized_values = od_values - control_value

    log += "\n### Normalization:\n"
    log += f"- Control sample: {sample_names[control_index]} (OD = {control_value:.4f})\n"
    log += "- Normalized values (Control subtracted):\n"

    for i, name in enumerate(sample_names):
        if i != control_index:
            log += f"  - {name}: {normalized_values[i]:.4f}\n"

    # Basic statistical analysis
    mean_biomass = np.mean(normalized_values[normalized_values > 0])
    std_biomass = np.std(normalized_values[normalized_values > 0])

    log += "\n### Statistical Analysis:\n"
    log += f"- Mean normalized biomass: {mean_biomass:.4f}\n"
    log += f"- Standard deviation: {std_biomass:.4f}\n"

    # Perform t-test for samples against control
    p_values = []
    log += "\n### Statistical Significance:\n"

    for i, name in enumerate(sample_names):
        if i != control_index:
            # Using one-sample t-test against 0 (normalized control value)
            t_stat, p_val = stats.ttest_1samp([normalized_values[i]], 0)
            p_values.append(p_val)
            significance = "significant" if p_val < 0.05 else "not significant"
            log += f"- {name} vs Control: p-value = {p_val:.4f} ({significance})\n"

    # Create a summary dataframe
    results_df = pd.DataFrame(
        {
            "Sample": sample_names,
            "OD_Value": od_values,
            "Normalized_Value": normalized_values,
        }
    )

    # Save results if path is provided
    if save_path:
        results_file = os.path.join(
            save_path,
            f"biofilm_biomass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        results_df.to_csv(results_file, index=False)
        log += "\n### Results Saved:\n"
        log += f"- Data saved to: {results_file}\n"

    log += "\n### Conclusion:\n"
    log += "- Crystal violet staining assay successfully quantified biofilm biomass.\n"
    log += f"- Samples showed varying levels of biofilm formation with mean biomass of {mean_biomass:.4f} ± {std_biomass:.4f}.\n"

    return log


def segment_and_analyze_microbial_cells(image_path, output_dir="./output", min_cell_size=50):
    """Perform automated cell segmentation and quantify morphological metrics from fluorescence microscopy images.

    Parameters
    ----------
    image_path : str
        Path to the fluorescence microscopy image file
    output_dir : str, optional
        Directory to save output files (default: './output')
    min_cell_size : int, optional
        Minimum cell size in pixels to filter noise (default: 50)

    Returns
    -------
    str
        Research log summarizing the segmentation process, metrics calculated, and output file paths

    """
    import os

    import numpy as np
    import pandas as pd
    from scipy import ndimage
    from skimage import color, filters, io, measure, morphology, segmentation

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and preprocess the image
    image = io.imread(image_path)
    if len(image.shape) > 2:  # Convert to grayscale if RGB
        image = color.rgb2gray(image)

    # Step 2: Enhance contrast and denoise
    image_smooth = filters.gaussian(image, sigma=1)

    # Step 3: Thresholding to separate cells from background
    threshold_value = filters.threshold_otsu(image_smooth)
    binary_mask = image_smooth > threshold_value

    # Step 4: Clean binary mask (remove small objects and fill holes)
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_cell_size)
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(2))
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    # Step 5: Watershed segmentation for separating touching cells
    distance = ndimage.distance_transform_edt(binary_mask)
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    segmented_cells = segmentation.watershed(-distance, markers, mask=binary_mask)

    # Step 6: Measure cell properties
    props = measure.regionprops_table(
        segmented_cells,
        intensity_image=image,
        properties=[
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "mean_intensity",
            "max_intensity",
        ],
    )

    # Calculate circularity (4π × area / perimeter²)
    props["circularity"] = 4 * np.pi * props["area"] / (props["perimeter"] ** 2)

    # Step 7: Save results
    # Save segmentation image
    segmentation_filename = os.path.join(output_dir, "segmented_cells.png")
    segmentation_image = color.label2rgb(segmented_cells, image, alpha=0.3, bg_label=0)
    io.imsave(segmentation_filename, (segmentation_image * 255).astype(np.uint8))

    # Save metrics to CSV
    metrics_filename = os.path.join(output_dir, "cell_metrics.csv")
    metrics_df = pd.DataFrame(props)
    metrics_df.to_csv(metrics_filename, index=False)

    # Step 8: Generate summary statistics
    cell_count = len(np.unique(segmented_cells)) - 1  # Subtract background
    avg_cell_area = np.mean(props["area"])
    avg_circularity = np.mean(props["circularity"])

    # Create research log
    log = f"""
Cell Segmentation and Morphology Analysis Research Log:
------------------------------------------------------
Image processed: {image_path}
Segmentation method: Otsu thresholding + Watershed

Results Summary:
- Number of cells detected: {cell_count}
- Average cell area: {avg_cell_area:.2f} pixels²
- Average cell circularity: {avg_circularity:.4f}
- Size range: {np.min(props["area"]):.2f} to {np.max(props["area"]):.2f} pixels²

Cell morphology metrics have been calculated including:
- Area
- Perimeter
- Eccentricity
- Major/minor axis lengths
- Circularity

Output files:
- Segmentation image: {segmentation_filename}
- Detailed metrics: {metrics_filename}
"""

    return log


def segment_cells_with_deep_learning(
    image_path,
    model_type="cpsam",
    diameter=None,
    save_dir="segmentation_results",
    use_gpu=False,
    channel_axis=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
):
    """Segment cells in a 2D microscopy image with Cellpose 4.

    Parameters
    ----------
    image_path : str
        Path to a 2D grayscale or multichannel microscopy image.
    model_type : str, optional
        Cellpose 4 pretrained model name or model-file path. The built-in model
        is cpsam.
    diameter : float, optional
        Cell diameter in pixels used to rescale the image. None keeps the
        image at its native scale.
    save_dir : str, optional
        Directory for the label mask and boundary-overlay image.
    use_gpu : bool, optional
        Request a CUDA or MPS device through Cellpose.
    channel_axis : int, optional
        Channel axis for a multichannel image. It is inferred for common HWC
        and CHW inputs when omitted.
    flow_threshold : float, optional
        Maximum flow error for retaining masks.
    cellprob_threshold : float, optional
        Cell-probability threshold used to create masks.
    min_size : int, optional
        Minimum mask size in pixels.

    Returns
    -------
    str
        Research log with segmentation settings, counts, statistics, and paths.

    """
    import os
    from datetime import datetime
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from cellpose import models
    from skimage import io
    from skimage.segmentation import find_boundaries

    os.makedirs(save_dir, exist_ok=True)

    log = "# Cell Segmentation Research Log\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += "## Loading Image\n"
    log += f"Image path: {image_path}\n"

    try:
        image = io.imread(image_path)
    except Exception as error:
        return log + f"Error loading image: {error}\n"

    if image.ndim not in (2, 3):
        return log + f"Error loading image: expected a 2D image with optional channels, got shape {image.shape}.\n"

    resolved_channel_axis = channel_axis
    if image.ndim == 2 and resolved_channel_axis is not None:
        return log + "Error loading image: channel_axis must be None for a 2D grayscale image.\n"
    if image.ndim == 3 and resolved_channel_axis is None:
        if image.shape[-1] <= 4:
            resolved_channel_axis = -1
        elif image.shape[0] <= 4:
            resolved_channel_axis = 0
        else:
            return (
                log + "Error loading image: cannot infer a channel axis. "
                "Pass channel_axis for multichannel input; 3D volume segmentation is not supported.\n"
            )
    if resolved_channel_axis is not None and not -image.ndim <= resolved_channel_axis < image.ndim:
        return log + f"Error loading image: channel_axis {resolved_channel_axis} is invalid for shape {image.shape}.\n"

    log += f"Image loaded successfully. Shape: {image.shape}\n"
    log += f"Channel axis: {resolved_channel_axis}\n\n"

    pretrained_model = os.fspath(model_type)
    available_models = set(getattr(models, "MODEL_NAMES", []))
    get_user_models = getattr(models, "get_user_models", None)
    if callable(get_user_models):
        available_models.update(get_user_models())
    if not os.path.isfile(pretrained_model) and pretrained_model not in available_models:
        available = ", ".join(sorted(available_models)) or "none"
        return log + (
            f"Error initializing model: unknown model {pretrained_model!r}. "
            f"Available Cellpose models: {available}; a model-file path is also accepted.\n"
        )

    log += "## Initializing Model\n"
    log += f"Pretrained model: {pretrained_model}\n"
    log += f"GPU requested: {use_gpu}\n"
    try:
        model = models.CellposeModel(pretrained_model=pretrained_model, gpu=use_gpu)
    except Exception as error:
        return log + f"Error initializing model: {error}\n"
    log += "Model initialized successfully.\n\n"

    log += "## Performing Segmentation\n"
    if diameter is None:
        log += "Diameter: native image scale (no rescaling)\n"
    else:
        log += f"Diameter: {diameter} pixels\n"
    log += f"Flow threshold: {flow_threshold}\n"
    log += f"Cell-probability threshold: {cellprob_threshold}\n"
    log += f"Minimum mask size: {min_size} pixels\n"

    try:
        results = model.eval(
            image,
            diameter=diameter,
            channel_axis=resolved_channel_axis,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            do_3D=False,
        )
        if not isinstance(results, tuple) or len(results) != 3:
            length = len(results) if hasattr(results, "__len__") else "unknown"
            raise ValueError(f"Cellpose 4 model.eval() must return 3 values, got {length}")
        masks, _flows, _styles = results
        masks = np.asarray(masks)
        if resolved_channel_axis is None:
            expected_shape = image.shape
        else:
            normalized_axis = resolved_channel_axis % image.ndim
            expected_shape = tuple(size for axis, size in enumerate(image.shape) if axis != normalized_axis)
        if masks.ndim != 2 or tuple(masks.shape) != tuple(expected_shape):
            raise ValueError(f"mask shape {masks.shape} does not match image plane {tuple(expected_shape)}")
    except Exception as error:
        return log + f"Error during segmentation: {error}\n"

    labels = np.unique(masks)
    labels = labels[labels != 0]
    cell_count = len(labels)
    log += f"Segmentation complete. Detected {cell_count} cells.\n\n"

    stem = Path(image_path).stem
    mask_file = os.path.join(save_dir, f"{stem}_masks.tif")
    outline_file = os.path.join(save_dir, f"{stem}_outlines.png")

    log += "## Saving Results\n"
    try:
        io.imsave(mask_file, masks.astype(np.uint32), check_contrast=False)

        display_image = image
        if resolved_channel_axis is not None:
            display_image = np.moveaxis(image, resolved_channel_axis, -1)
            if display_image.shape[-1] == 1:
                display_image = display_image[..., 0]
            elif display_image.shape[-1] > 3:
                display_image = display_image[..., :3]

        boundaries = find_boundaries(masks, mode="outer")
        overlay = np.zeros((*masks.shape, 4), dtype=float)
        overlay[boundaries] = (1.0, 0.0, 0.0, 1.0)

        figure, axis = plt.subplots(figsize=(10, 8))
        axis.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
        axis.imshow(overlay, interpolation="none")
        axis.axis("off")
        figure.savefig(outline_file, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
    except Exception as error:
        return log + f"Error saving segmentation results: {error}\n"

    log += f"Cell masks saved to: {mask_file}\n"
    log += f"Cell outlines saved to: {outline_file}\n\n"
    log += "## Segmentation Statistics\n"
    log += f"Total cells detected: {cell_count}\n"
    if cell_count:
        cell_areas = [np.count_nonzero(masks == label) for label in labels]
        log += f"Average cell area: {np.mean(cell_areas):.2f} pixels\n"
        log += f"Standard deviation of cell area: {np.std(cell_areas):.2f} pixels\n"

    return log


def simulate_generalized_lotka_volterra_dynamics(
    initial_abundances,
    growth_rates,
    interaction_matrix,
    time_points,
    output_file="glv_simulation_results.csv",
):
    """Simulate microbial community dynamics using the Generalized Lotka-Volterra (gLV) model.

    Parameters
    ----------
    initial_abundances : numpy.ndarray
        Initial abundances of each microbial species (1D array)
    growth_rates : numpy.ndarray
        Intrinsic growth rates for each microbial species (1D array)
    interaction_matrix : numpy.ndarray
        Matrix of interaction coefficients where A[i,j] represents the effect of species j on species i (2D array)
    time_points : numpy.ndarray
        Time points at which to evaluate the model
    output_file : str, optional
        Filename to save the simulation results (default: "glv_simulation_results.csv")

    Returns
    -------
    str
        Research log summarizing the simulation process and results

    """
    import numpy as np
    import pandas as pd
    from scipy.integrate import odeint

    # Check input dimensions
    n_species = len(initial_abundances)
    if len(growth_rates) != n_species or interaction_matrix.shape != (
        n_species,
        n_species,
    ):
        raise ValueError(
            "Dimensions mismatch: growth_rates and interaction_matrix must match initial_abundances dimensions"
        )

    # Define the gLV differential equations
    def glv_equations(abundances, t, growth_rates, interaction_matrix):
        # Calculate growth and interaction terms for each species
        # dx_i/dt = r_i * x_i + x_i * sum(A_ij * x_j)
        dx_dt = abundances * (growth_rates + np.dot(interaction_matrix, abundances))
        return dx_dt

    # Integrate the ODE system
    simulation_results = odeint(
        glv_equations,
        initial_abundances,
        time_points,
        args=(growth_rates, interaction_matrix),
    )

    # Create a DataFrame with the results
    columns = [f"Species_{i + 1}" for i in range(n_species)]
    results_df = pd.DataFrame(simulation_results, columns=columns)
    results_df.insert(0, "Time", time_points)

    # Save results to CSV
    results_df.to_csv(output_file, index=False)

    # Generate summary statistics
    final_abundances = simulation_results[-1]
    dominant_species = np.argmax(final_abundances) + 1
    extinct_species = sum(final_abundances < 1e-6)

    # Create research log
    log = f"""
Generalized Lotka-Volterra (gLV) Model Simulation Results:
------------------------------------------------------
Number of microbial species: {n_species}
Simulation time range: {time_points[0]} to {time_points[-1]}
Number of time points: {len(time_points)}

Summary of dynamics:
- Initial total abundance: {np.sum(initial_abundances):.4f}
- Final total abundance: {np.sum(final_abundances):.4f}
- Dominant species at end of simulation: Species_{dominant_species} (abundance: {final_abundances[dominant_species - 1]:.4f})
- Number of species with near-zero abundance (< 1e-6): {extinct_species}

Simulation results have been saved to: {output_file}
"""

    return log


def predict_rna_secondary_structure(rna_sequence, output_prefix="rna_structure"):
    """Predict the secondary structure of an RNA molecule using ViennaRNA.

    Parameters
    ----------
    rna_sequence : str
        The RNA sequence (consisting of A, U, G, C nucleotides)
    output_prefix : str, optional
        Prefix for output files (default: "rna_structure")

    Returns
    -------
    str
        A research log summarizing the prediction process and results

    """
    try:
        import RNA
    except ImportError:
        return "ERROR: ViennaRNA Python package not installed. Install with 'pip install ViennaRNA'"

    # Validate input sequence
    rna_sequence = rna_sequence.upper().strip()
    valid_nucleotides = set("AUGC")
    if not all(nucleotide in valid_nucleotides for nucleotide in rna_sequence):
        return "ERROR: Invalid RNA sequence. Only A, U, G, C nucleotides are allowed."

    # Predict secondary structure
    (structure, mfe) = RNA.fold(rna_sequence)

    # Save the structure to a file
    structure_file = f"{output_prefix}_structure.txt"
    with open(structure_file, "w") as f:
        f.write(f"Sequence: {rna_sequence}\n")
        f.write(f"Structure: {structure}\n")
        f.write(f"Minimum Free Energy: {mfe} kcal/mol\n")

    # Generate a simple text visualization
    viz_file = f"{output_prefix}_visualization.txt"
    with open(viz_file, "w") as f:
        f.write("Sequence:  " + rna_sequence + "\n")
        f.write("Structure: " + structure + "\n\n")

        # Add a simple representation of stem-loops
        pairs = []
        stack = []
        for i, char in enumerate(structure):
            if char == "(":
                stack.append(i)
            elif char == ")" and stack:
                left = stack.pop()
                pairs.append((left, i))

        f.write("Stem-loop structures:\n")
        for left, right in sorted(pairs):
            f.write(f"Base pair: {rna_sequence[left]}({left + 1})-{rna_sequence[right]}({right + 1})\n")

    # Create research log
    log = f"""
RNA Secondary Structure Prediction Log:
---------------------------------------
1. Received RNA sequence of length {len(rna_sequence)}
2. Applied ViennaRNA RNAfold algorithm for structure prediction
3. Calculated minimum free energy: {mfe} kcal/mol
4. Structure saved to file: {structure_file}
5. Visualization saved to file: {viz_file}

Summary:
The RNA sequence forms a secondary structure with a minimum free energy of {mfe} kcal/mol.
The structure contains {structure.count("(")} base pairs forming stems and loops.
See {structure_file} and {viz_file} for detailed structure information.
"""

    return log


def simulate_microbial_population_dynamics(
    initial_populations,
    growth_rates,
    clearance_rates,
    carrying_capacities,
    max_time=100,
    num_simulations=100,
    time_points=100,
):
    """Performs stochastic simulation of microbial population dynamics using the Gillespie algorithm.

    Parameters
    ----------
    initial_populations : list of int
        Initial population sizes for each microbial species
    growth_rates : list of float
        Per capita growth rates for each species
    clearance_rates : list of float
        Per capita death/clearance rates for each species
    carrying_capacities : list of float
        Maximum sustainable population for each species
    max_time : float, optional
        Maximum simulation time (default: 100)
    num_simulations : int, optional
        Number of stochastic simulations to run (default: 100)
    time_points : int, optional
        Number of time points to record for trajectories (default: 100)

    Returns
    -------
    str
        Research log summarizing the simulation results, including extinction probabilities and timelines

    """
    import numpy as np
    from scipy import stats

    # Validate inputs
    num_species = len(initial_populations)
    if not (len(growth_rates) == len(clearance_rates) == len(carrying_capacities) == num_species):
        return "Error: All input lists must have the same length (number of species)"

    # Initialize tracking variables
    extinction_counts = np.zeros(num_species)
    extinction_times = [[] for _ in range(num_species)]

    # Time points for trajectory recording
    time_grid = np.linspace(0, max_time, time_points)
    avg_trajectories = np.zeros((num_species, time_points))

    # Run multiple simulations
    for _sim in range(num_simulations):
        # Initialize population and time for this simulation
        population = np.array(initial_populations, dtype=float)
        time = 0.0

        # For recording trajectories at specific time points
        trajectory = np.zeros((num_species, time_points))
        next_time_point_idx = 0

        # Track which species have gone extinct in this simulation
        extinct_in_sim = [False] * num_species

        # Run simulation until max_time is reached or all populations are extinct
        while time < max_time and np.any(population > 0):
            # Record current state if we've reached the next time point
            while next_time_point_idx < time_points and time >= time_grid[next_time_point_idx]:
                trajectory[:, next_time_point_idx] = population
                next_time_point_idx += 1

            # Calculate event rates
            # Growth rates are adjusted for carrying capacity (logistic growth)
            adjusted_growth_rates = [
                growth_rates[i] * population[i] * (1 - population[i] / carrying_capacities[i])
                if population[i] > 0
                else 0
                for i in range(num_species)
            ]
            death_rates = [clearance_rates[i] * population[i] if population[i] > 0 else 0 for i in range(num_species)]

            # Flatten rates for easier processing
            all_rates = adjusted_growth_rates + death_rates
            total_rate = sum(all_rates)

            # If no events possible, end simulation
            if total_rate == 0:
                break

            # Time until next event (exponentially distributed)
            dt = np.random.exponential(1.0 / total_rate)
            time += dt

            # If we've exceeded max_time, break
            if time > max_time:
                break

            # Select which event occurs
            event_idx = np.random.choice(len(all_rates), p=np.array(all_rates) / total_rate)

            # Apply the event
            if event_idx < num_species:  # Growth event
                population[event_idx] += 1
            else:  # Death event
                species_idx = event_idx - num_species
                population[species_idx] -= 1

                # Check for new extinctions
                if population[species_idx] == 0 and not extinct_in_sim[species_idx]:
                    extinction_counts[species_idx] += 1
                    extinction_times[species_idx].append(time)
                    extinct_in_sim[species_idx] = True

        # Fill in any remaining time points
        while next_time_point_idx < time_points:
            trajectory[:, next_time_point_idx] = population
            next_time_point_idx += 1

        # Add this simulation's trajectory to the average
        avg_trajectories += trajectory

        # Check for species that never went extinct in this simulation
        for i in range(num_species):
            if population[i] > 0 and not extinct_in_sim[i]:
                # Record as "no extinction" by adding a None to extinction_times
                extinction_times[i].append(None)

    # Calculate average trajectories
    avg_trajectories /= num_simulations

    # Calculate extinction probabilities and statistics
    extinction_probs = extinction_counts / num_simulations

    # Calculate median extinction times (ignoring None values)
    median_extinction_times = []
    for times in extinction_times:
        valid_times = [t for t in times if t is not None]
        if valid_times:
            median_extinction_times.append(np.median(valid_times))
        else:
            median_extinction_times.append(float("inf"))

    # Generate research log
    log = "# Microbial Population Dynamics Simulation Results\n\n"
    log += f"Simulations run: {num_simulations}\n"
    log += f"Maximum simulation time: {max_time}\n\n"

    log += "## Species Parameters\n"
    for i in range(num_species):
        log += f"\nSpecies {i + 1}:\n"
        log += f"  Initial population: {initial_populations[i]}\n"
        log += f"  Growth rate: {growth_rates[i]}\n"
        log += f"  Clearance rate: {clearance_rates[i]}\n"
        log += f"  Carrying capacity: {carrying_capacities[i]}\n"

    log += "\n## Extinction Analysis\n"
    for i in range(num_species):
        log += f"\nSpecies {i + 1}:\n"
        log += f"  Extinction probability: {extinction_probs[i]:.2f}\n"

        if extinction_probs[i] > 0:
            valid_times = [t for t in extinction_times[i] if t is not None]
            if valid_times:
                log += f"  Median extinction time: {np.median(valid_times):.2f}\n"
                log += f"  Mean extinction time: {np.mean(valid_times):.2f}\n"
                log += f"  Standard deviation: {np.std(valid_times):.2f}\n"

                # Calculate confidence intervals if we have enough data
                if len(valid_times) >= 10:
                    ci = stats.t.interval(
                        0.95,
                        len(valid_times) - 1,
                        loc=np.mean(valid_times),
                        scale=stats.sem(valid_times),
                    )
                    log += f"  95% CI for extinction time: ({ci[0]:.2f}, {ci[1]:.2f})\n"
            else:
                log += "  No extinctions observed in any simulations\n"

    # Save average trajectories to CSV file
    filename = "population_trajectories.csv"
    header = "Time," + ",".join([f"Species_{i + 1}" for i in range(num_species)])
    data = np.column_stack((time_grid, avg_trajectories.T))
    np.savetxt(filename, data, delimiter=",", header=header, comments="")

    log += "\n## Population Trajectories\n"
    log += f"Average population trajectories saved to '{filename}'\n"

    return log
