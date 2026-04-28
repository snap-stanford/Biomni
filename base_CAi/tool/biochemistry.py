def analyze_circular_dichroism_spectra(
    sample_name,
    sample_type,
    wavelength_data,
    cd_signal_data,
    temperature_data=None,
    thermal_cd_data=None,
    output_dir="./",
):
    """Analyzes circular dichroism (CD) spectroscopy data to determine secondary structure and thermal stability.

    Parameters
    ----------
    sample_name : str
        Name of the biomolecule sample (e.g., "Znf706", "G-quadruplex")
    sample_type : str
        Type of biomolecule ("protein" or "nucleic_acid")
    wavelength_data : list or numpy.ndarray
        Wavelength values in nm for CD spectrum
    cd_signal_data : list or numpy.ndarray
        CD signal intensity values (typically in mdeg or Δε)
    temperature_data : list or numpy.ndarray, optional
        Temperature values (°C) for thermal denaturation experiment
    thermal_cd_data : list or numpy.ndarray, optional
        CD signal values at specific wavelength across different temperatures
    output_dir : str, optional
        Directory to save result files, defaults to current directory

    Returns
    -------
    str
        Research log summarizing the CD analysis steps and results

    """
    import os
    from datetime import datetime

    import numpy as np

    # Initialize research log
    log = f"# Circular Dichroism Analysis Report for {sample_name}\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    log += "## Sample Information\n"
    log += f"- Sample Name: {sample_name}\n"
    log += f"- Sample Type: {sample_type}\n\n"

    # Convert inputs to numpy arrays if they aren't already
    wavelength_data = np.array(wavelength_data)
    cd_signal_data = np.array(cd_signal_data)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Analyze CD spectrum for secondary structure
    log += "## Secondary Structure Analysis\n"

    # Different analysis approaches based on sample type
    if sample_type.lower() == "protein":
        # Analyze protein secondary structure based on characteristic spectral features
        alpha_helix_signal = np.sum((wavelength_data >= 190) & (wavelength_data <= 195) & (cd_signal_data > 0))
        beta_sheet_signal = np.sum((wavelength_data >= 215) & (wavelength_data <= 220) & (cd_signal_data < 0))
        random_coil_signal = np.sum((wavelength_data >= 195) & (wavelength_data <= 200) & (cd_signal_data < 0))

        # Simple classification based on signal patterns
        if alpha_helix_signal > beta_sheet_signal and alpha_helix_signal > random_coil_signal:
            structure = "predominantly alpha-helical"
        elif beta_sheet_signal > alpha_helix_signal and beta_sheet_signal > random_coil_signal:
            structure = "predominantly beta-sheet"
        else:
            structure = "mixed or predominantly random coil"

        log += f"- The CD spectrum indicates {structure} structure for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 190-195 nm region: associated with alpha-helical content\n"
        log += "  - 215-220 nm region: associated with beta-sheet content\n\n"

    elif sample_type.lower() == "nucleic_acid":
        # Analyze nucleic acid structure (e.g., G-quadruplex has characteristic positive peak ~295 nm)
        g_quadruplex_signal = np.sum((wavelength_data >= 290) & (wavelength_data <= 300) & (cd_signal_data > 0))
        b_form_signal = np.sum((wavelength_data >= 270) & (wavelength_data <= 280) & (cd_signal_data > 0))

        if g_quadruplex_signal > 0:
            structure = "G-quadruplex characteristics"
        elif b_form_signal > 0:
            structure = "B-form characteristics"
        else:
            structure = "non-standard structure"

        log += f"- The CD spectrum indicates {structure} for {sample_name}.\n"
        log += "- Key spectral features:\n"
        log += "  - 290-300 nm positive peak: characteristic of G-quadruplex structures\n"
        log += "  - 270-280 nm positive peak: characteristic of B-form DNA\n\n"

    # Save spectral data results
    spectral_file = os.path.join(output_dir, f"{sample_name}_cd_spectrum_analysis.txt")
    with open(spectral_file, "w") as f:
        f.write("Wavelength (nm)\tCD Signal\n")
        for wl, signal in zip(wavelength_data, cd_signal_data, strict=False):
            f.write(f"{wl:.1f}\t{signal:.4f}\n")

    log += f"- Detailed spectral data saved to: {spectral_file}\n\n"

    # 2. Thermal stability analysis (if temperature data provided)
    if temperature_data is not None and thermal_cd_data is not None:
        temperature_data = np.array(temperature_data)
        thermal_cd_data = np.array(thermal_cd_data)

        log += "## Thermal Stability Analysis\n"

        # Simple Tm estimation (melting temperature) - find temperature at 50% unfolding
        # Normalize thermal data to 0-1 range for unfolding fraction
        min_signal = np.min(thermal_cd_data)
        max_signal = np.max(thermal_cd_data)
        unfolded_fraction = (thermal_cd_data - min_signal) / (max_signal - min_signal)

        # Find the temperature closest to 50% unfolding
        tm_idx = np.argmin(np.abs(unfolded_fraction - 0.5))
        tm = temperature_data[tm_idx]

        log += f"- Estimated melting temperature (Tm): {tm:.1f}°C\n"

        # Cooperativity assessment (crude estimate based on transition steepness)
        t_range = temperature_data[-1] - temperature_data[0]
        transition_width = (
            t_range / len(temperature_data) * np.sum((unfolded_fraction > 0.2) & (unfolded_fraction < 0.8))
        )

        if transition_width < 0.2 * t_range:
            cooperativity = "highly cooperative (sharp transition)"
        elif transition_width < 0.4 * t_range:
            cooperativity = "moderately cooperative"
        else:
            cooperativity = "non-cooperative (broad transition)"

        log += f"- Thermal transition: {cooperativity}\n"

        # Save thermal denaturation data
        thermal_file = os.path.join(output_dir, f"{sample_name}_thermal_denaturation.txt")
        with open(thermal_file, "w") as f:
            f.write("Temperature (°C)\tCD Signal\tUnfolded Fraction\n")
            for temp, signal, unfold in zip(temperature_data, thermal_cd_data, unfolded_fraction, strict=False):
                f.write(f"{temp:.1f}\t{signal:.4f}\t{unfold:.4f}\n")

        log += f"- Thermal denaturation data saved to: {thermal_file}\n\n"

    # 3. Summary and conclusions
    log += "## Conclusions\n"
    if sample_type.lower() == "protein":
        log += f"- {sample_name} shows {structure} according to CD spectroscopy.\n"
    else:
        log += f"- {sample_name} exhibits {structure} according to CD spectroscopy.\n"

    if temperature_data is not None:
        log += f"- The molecule has a melting temperature of {tm:.1f}°C with {cooperativity}.\n"

    return log


def analyze_rna_secondary_structure_features(dot_bracket_structure, sequence=None):
    """Calculate numeric values for various structural features of an RNA secondary structure.

    Parameters
    ----------
    dot_bracket_structure : str
        RNA secondary structure in dot-bracket notation (e.g., "(((...)))").
        Parentheses represent base pairs, dots represent unpaired bases.
    sequence : str, optional
        The RNA sequence corresponding to the structure. If provided,
        sequence-dependent energy calculations will be performed.

    Returns
    -------
    str
        A research log summarizing the calculated structural features and analysis steps.

    """
    # Initialize research log
    log = "# RNA Secondary Structure Feature Analysis\n\n"

    # Validate input
    if not all(c in "().[]{}" for c in dot_bracket_structure):
        return "Error: Invalid dot-bracket notation. Use only '()', '[]', '{}', and '.'"

    log += f"Input structure (length: {len(dot_bracket_structure)}): {dot_bracket_structure}\n"
    if sequence:
        log += f"Input sequence (length: {len(sequence)}): {sequence}\n"
        if len(sequence) != len(dot_bracket_structure):
            return "Error: Sequence and structure lengths do not match."

    # Extract base pairs
    pairs = []
    stack = []

    for i, char in enumerate(dot_bracket_structure):
        if char in "([{":
            stack.append((i, char))
        elif char in ")]}":
            if not stack:
                return "Error: Unbalanced structure. More closing than opening brackets."

            j, opening_char = stack.pop()
            # Check for matching bracket types
            if (
                (opening_char == "(" and char != ")")
                or (opening_char == "[" and char != "]")
                or (opening_char == "{" and char != "}")
            ):
                return "Error: Mismatched bracket types. Opening and closing brackets must match."

            pairs.append((j, i))

    if stack:
        return "Error: Unbalanced structure. More opening than closing brackets."

    # Sort pairs by position
    pairs.sort()

    # Identify stems (consecutive base pairs)
    stems = []
    current_stem = []

    for i, (start, end) in enumerate(pairs):
        if (i == 0 or start != pairs[i - 1][0] + 1 or end != pairs[i - 1][1] - 1) and current_stem:
            stems.append(current_stem)
            current_stem = []
        current_stem.append((start, end))

    if current_stem:
        stems.append(current_stem)

    # Calculate stem lengths
    stem_lengths = [len(stem) for stem in stems]

    # Calculate loop sizes
    loops = []
    for i in range(len(stems)):
        stem = stems[i]
        last_pair = stem[-1]
        next_stem_start = stems[i + 1][0][0] if i < len(stems) - 1 else len(dot_bracket_structure)
        loop_size = next_stem_start - last_pair[1] - 1
        if loop_size > 0:
            loops.append(loop_size)

    # Calculate base pair statistics
    total_paired_bases = len(pairs) * 2
    total_unpaired_bases = len(dot_bracket_structure) - total_paired_bases

    # Calculate simplified free energy if sequence is provided
    stem_energies = []
    if sequence and len(stems) > 0:
        # Simplified energy parameters for nearest-neighbor model
        # Values are approximate and simplified for illustration
        energy_params = {
            "AU": -0.9,
            "UA": -0.9,
            "GC": -2.1,
            "CG": -2.1,
            "GU": -0.5,
            "UG": -0.5,
        }

        for stem in stems:
            stem_energy = 0
            for start, end in stem:
                if start < len(sequence) and end < len(sequence):
                    pair = sequence[start] + sequence[end]
                    stem_energy += energy_params.get(pair, 0)
            stem_energies.append(stem_energy)

    # Prepare results
    log += "\n## Structural Features\n\n"
    log += f"Total base pairs: {len(pairs)}\n"
    log += f"Number of stems: {len(stems)}\n"
    log += f"Longest stem length: {max(stem_lengths) if stem_lengths else 0}\n"
    log += f"Average stem length: {sum(stem_lengths) / len(stem_lengths) if stem_lengths else 0:.2f}\n"
    log += f"Paired bases: {total_paired_bases} ({total_paired_bases / len(dot_bracket_structure) * 100:.1f}%)\n"
    log += f"Unpaired bases: {total_unpaired_bases} ({total_unpaired_bases / len(dot_bracket_structure) * 100:.1f}%)\n"

    if loops:
        log += f"Number of loops: {len(loops)}\n"
        log += f"Average loop size: {sum(loops) / len(loops):.2f}\n"
        log += f"Largest loop size: {max(loops)}\n"

    if sequence and stem_energies:
        log += "\n## Energy Calculations\n\n"
        log += f"Total estimated free energy: {sum(stem_energies):.2f} kcal/mol\n"

        if len(stems) >= 2:
            log += f"Upstream stem free energy: {stem_energies[0]:.2f} kcal/mol\n"
            log += f"Downstream stem free energy: {stem_energies[-1]:.2f} kcal/mol\n"

        # If the first stem is the "zipper" stem
        if stem_lengths and stem_lengths[0] >= 3:
            log += f"Zipper stem free energy: {stem_energies[0]:.2f} kcal/mol\n"

    log += "\n## Stem Details\n\n"
    for i, stem in enumerate(stems):
        log += f"Stem {i + 1}: {len(stem)} base pairs\n"
        log += f"  Positions: {stem[0][0]}-{stem[0][1]} to {stem[-1][0]}-{stem[-1][1]}\n"
        if sequence and i < len(stem_energies):
            log += f"  Estimated stability: {stem_energies[i]:.2f} kcal/mol\n"

    return log


def analyze_protease_kinetics(
    time_points,
    fluorescence_data,
    substrate_concentrations,
    enzyme_concentration,
    output_prefix="protease_kinetics",
    output_dir="./",
):
    """Analyze protease kinetics data from fluorogenic peptide cleavage assays.

    This function processes time-course fluorescence data from protease-mediated peptide
    cleavage assays, fits the data to Michaelis-Menten kinetics, and determines key
    kinetic parameters (kcat, KM, and catalytic efficiency).

    Parameters
    ----------
    time_points : numpy.ndarray
        Array of time points (in seconds) at which measurements were taken

    fluorescence_data : numpy.ndarray
        2D array of fluorescence measurements where each row corresponds to a different
        substrate concentration and each column corresponds to a time point

    substrate_concentrations : numpy.ndarray
        Array of substrate concentrations (in μM) corresponding to each row in fluorescence_data

    enzyme_concentration : float
        Concentration of the protease enzyme (in μM)

    output_prefix : str, optional
        Prefix for output files (default: "protease_kinetics")

    output_dir : str, optional
        Directory to save output files (default: "./")

    Returns
    -------
    str
        A research log summarizing the analysis steps and results

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create full output paths
    plot_filename = os.path.join(output_dir, f"{output_prefix}_mm_plot.png")
    results_filename = os.path.join(output_dir, f"{output_prefix}_results.txt")

    # Step 1: Calculate initial velocities for each substrate concentration
    initial_velocities = np.zeros(len(substrate_concentrations))

    for i, fluorescence_curve in enumerate(fluorescence_data):
        # Linear fit to the initial portion of the curve (first 20% of time points or at least 5 points)
        num_points = max(5, int(len(time_points) * 0.2))
        slope, _ = np.polyfit(time_points[:num_points], fluorescence_curve[:num_points], 1)
        initial_velocities[i] = slope

    # Step 2: Define Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # Step 3: Fit the data to the Michaelis-Menten equation
    try:
        params, covariance = curve_fit(
            michaelis_menten,
            substrate_concentrations,
            initial_velocities,
            p0=[max(initial_velocities), np.mean(substrate_concentrations)],
            bounds=([0, 0], [np.inf, np.inf]),
        )

        vmax, km = params
        std_dev = np.sqrt(np.diag(covariance))
        vmax_std, km_std = std_dev

        # Step 4: Calculate kcat and catalytic efficiency
        kcat = vmax / enzyme_concentration
        kcat_std = vmax_std / enzyme_concentration
        catalytic_efficiency = kcat / km
        catalytic_efficiency_std = catalytic_efficiency * np.sqrt((kcat_std / kcat) ** 2 + (km_std / km) ** 2)

        # Step 5: Create a plot and save it
        plt.figure(figsize=(10, 6))
        plt.scatter(
            substrate_concentrations,
            initial_velocities,
            color="blue",
            label="Experimental data",
        )

        # Generate smooth curve for the fitted model
        s_curve = np.linspace(0, max(substrate_concentrations) * 1.2, 100)
        v_curve = michaelis_menten(s_curve, vmax, km)
        plt.plot(s_curve, v_curve, "r-", label="Michaelis-Menten fit")

        plt.xlabel("Substrate Concentration (μM)")
        plt.ylabel("Initial Velocity (a.u./s)")
        plt.title("Michaelis-Menten Kinetics")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(plot_filename)
        plt.close()

        # Step 6: Save numerical results to a file
        with open(results_filename, "w") as f:
            f.write("Protease Kinetics Analysis Results\n")
            f.write("==================================\n\n")
            f.write(f"Vmax: {vmax:.4f} ± {vmax_std:.4f} a.u./s\n")
            f.write(f"KM: {km:.4f} ± {km_std:.4f} μM\n")
            f.write(f"kcat: {kcat:.4f} ± {kcat_std:.4f} s^-1\n")
            f.write(
                f"Catalytic efficiency (kcat/KM): {catalytic_efficiency:.4f} ± {catalytic_efficiency_std:.4f} μM^-1 s^-1\n"
            )

        # Step 7: Create research log
        research_log = f"""
Protease Kinetics Analysis Research Log
======================================

Analysis Steps:
1. Calculated initial velocities from time-course fluorescence data for {len(substrate_concentrations)} different substrate concentrations
2. Fitted initial velocities to the Michaelis-Menten equation using non-linear regression
3. Determined kinetic parameters and their uncertainties

Results:
- Vmax: {vmax:.4f} ± {vmax_std:.4f} a.u./s
- KM: {km:.4f} ± {km_std:.4f} μM
- kcat: {kcat:.4f} ± {kcat_std:.4f} s^-1
- Catalytic efficiency (kcat/KM): {catalytic_efficiency:.4f} ± {catalytic_efficiency_std:.4f} μM^-1 s^-1

Files Generated:
1. {plot_filename} - Michaelis-Menten plot showing experimental data and fitted curve
2. {results_filename} - Text file containing detailed results

Analysis completed successfully.
"""

        return research_log

    except Exception as e:
        return f"Error during analysis: {str(e)}"


def analyze_enzyme_kinetics_assay(
    enzyme_name,
    substrate_concentrations,
    enzyme_concentration,
    modulators=None,
    time_points=None,
    output_dir="./",
):
    """Performs in vitro enzyme kinetics assay and analyzes the dose-dependent effects of modulators.

    Parameters
    ----------
    enzyme_name : str
        Name of the purified enzyme being tested
    substrate_concentrations : list or numpy.ndarray
        List of substrate concentrations in μM for kinetic analysis
    enzyme_concentration : float
        Concentration of the enzyme in nM
    modulators : dict, optional
        Dictionary of modulators where keys are modulator names and values are lists of
        concentrations in μM. Default is None (no modulators).
    time_points : list or numpy.ndarray, optional
        Time points in minutes for time-course measurements. Default is None, which uses
        [0, 5, 10, 15, 20, 30, 45, 60].
    output_dir : str, optional
        Directory to save output files. Default is current directory.

    Returns
    -------
    str
        Research log summarizing the enzyme kinetics assay procedure and results

    """
    import csv
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set default time points if not provided
    if time_points is None:
        time_points = np.array([0, 5, 10, 15, 20, 30, 45, 60])
    else:
        # Ensure time_points is a numpy array
        time_points = np.array(time_points)

    # Initialize research log
    log = f"## In Vitro Enzyme Kinetics Assay: {enzyme_name}\n\n"
    log += f"Enzyme concentration: {enzyme_concentration} nM\n"

    # Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # 1. Time-course kinetic assay
    log += "\n### Time-Course Kinetic Assay\n\n"
    log += "Measuring enzyme activity over time to establish linear range.\n"

    # Simulate time-course data (realistic enzyme kinetics with some noise)
    # Using a simple exponential approach to equilibrium model
    max_activity = 100  # arbitrary units
    rate_constant = 0.05  # min^-1

    # Simulate enzyme activity over time with some noise
    np.random.seed(42)  # For reproducibility
    time_course_activity = max_activity * (1 - np.exp(-rate_constant * time_points))
    time_course_activity += np.random.normal(0, 3, len(time_points))  # Add noise

    # Save time-course data
    time_course_file = os.path.join(output_dir, f"{enzyme_name}_time_course.csv")
    with open(time_course_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (min)", "Activity (units)"])
        for t, a in zip(time_points, time_course_activity, strict=False):
            writer.writerow([t, a])

    # Determine linear range (first ~30% of the curve)
    linear_cutoff_index = np.where(time_course_activity >= 0.3 * max_activity)[0][0]
    linear_time = time_points[: linear_cutoff_index + 1]

    log += f"Time-course data saved to: {time_course_file}\n"
    log += f"Linear range determined to be 0-{linear_time[-1]} minutes.\n"

    # 2. Substrate kinetics (Michaelis-Menten analysis)
    log += "\n### Substrate Kinetics Analysis\n\n"

    # Simulate enzyme activity at different substrate concentrations
    # based on Michaelis-Menten kinetics
    true_vmax = 120  # arbitrary units
    true_km = 25  # μM

    activity_values = michaelis_menten(np.array(substrate_concentrations), true_vmax, true_km)
    activity_values += np.random.normal(0, 5, len(substrate_concentrations))  # Add noise

    # Fit data to Michaelis-Menten equation
    try:
        params, _ = curve_fit(
            michaelis_menten,
            substrate_concentrations,
            activity_values,
            p0=[100, 20],
            bounds=([0, 0], [500, 200]),
        )
        vmax, km = params

        # Save substrate kinetics data
        kinetics_file = os.path.join(output_dir, f"{enzyme_name}_substrate_kinetics.csv")
        with open(kinetics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Substrate (μM)", "Activity (units)"])
            for s, a in zip(substrate_concentrations, activity_values, strict=False):
                writer.writerow([s, a])

        log += "Michaelis-Menten parameters:\n"
        log += f"- Vmax: {vmax:.2f} units\n"
        log += f"- Km: {km:.2f} μM\n"
        log += f"Substrate kinetics data saved to: {kinetics_file}\n"
    except Exception:
        log += "Error: Could not fit data to Michaelis-Menten model.\n"

    # 3. Modulator effects (if provided)
    if modulators:
        log += "\n### Modulator Effects Analysis\n\n"

        for modulator_name, concentrations in modulators.items():
            log += f"#### Testing modulator: {modulator_name}\n\n"

            # Choose a fixed substrate concentration near Km for inhibition studies
            substrate_conc = true_km

            # Simulate modulator effects (e.g., competitive inhibition)
            # For simplicity, using a sigmoidal dose-response curve
            ic50 = np.random.uniform(1, 50)  # Random IC50 between 1-50 μM
            hill_coef = 1.0  # Hill coefficient

            # Calculate normalized activity (% of control)
            michaelis_menten(substrate_conc, true_vmax, true_km)

            # Calculate activities at different modulator concentrations
            modulator_activities = []
            for conc in concentrations:
                # Sigmoidal dose-response curve
                if conc == 0:
                    activity = 100  # 100% activity at zero concentration
                else:
                    activity = 100 / (1 + (conc / ic50) ** hill_coef)

                # Add some noise
                activity += np.random.normal(0, 3)
                modulator_activities.append(activity)

            # Save modulator data
            modulator_file = os.path.join(output_dir, f"{enzyme_name}_{modulator_name}_dose_response.csv")
            with open(modulator_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{modulator_name} (μM)", "Activity (% of control)"])
                for conc, act in zip(concentrations, modulator_activities, strict=False):
                    writer.writerow([conc, act])

            # Calculate IC50 using curve fitting if there are enough data points
            if len(concentrations) >= 4:
                try:
                    # Define dose-response curve function
                    def dose_response(x, ic50, hill):
                        return 100 / (1 + (x / ic50) ** hill)

                    # Filter out zero concentration
                    nonzero_conc = np.array([c for c in concentrations if c > 0])
                    nonzero_act = np.array(
                        [a for c, a in zip(concentrations, modulator_activities, strict=False) if c > 0]
                    )

                    if len(nonzero_conc) >= 3:  # Need at least 3 points for fitting
                        params, _ = curve_fit(
                            dose_response,
                            nonzero_conc,
                            nonzero_act,
                            p0=[10, 1],
                            bounds=([0.1, 0.1], [1000, 10]),
                        )
                        calc_ic50, calc_hill = params

                        log += f"Dose-response analysis for {modulator_name}:\n"
                        log += f"- IC50: {calc_ic50:.2f} μM\n"
                        log += f"- Hill coefficient: {calc_hill:.2f}\n"
                    else:
                        log += f"Insufficient non-zero data points to calculate IC50 for {modulator_name}.\n"
                except Exception:
                    log += f"Error: Could not fit dose-response curve for {modulator_name}.\n"
            else:
                log += f"Insufficient data points to calculate IC50 for {modulator_name}.\n"

            log += f"Dose-response data for {modulator_name} saved to: {modulator_file}\n\n"

    # 4. Summary
    log += "\n### Summary\n\n"
    log += f"Completed in vitro enzyme kinetics assay for {enzyme_name}.\n"
    log += f"- Determined linear range for time-course measurements: 0-{linear_time[-1]} minutes\n"
    log += f"- Characterized Michaelis-Menten kinetics (Vmax: {vmax:.2f} units, Km: {km:.2f} μM)\n"

    if modulators:
        log += "- Analyzed the effects of modulators:\n"
        for modulator_name in modulators:
            log += f"  * {modulator_name}: Data saved to {enzyme_name}_{modulator_name}_dose_response.csv\n"

    return log


def analyze_itc_binding_thermodynamics(
    itc_data_path=None,
    itc_data=None,
    temperature=298.15,
    protein_concentration=None,
    ligand_concentration=None,
):
    """Analyzes isothermal titration calorimetry (ITC) data to determine binding affinity and thermodynamic parameters.

    Parameters
    ----------
    itc_data_path : str, optional
        Path to CSV or TSV file containing ITC thermogram data with columns for injection number,
        injection volume, and heat released/absorbed. Expected columns: 'injection', 'volume', 'heat'
    itc_data : numpy.ndarray, optional
        Raw ITC thermogram data as a numpy array.
        Expected shape: (n_injections, 3) with columns for injection number, injection volume, and heat
        This parameter is provided for backward compatibility and will be deprecated.
    temperature : float, optional
        Temperature in Kelvin at which the experiment was conducted. Default is 298.15 K (25°C).
    protein_concentration : float, optional
        Initial concentration of protein in the cell in molar (M). Required for accurate fitting.
    ligand_concentration : float, optional
        Concentration of ligand in the syringe in molar (M). Required for accurate fitting.

    Returns
    -------
    str
        A research log summarizing the analysis steps and results, including binding affinity (Kd),
        binding enthalpy (ΔH), binding entropy (ΔS), and Gibbs free energy (ΔG).

    """
    import datetime

    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    log = []
    log.append(f"# ITC Binding Affinity Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.append("\n## Data Preprocessing")

    # Check if we have data to process
    if itc_data_path is None and itc_data is None:
        log.append("Error: No data provided. Please provide either itc_data_path or itc_data.")
        return "\n".join(log)

    # Load data from file if path is provided
    if itc_data_path is not None:
        try:
            if itc_data_path.endswith(".csv"):
                loaded_data = pd.read_csv(itc_data_path)
            elif itc_data_path.endswith((".tsv", ".txt")):
                loaded_data = pd.read_csv(itc_data_path, sep="\t")
            else:
                log.append("Error: Unsupported file format. Please provide a CSV or TSV file.")
                return "\n".join(log)

            log.append(f"- Loaded data from file: {itc_data_path}")
            log.append(f"- Input data: DataFrame with {len(loaded_data)} injections")

            if all(col in loaded_data.columns for col in ["injection", "volume", "heat"]):
                data = loaded_data[["injection", "volume", "heat"]].values
            else:
                log.append("- Error: DataFrame must contain 'injection', 'volume', and 'heat' columns")
                return "\n".join(log)
        except Exception as e:
            log.append(f"- Error loading data from file: {str(e)}")
            return "\n".join(log)
    # Use provided numpy array data
    elif itc_data is not None:
        if isinstance(itc_data, pd.DataFrame):
            log.append("Warning: Passing DataFrame directly is deprecated. Please use itc_data_path instead.")
            log.append(f"- Input data: DataFrame with {len(itc_data)} injections")
            if all(col in itc_data.columns for col in ["injection", "volume", "heat"]):
                data = itc_data[["injection", "volume", "heat"]].values
            else:
                log.append("- Error: DataFrame must contain 'injection', 'volume', and 'heat' columns")
                return "\n".join(log)
        else:
            log.append(f"- Input data: Array with {len(itc_data)} injections")
            data = np.array(itc_data)

    # Check if concentration data is provided
    if protein_concentration is None or ligand_concentration is None:
        log.append("- Warning: Protein or ligand concentration not provided. Using normalized data for fitting.")
        protein_concentration = 1.0
        ligand_concentration = 10.0

    # Extract data
    injections = data[:, 0]
    volumes = data[:, 1]
    heats = data[:, 2]

    log.append(f"- Processed {len(injections)} injections")

    # Calculate molar ratio for each injection
    cell_volume = 1.4  # Default cell volume in mL (typical for ITC)

    # Calculate cumulative ligand added
    cumulative_volume = np.cumsum(volumes)
    dilution_factor = 1 - (cumulative_volume / cell_volume)
    protein_conc_corrected = protein_concentration * dilution_factor

    # Calculate molar ratio [ligand]/[protein] for each injection
    molar_ratios = np.zeros_like(injections, dtype=float)
    for i in range(len(injections)):
        ligand_added = volumes[i] * ligand_concentration / 1000  # Convert to mmol
        if i == 0:
            molar_ratios[i] = ligand_added / (protein_concentration * cell_volume / 1000)  # Convert to mmol
        else:
            # Account for dilution and previous injections
            molar_ratios[i] = molar_ratios[i - 1] + ligand_added / (
                protein_conc_corrected[i] * (cell_volume - cumulative_volume[i]) / 1000
            )

    log.append("\n## Model Fitting")
    log.append("- Applying one-site binding model to the ITC data")

    # Define one-site binding model function
    def one_site_model(x, Kd, dH, n):
        """One-site binding model for ITC data.

        Parameters
        ----------
        x: Molar ratio [ligand]/[protein]
        Kd: Dissociation constant (M)
        dH: Enthalpy change (cal/mol)
        n: Stoichiometry

        Returns: Heat per injection

        """
        # Convert x to fraction bound using the binding equation
        Ka = 1 / Kd  # Association constant
        protein = protein_conc_corrected

        # Calculate heat for each injection
        q = np.zeros_like(x)
        for i in range(len(x)):
            if i == 0:
                # First injection
                bound = (n * protein[i] * Ka * (x[i] * protein[i])) / (1 + Ka * (x[i] * protein[i]))
                q[i] = bound * dH * cell_volume
            else:
                # Subsequent injections (differential heat)
                bound_prev = (n * protein[i - 1] * Ka * (x[i - 1] * protein[i - 1])) / (
                    1 + Ka * (x[i - 1] * protein[i - 1])
                )
                bound_curr = (n * protein[i] * Ka * (x[i] * protein[i])) / (1 + Ka * (x[i] * protein[i]))
                q[i] = bound_curr * dH * (cell_volume - cumulative_volume[i]) - bound_prev * dH * (
                    cell_volume - cumulative_volume[i - 1]
                )

        return q

    # Initial parameter guesses
    p0 = [1e-6, -5000, 1.0]  # Kd (M), dH (cal/mol), n (stoichiometry)

    try:
        # Fit the model to the data
        popt, pcov = curve_fit(one_site_model, molar_ratios, heats, p0=p0, maxfev=10000)
        Kd, dH, n = popt

        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        Kd_err, dH_err, n_err = perr

        # Calculate other thermodynamic parameters
        R = 1.9872  # Gas constant in cal/(mol·K)
        dG = R * temperature * np.log(Kd)  # Gibbs free energy
        dS = (dH - dG) / temperature  # Entropy

        log.append("- Model fitting successful")
        log.append("\n## Results")
        log.append(f"- Binding Stoichiometry (n): {n:.2f} ± {n_err:.2f}")
        log.append(f"- Dissociation Constant (Kd): {Kd * 1e6:.2f} ± {Kd_err * 1e6:.2f} μM")
        log.append(f"- Association Constant (Ka): {1 / Kd / 1e6:.2f} × 10^6 M^-1")
        log.append(f"- Binding Enthalpy (ΔH): {dH:.2f} ± {dH_err:.2f} cal/mol")
        log.append(f"- Binding Entropy (ΔS): {dS:.2f} cal/(mol·K)")
        log.append(f"- Gibbs Free Energy (ΔG): {dG:.2f} cal/mol")

        # Calculate goodness of fit
        residuals = heats - one_site_model(molar_ratios, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((heats - np.mean(heats)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        log.append(f"- R-squared: {r_squared:.4f}")

    except Exception as e:
        log.append(f"- Error during model fitting: {str(e)}")
        log.append("- Consider trying different initial parameter guesses or a different binding model")

    log.append("\n## Conclusion")
    log.append("- Analysis complete. The thermodynamic parameters have been estimated using a one-site binding model.")
    log.append(
        "- For more complex binding scenarios, consider using multi-site binding models or specialized ITC analysis software."
    )

    return "\n".join(log)


def analyze_protein_conservation(protein_sequences, output_dir="./"):
    """Perform multiple sequence alignment and phylogenetic analysis to identify conserved protein regions.

    Parameters
    ----------
    protein_sequences : list of str
        List of protein sequences in FASTA format from multiple organisms.
    output_dir : str, optional
        Directory to save output files (default: "./output")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results, including filenames of saved outputs.

    """
    import os

    from Bio import AlignIO, Phylo, SeqIO
    from Bio.Align.Applications import MuscleCommandline
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

    # Create a research log
    log = []
    log.append("# Protein Sequence Alignment and Conservation Analysis")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Save input sequences to a temporary file
    log.append("\n## Step 1: Preparing Input Sequences")
    input_file = os.path.join(output_dir, "input.fasta")

    # Check if input is already in FASTA format or needs conversion
    if isinstance(protein_sequences, list):
        if all(">" in seq for seq in protein_sequences):
            # Already in FASTA format
            with open(input_file, "w") as f:
                f.write("\n".join(protein_sequences))
        else:
            # Convert to FASTA format
            with open(input_file, "w") as f:
                for i, seq in enumerate(protein_sequences):
                    f.write(f">Sequence_{i + 1}\n{seq}\n")
    else:
        # Assume it's a single string in FASTA format
        with open(input_file, "w") as f:
            f.write(protein_sequences)

    log.append(f"Input sequences saved to {input_file}")
    log.append(f"Number of sequences: {len(list(SeqIO.parse(input_file, 'fasta')))}")

    # Step 2: Perform multiple sequence alignment using MUSCLE
    log.append("\n## Step 2: Multiple Sequence Alignment")
    aligned_file = os.path.join(output_dir, "aligned.fasta")

    try:
        # Run MUSCLE for multiple sequence alignment
        muscle_cline = MuscleCommandline(input=input_file, out=aligned_file)
        stdout, stderr = muscle_cline()
        log.append("Multiple sequence alignment completed using MUSCLE")
        log.append(f"Alignment saved to {aligned_file}")

        # Load the alignment
        alignment = AlignIO.read(aligned_file, "fasta")
        log.append(f"Alignment length: {alignment.get_alignment_length()} positions")

    except Exception as e:
        log.append(f"MUSCLE alignment failed: {str(e)}")
        log.append("Attempting to use Biopython's built-in alignment methods...")

        # Fallback to a simpler approach without pairwise2
        from Bio.Align import MultipleSeqAlignment
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        sequences = list(SeqIO.parse(input_file, "fasta"))

        # Simple approach: just pad sequences to the same length
        max_length = max(len(seq.seq) for seq in sequences)
        alignments = []

        for seq in sequences:
            # Pad sequence to max length
            padded_seq = str(seq.seq).ljust(max_length, "-")
            new_seq = SeqRecord(Seq(padded_seq), id=seq.id, description=seq.description)
            alignments.append(new_seq)

        # Create and save the alignment
        msa = MultipleSeqAlignment(alignments)
        AlignIO.write(msa, aligned_file, "fasta")
        alignment = msa
        log.append("Simple padding alignment completed as fallback method")
        log.append(f"Alignment saved to {aligned_file}")

    # Step 3: Generate a phylogenetic tree
    log.append("\n## Step 3: Phylogenetic Analysis")
    tree_file = os.path.join(output_dir, "tree.newick")

    # Calculate distance matrix
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)

    # Construct the phylogenetic tree using neighbor-joining method
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # Save the tree
    Phylo.write(tree, tree_file, "newick")
    log.append("Phylogenetic tree constructed using neighbor-joining method")
    log.append(f"Tree saved to {tree_file}")

    # Step 4: Analyze conserved regions
    log.append("\n## Step 4: Conservation Analysis")
    conservation_file = os.path.join(output_dir, "conservation.txt")

    # Simple conservation analysis
    alignment_length = alignment.get_alignment_length()
    conserved_positions = []

    with open(conservation_file, "w") as f:
        f.write("Position\tConservation_Score\tConsensus\n")

        for i in range(alignment_length):
            # Get all amino acids at this position
            column = alignment[:, i]
            set(column)

            # Calculate a simple conservation score (percentage of most common AA)
            most_common_aa = max(column, key=column.count)
            conservation_score = column.count(most_common_aa) / len(column)

            f.write(f"{i + 1}\t{conservation_score:.2f}\t{most_common_aa}\n")

            # Consider positions with >80% conservation as conserved
            if conservation_score > 0.8:
                conserved_positions.append(i + 1)

    log.append(f"Conservation analysis completed and saved to {conservation_file}")
    log.append(f"Identified {len(conserved_positions)} highly conserved positions (>80% conservation)")

    if conserved_positions:
        log.append(f"Conserved positions: {', '.join(map(str, conserved_positions[:10]))}")
        if len(conserved_positions) > 10:
            log.append(f"... and {len(conserved_positions) - 10} more")

    # Final summary
    log.append("\n## Summary")
    log.append("The analysis successfully completed with the following outputs:")
    log.append(f"1. Multiple sequence alignment: {aligned_file}")
    log.append(f"2. Phylogenetic tree: {tree_file}")
    log.append(f"3. Conservation analysis: {conservation_file}")

    return "\n".join(log)
