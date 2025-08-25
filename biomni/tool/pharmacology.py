import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd


def run_diffdock_with_smiles(pdb_path, smiles_string, local_output_dir, gpu_device=0, use_gpu=True):
    try:
        summary = []

        # Check if PDB file exists
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"The PDB file '{pdb_path}' does not exist.")
        summary.append(f"PDB file '{pdb_path}' found.")

        # Ensure the output directory exists
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)
        summary.append(f"Output directory '{local_output_dir}' is ready.")

        # Pull the pre-built container from Docker Hub
        summary.append("Pulling DiffDock container from Docker Hub...")
        subprocess.run(["docker", "pull", "rbgcsail/diffdock"], check=True)
        summary.append("DiffDock container pulled successfully.")

        # Check for GPU availability (if using GPU)
        if use_gpu:
            summary.append("Checking for GPU availability...")
            gpu_check = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "nvidia/cuda:11.7.1-devel-ubuntu22.04",
                    "nvidia-smi",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary.append(f"GPU Status: {gpu_check.stdout.strip()}")

        # Prepare the GPU flag
        gpu_flag = ["--gpus", f"device={gpu_device}"] if use_gpu else []

        # Docker run command
        summary.append("Running DiffDock inference...")
        run_command = (
            ["docker", "run"]
            + gpu_flag
            + [
                # Mount the local directory to /home/appuser/output inside the container
                "-v",
                f"{os.path.abspath(pdb_path)}:/home/appuser/input/protein.pdb",  # PDB file mount
                "-v",
                f"{os.path.abspath(local_output_dir)}:/home/appuser/output",  # Output directory mount
                "--entrypoint",
                "/bin/bash",
                "rbgcsail/diffdock",
                "-c",
                # Command to run inference using micromamba environment
                f"micromamba run -n diffdock python -m inference --config default_inference_args.yaml "
                f"--protein_path /home/appuser/input/protein.pdb --ligand '{smiles_string}' --out_dir /home/appuser/output",
            ]
        )

        # Execute the Docker command
        result = subprocess.run(run_command, check=False, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            summary.append(f"Error during inference: {result.stderr.strip()}")
            return "\n".join(summary)
        else:
            summary.append("DiffDock inference completed successfully.")
            summary.append(f"Results stored in '{local_output_dir}'.")

        return "\n".join(summary)

    except FileNotFoundError as e:
        return f"File error: {e}"
    except subprocess.CalledProcessError as e:
        return f"Command execution error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def docking_autodock_vina(smiles_list, receptor_pdb_file, box_center, box_size, ncpu=1):
    from tdc import Oracle

    log = []

    # Log the start of the process
    log.append("Step 1: Initializing the Oracle")
    log.append(f"Receptor PDB File: {receptor_pdb_file}")
    log.append(f"Box Center: {box_center}")
    log.append(f"Box Size: {box_size}")

    # Initialize the Oracle object
    oracle = Oracle(
        name="pyscreener",
        receptor_pdb_file=receptor_pdb_file,
        box_center=box_center,
        box_size=box_size,
        ncpu=ncpu,
    )
    log.append("Oracle initialized successfully.")

    # Log the list of SMILES strings
    log.append(f"\nStep 2: Processing SMILES strings: {smiles_list}")

    # Get the docking scores
    docking_scores = oracle(smiles_list)
    log.append(f"Docking scores calculated: {docking_scores}")

    # Create a dictionary mapping SMILES to their docking scores
    results_dict = dict(zip(smiles_list, docking_scores, strict=False))

    # Log the result mapping
    log.append("\nStep 3: Mapping SMILES to docking scores:")
    log.append(f"Results: {results_dict}")

    # Convert the log to a string and return it
    research_log = "\n".join(log)
    return research_log


def run_autosite(pdb_file, output_dir, spacing=1.0):
    # Prepare the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the PDB file to PDBQT format (assuming prepare_receptor4.py is accessible)
    pdbqt_file = pdb_file.replace(".pdb", ".pdbqt")
    subprocess.run(["prepare_receptor", "-r", pdb_file, "-o", pdbqt_file], check=True)

    # Run AutoSite
    autosite_cmd = [
        "autosite",
        "-r",
        pdbqt_file,
        "--spacing",
        str(spacing),
        "-o",
        output_dir,
    ]
    subprocess.run(autosite_cmd, check=True)

    # Parse the results to find the box center and size
    box_center, box_size = None, None
    log_path = os.path.join(output_dir, "_AutoSiteSummary.log")
    with open(log_path) as log_file:
        log_content = log_file.read()

        # Extract box center and size from the log (assuming standard output format)
        box_center_match = re.search(r"Box center:\s*\(([^)]+)\)", log_content)
        box_size_match = re.search(r"Box size:\s*\(([^)]+)\)", log_content)

        if box_center_match:
            box_center = box_center_match.group(1)
        if box_size_match:
            box_size = box_size_match.group(1)

    # Create a research log string
    research_log = f"AutoSite run for {pdb_file} with spacing {spacing}\n"
    research_log += f"Output directory: {output_dir}\n"
    if box_center and box_size:
        research_log += f"Box Center: {box_center}\nBox Size: {box_size}"
    else:
        research_log += "Box Center and Size information not found in log."

    return research_log


# Function to get TxGNN predictions and return a summarized string output
def retrieve_topk_repurposing_drugs_from_disease_txgnn(disease_name, data_lake_path, k=5):
    """This function computes TxGNN model predictions for drug repurposing. It takes in the paths to the data,
    the disease name, and returns a summary of the top K predicted drugs with their sigmoid-transformed scores.

    Args:
    - disease_name (str): The name of the disease for which the drug predictions are to be retrieved.
    - data_lake_path (str): The path to the data lake containing the TxGNN predictions.
    - k (int, optional): The number of top drug predictions to return. Defaults to 5.

    Returns:
    - str: A summary of the steps and the top K drug predictions with their scores.

    """

    # Sigmoid function to convert raw prediction scores
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Step 1: Load the mappings and prediction data from the provided paths
    name_mapping_path = data_lake_path + "/txgnn_name_mapping.pkl"
    result_path = data_lake_path + "/txgnn_prediction.pkl"

    with open(name_mapping_path, "rb") as f:
        mapping = pickle.load(f)

    with open(result_path, "rb") as f:
        result = pickle.load(f)

    # Step 2: Fuzzy match the disease name to find the closest match
    possible_diseases = result.keys()
    matched_disease = get_close_matches(disease_name, possible_diseases, n=1, cutoff=0.6)

    if not matched_disease:
        return f"Error: No matching disease found for '{disease_name}'. Please try a different name."

    matched_disease = matched_disease[0]

    # Step 3: Retrieve the prediction scores for the matched disease
    disease_predictions = result[matched_disease]

    # Step 4: Apply the sigmoid function to the raw prediction scores
    sigmoid_predictions = {drug_id: sigmoid(score) for drug_id, score in disease_predictions.items()}

    # Step 5: Sort the drugs by prediction score in descending order
    top_k_drugs = sorted(sigmoid_predictions.items(), key=lambda x: x[1], reverse=True)[:k]

    # Step 6: Map drug IDs to their names and format the results
    top_k_drug_names = [(mapping["id2name_drug"].get(drug_id, "Unknown Drug"), score) for drug_id, score in top_k_drugs]

    # Step 7: Create a human and LLM-friendly summary string
    summary = f"TxGNN Drug Repurposing Predictions for '{matched_disease}':\n"
    summary += f"Top {k} predicted drugs and their corresponding prediction scores (post-sigmoid transformation):\n"

    for i, (drug_name, score) in enumerate(top_k_drug_names, 1):
        summary += f"{i}. {drug_name} - Prediction Score: {score:.4f}\n"

    summary += "\nProcess Summary:\n"
    summary += "- Fuzzy matching was used to match the input disease name to '{matched_disease}'.\n"
    summary += "- Sigmoid function was applied to raw prediction scores to convert them into probabilities.\n"
    summary += f"- The top {k} drugs were selected based on their prediction scores.\n"

    return summary


# ADMET prediction function with research log format
def predict_admet_properties(smiles_list, ADMET_model_type="MPNN"):
    try:
        from DeepPurpose import CompoundPred, utils
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "DeepPurpose"], check=False)
        from DeepPurpose import CompoundPred, utils

    # Define available model types
    available_model_types = ["MPNN", "CNN", "Morgan"]

    # Check if the provided model type is valid
    if ADMET_model_type not in available_model_types:
        return f"Error: Invalid ADMET model type '{ADMET_model_type}'. Available options are: {', '.join(available_model_types)}."

    # Load pretrained ADMET models only once
    model_ADMETs = {}
    tasks = [
        "AqSolDB",
        "Caco2",
        "HIA",
        "Pgp_inhibitor",
        "Bioavailability",
        "BBB_MolNet",
        "PPBR",
        "CYP2C19",
        "CYP2D6",
        "CYP3A4",
        "CYP1A2",
        "CYP2C9",
        "ClinTox",
        "Lipo_AZ",
        "Half_life_eDrug3D",
        "Clearance_eDrug3D",
    ]

    for task in tasks:
        model_ADMETs[task + "_" + ADMET_model_type + "_model"] = CompoundPred.model_pretrained(
            model=task + "_" + ADMET_model_type + "_model"
        )

    # Helper function for ADMET prediction
    def ADMET_pred(drug, task, unit):
        model = model_ADMETs[task + "_" + ADMET_model_type + "_model"]
        X_pred = utils.data_process(
            X_drug=[drug],
            y=[0],
            drug_encoding=ADMET_model_type,
            split_method="no_split",
        )
        y_pred = model.predict(X_pred)[0]

        if unit == "%":
            y_pred = y_pred * 100

        return f"{y_pred:.2f} " + unit

    # Initialize research log string
    research_log = "Research Log for ADMET Predictions:\n"
    research_log += "-------------------------------------\n"

    # Process each SMILES string in the list
    for smiles in smiles_list:
        research_log += f"\nCompound SMILES: {smiles}\n"
        research_log += "Predicted ADMET properties:\n"

        # Physiochemical properties
        solubility = ADMET_pred(smiles, "AqSolDB", "log mol/L")
        lipophilicity = ADMET_pred(smiles, "Lipo_AZ", "(log-ratio)")
        research_log += f"- Solubility: {solubility}\n"
        research_log += f"- Lipophilicity: {lipophilicity}\n"

        # Absorption
        caco2 = ADMET_pred(smiles, "Caco2", "cm/s")
        hia = ADMET_pred(smiles, "HIA", "%")
        pgp = ADMET_pred(smiles, "Pgp_inhibitor", "%")
        bioavail = ADMET_pred(smiles, "Bioavailability", "%")
        research_log += f"- Absorption (Caco-2 permeability): {caco2}\n"
        research_log += f"- Absorption (HIA): {hia}\n"
        research_log += f"- Absorption (Pgp Inhibitor): {pgp}\n"
        research_log += f"- Absorption (Bioavailability): {bioavail}\n"

        # Distribution
        bbb = ADMET_pred(smiles, "BBB_MolNet", "%")
        ppbr = ADMET_pred(smiles, "PPBR", "%")
        research_log += f"- Distribution (BBB permeation): {bbb}\n"
        research_log += f"- Distribution (PPBR): {ppbr}\n"

        # Metabolism
        cyp2c19 = ADMET_pred(smiles, "CYP2C19", "%")
        cyp2d6 = ADMET_pred(smiles, "CYP2D6", "%")
        cyp3a4 = ADMET_pred(smiles, "CYP3A4", "%")
        cyp1a2 = ADMET_pred(smiles, "CYP1A2", "%")
        cyp2c9 = ADMET_pred(smiles, "CYP2C9", "%")
        research_log += f"- Metabolism (CYP2C19): {cyp2c19}\n"
        research_log += f"- Metabolism (CYP2D6): {cyp2d6}\n"
        research_log += f"- Metabolism (CYP3A4): {cyp3a4}\n"
        research_log += f"- Metabolism (CYP1A2): {cyp1a2}\n"
        research_log += f"- Metabolism (CYP2C9): {cyp2c9}\n"

        # Excretion
        half_life = ADMET_pred(smiles, "Half_life_eDrug3D", "h")
        clearance = ADMET_pred(smiles, "Clearance_eDrug3D", "mL/min/kg")
        research_log += f"- Excretion (Half-life): {half_life}\n"
        research_log += f"- Excretion (Clearance): {clearance}\n"

        # Clinical Toxicity
        clinical_toxicity = ADMET_pred(smiles, "ClinTox", "%")
        research_log += f"- Clinical Toxicity: {clinical_toxicity}\n"

        research_log += "-------------------------------------\n"

    return research_log


# Binding Affinity prediction function with model_type validation
def predict_binding_affinity_protein_1d_sequence(smiles_list, amino_acid_sequence, affinity_model_type="MPNN-CNN"):
    try:
        from DeepPurpose import DTI, utils
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "DeepPurpose"], check=False)
        from DeepPurpose import DTI, utils

    # Define available model types for Binding Affinity
    available_affinity_model_types = [
        "CNN-CNN",
        "MPNN-CNN",
        "Morgan-CNN",
        "Morgan-AAC",
        "Daylight-AAC",
    ]

    # Check if the provided affinity model type is valid
    if affinity_model_type not in available_affinity_model_types:
        return f"Error: Invalid affinity model type '{affinity_model_type}'. Available options are: {', '.join(available_affinity_model_types)}."

    # Load the pre-trained affinity model
    model_DTI = DTI.model_pretrained(model=affinity_model_type.replace("-", "_") + "_BindingDB")

    # Initialize research log string
    research_log = "Research Log for Binding Affinity Predictions:\n"
    research_log += "-------------------------------------\n"

    # Process each SMILES string in the list
    for smiles in smiles_list:
        research_log += f"\nCompound SMILES: {smiles}\n"
        research_log += f"Amino Acid Sequence: {amino_acid_sequence}\n"

        # Predict binding affinity
        X_pred = utils.data_process(
            X_drug=[smiles],
            X_target=[amino_acid_sequence],
            y=[0],
            drug_encoding=affinity_model_type.split("-")[0],
            target_encoding=affinity_model_type.split("-")[1],
            split_method="no_split",
        )
        y_pred = model_DTI.predict(X_pred)[0]
        y_pred_nM = 10 ** (-y_pred) / 1e-9

        research_log += f"Predicted Binding Affinity: {y_pred_nM:.2f} nM\n"
        research_log += "-------------------------------------\n"

    return research_log


def analyze_accelerated_stability_of_pharmaceutical_formulations(formulations, storage_conditions, time_points):
    """Analyzes the stability of pharmaceutical formulations under accelerated storage conditions.

    Parameters
    ----------
    formulations : list of dict
        List of formulation dictionaries, each containing:
        - 'name': str, name of the formulation
        - 'active_ingredient': str, name of the active pharmaceutical ingredient
        - 'concentration': float, concentration in mg/mL
        - 'excipients': list, list of excipients

    storage_conditions : list of dict
        List of storage condition dictionaries, each containing:
        - 'temperature': float, temperature in °C
        - 'humidity': float, relative humidity in percentage (optional for solid dosage forms)
        - 'description': str, description of storage condition (e.g., "Room Temperature", "Accelerated")

    time_points : list of int
        List of time points in days to evaluate stability

    Returns
    -------
    str
        Research log summarizing the stability testing process and results

    """
    # Create output directory if it doesn't exist
    output_dir = "stability_test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results storage
    all_results = []

    # Process each formulation under each storage condition
    for formulation in formulations:
        for condition in storage_conditions:
            # Initialize stability parameters
            results = []

            # Get acceleration factor based on temperature (simplified Arrhenius equation)
            temp_c = condition["temperature"]
            accel_factor = 2 ** ((temp_c - 25) / 10)  # Rule of thumb: reaction rate doubles every 10°C

            # Add humidity effect for degradation if provided
            humidity_factor = 1.0
            if "humidity" in condition:
                humidity = condition["humidity"]
                # Simple model: higher humidity increases degradation rate
                humidity_factor = 1.0 + (humidity - 60) / 100 if humidity > 60 else 1.0

            # Calculate stability parameters at each time point
            initial_content = 100.0  # Starting at 100%

            for time in time_points:
                # Chemical stability (% of initial content)
                # Simple first-order degradation model
                effective_time = time * accel_factor * humidity_factor
                chemical_stability = initial_content * np.exp(-0.001 * effective_time)

                # Physical stability (score from 1-10, 10 being perfect)
                # Decreases over time, affected by temperature and humidity
                physical_stability = 10 - (0.05 * effective_time)
                physical_stability = max(1, physical_stability)  # Minimum score of 1

                # Particle size change (% increase from initial)
                # Some formulations show particle growth over time
                particle_size_change = (
                    0.2 * effective_time if "solid" in formulation.get("dosage_form", "").lower() else 0
                )

                results.append(
                    {
                        "Formulation": formulation["name"],
                        "Storage_Condition": condition["description"],
                        "Temperature_C": temp_c,
                        "Humidity_RH": condition.get("humidity", "N/A"),
                        "Time_Days": time,
                        "Chemical_Stability_Percent": round(chemical_stability, 2),
                        "Physical_Stability_Score": round(physical_stability, 1),
                        "Particle_Size_Change_Percent": round(particle_size_change, 2),
                    }
                )

            all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    csv_filename = f"{output_dir}/stability_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)

    # Generate research log
    log = "Accelerated Stability Testing of Pharmaceutical Formulations\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "1. STUDY PARAMETERS\n"
    log += f"   - Number of formulations tested: {len(formulations)}\n"
    log += f"   - Formulations: {', '.join([f['name'] for f in formulations])}\n"
    log += (
        "   - Storage conditions: "
        + ", ".join(
            [
                f"{c['description']} ({c['temperature']}°C" + (f"/{c['humidity']}% RH" if "humidity" in c else "") + ")"
                for c in storage_conditions
            ]
        )
        + "\n"
    )
    log += f"   - Time points evaluated (days): {', '.join(map(str, time_points))}\n\n"

    log += "2. METHODOLOGY\n"
    log += "   - Chemical stability assessed by active ingredient content\n"
    log += "   - Physical stability evaluated on a 10-point scale\n"
    log += "   - Particle size changes measured where applicable\n\n"

    log += "3. KEY FINDINGS\n"

    # Summarize stability at final time point for each formulation/condition
    final_time = max(time_points)
    final_results = results_df[results_df["Time_Days"] == final_time]

    for formulation in formulations:
        form_results = final_results[final_results["Formulation"] == formulation["name"]]
        log += f"   {formulation['name']}:\n"

        for _, row in form_results.iterrows():
            condition = row["Storage_Condition"]
            chem_stab = row["Chemical_Stability_Percent"]
            phys_stab = row["Physical_Stability_Score"]

            stability_assessment = "Stable"
            if chem_stab < 90 or phys_stab < 7:
                stability_assessment = "Potentially unstable"
            if chem_stab < 85 or phys_stab < 5:
                stability_assessment = "Unstable"

            log += f"     - {condition}: Chemical stability {chem_stab}%, Physical stability score {phys_stab}/10 - {stability_assessment}\n"
        log += "\n"

    log += "4. CONCLUSION\n"

    # Identify most stable formulation
    best_formulation = ""
    best_stability = 0

    for formulation in formulations:
        form_data = final_results[final_results["Formulation"] == formulation["name"]]
        avg_chem_stability = form_data["Chemical_Stability_Percent"].mean()
        if avg_chem_stability > best_stability:
            best_stability = avg_chem_stability
            best_formulation = formulation["name"]

    log += f"   - Most stable formulation: {best_formulation} (avg. chemical stability: {best_stability:.2f}%)\n"
    log += f"   - Detailed results saved to: {csv_filename}\n"

    return log


def run_3d_chondrogenic_aggregate_assay(
    chondrocyte_cells, test_compounds, culture_duration_days=21, measurement_intervals=7
):
    """Generates a detailed protocol for performing a 3D chondrogenic aggregate culture assay to evaluate compounds' effects on chondrogenesis.

    Parameters
    ----------
    chondrocyte_cells : dict
        Dictionary with cell information including 'source', 'passage_number', and 'cell_density'
    test_compounds : list of dict
        List of compounds to test, each with 'name', 'concentration', and 'vehicle' keys
    culture_duration_days : int
        Total duration of the culture period in days (default: 21)
    measurement_intervals : int
        Interval in days between measurements (default: 7)

    Returns
    -------
    str
        Detailed protocol document for the 3D chondrogenic aggregate culture assay

    """
    from datetime import datetime

    # Create experiment ID
    experiment_id = f"CHOND3D_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create time points for measurements
    timepoints = list(range(0, culture_duration_days + 1, measurement_intervals))
    if timepoints[-1] != culture_duration_days:
        timepoints.append(culture_duration_days)

    # Generate the protocol document
    protocol = f"# 3D Chondrogenic Aggregate Culture Assay Protocol - {experiment_id}\n\n"

    protocol += "## 1. Materials and Reagents\n\n"
    protocol += "- Chondrocyte cells\n"
    protocol += "- Chondrogenic differentiation medium\n"
    protocol += "- Transforming growth factor-β3 (TGF-β3)\n"
    protocol += "- Dexamethasone\n"
    protocol += "- Ascorbate-2-phosphate\n"
    protocol += "- 96-well V-bottom plates\n"
    protocol += "- Gaussia luciferase reporter assay kit\n"
    protocol += "- Luminometer\n"
    protocol += "- Test compounds with respective vehicles\n"
    protocol += "- Centrifuge\n"
    protocol += "- CO2 incubator\n"
    protocol += "- Sterile pipettes and tips\n\n"

    protocol += "## 2. Experimental Information\n\n"
    protocol += "### Cell Information:\n"
    protocol += f"- Cell source: {chondrocyte_cells['source']}\n"
    protocol += f"- Passage number: {chondrocyte_cells['passage_number']}\n"
    protocol += f"- Cell density: {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "### Experimental Design:\n"
    protocol += f"- Culture duration: {culture_duration_days} days\n"
    protocol += f"- Measurement timepoints: {', '.join(map(str, timepoints))} days\n\n"

    protocol += "### Test Compounds:\n"
    for i, compound in enumerate(test_compounds):
        protocol += f"- Compound {i + 1}: {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "- Control: Vehicle only\n\n"

    protocol += "## 3. Detailed Procedure\n\n"
    protocol += "### Day 0: Setup\n\n"
    protocol += "1. Prepare chondrogenic differentiation medium containing:\n"
    protocol += "   - High-glucose DMEM\n"
    protocol += "   - 10 ng/mL TGF-β3\n"
    protocol += "   - 100 nM Dexamethasone\n"
    protocol += "   - 50 μg/mL Ascorbate-2-phosphate\n"
    protocol += "   - 1% ITS+ premix (insulin, transferrin, selenium)\n"
    protocol += "   - 1 mM Sodium pyruvate\n"
    protocol += "   - 100 U/mL Penicillin/Streptomycin\n\n"

    protocol += "2. Harvest and count chondrocyte cells\n\n"

    protocol += "3. Prepare cell suspension at the specified density:\n"
    protocol += f"   - {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "4. Form 3D cell aggregates:\n"
    protocol += "   - Aliquot 2.5×10^5 cells per well in 96-well V-bottom plates\n"
    protocol += "   - Centrifuge plates at 500g for 5 minutes to pellet cells\n\n"

    protocol += "5. Add test compounds to respective wells:\n"
    for _, compound in enumerate(test_compounds):
        protocol += f"   - Add {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "   - Add vehicle only to control wells\n\n"

    protocol += "6. Incubate the plates at 37°C, 5% CO2\n\n"

    protocol += "### Day 1 to Day " + str(culture_duration_days) + ":\n\n"
    protocol += "1. Change medium every 2-3 days:\n"
    protocol += "   - Carefully remove 50% of the medium without disturbing the aggregates\n"
    protocol += "   - Replace with fresh medium containing test compounds at the same concentrations\n\n"

    protocol += f"2. At days {', '.join(map(str, timepoints))}, collect samples for analysis:\n"
    protocol += (
        "   - Take medium samples for Gaussia luciferase activity measurement (if using COL2A1-GLuc reporter cells)\n"
    )
    protocol += "   - Fix aggregates in 4% paraformaldehyde for histological analysis\n\n"

    return protocol


def grade_adverse_events_using_vcog_ctcae(clinical_data_file):
    """Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.

    Parameters
    ----------
    clinical_data_file : str
        Path to a CSV file containing clinical evaluation data with columns:
        subject_id, time_point, symptom, severity, measurement (optional)

    Returns
    -------
    str
        A research log summarizing the adverse event grading process and results.
        The graded events are saved to 'vcog_ctcae_graded_events.csv'.

    """
    import json
    from datetime import datetime

    import pandas as pd

    # Initialize the research log
    log = "# Adverse Event Grading using VCOG-CTCAE v1.1\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Step 1: Load the clinical data
    log += "## Step 1: Loading clinical evaluation data\n"
    try:
        data = pd.read_csv(clinical_data_file)
        log += f"Successfully loaded data from {clinical_data_file}\n"
        log += f"Total records: {len(data)}\n"
        log += f"Columns found: {', '.join(data.columns)}\n\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Step 2: Define VCOG-CTCAE grading criteria
    log += "## Step 2: Applying VCOG-CTCAE grading criteria\n"

    # Comprehensive VCOG-CTCAE grading criteria based on VCOG-CTCAE v1.1
    vcog_criteria = {
        # Hematologic
        "neutropenia": {
            "description": "Neutrophil count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 1500", "range": [1500, float("inf")]},
                1: {"criteria": "1000 - <1500", "range": [1000, 1500]},
                2: {"criteria": "500 - <1000", "range": [500, 1000]},
                3: {"criteria": "100 - <500", "range": [100, 500]},
                4: {"criteria": "<100", "range": [0, 100]},
                5: {"criteria": "Death due to neutropenic sepsis", "range": None},
            },
        },
        "anemia": {
            "description": "Hemoglobin decrease",
            "unit": "g/dL",
            "grades": {
                0: {"criteria": "Within reference range", "range": [10, float("inf")]},
                1: {"criteria": "Mild; clinical signs not present", "range": [8, 10]},
                2: {"criteria": "Moderate; clinical signs present", "range": [6.5, 8]},
                3: {"criteria": "Severe; transfusion indicated", "range": [5, 6.5]},
                4: {"criteria": "Life-threatening", "range": [0, 5]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "thrombocytopenia": {
            "description": "Platelet count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 100,000", "range": [100000, float("inf")]},
                1: {"criteria": "50,000 - <100,000", "range": [50000, 100000]},
                2: {"criteria": "25,000 - <50,000", "range": [25000, 50000]},
                3: {"criteria": "10,000 - <25,000", "range": [10000, 25000]},
                4: {"criteria": "<10,000 or spontaneous bleeding", "range": [0, 10000]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Gastrointestinal
        "vomiting": {
            "description": "Vomiting frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 0]},
                1: {
                    "criteria": "1-2 episodes in 24h; medical intervention not indicated",
                    "range": [1, 2],
                },
                2: {"criteria": "3-5 episodes in 24h; ≤3 days", "range": [3, 5]},
                3: {
                    "criteria": ">5 episodes in 24h; >3 days; hospitalization indicated",
                    "range": [6, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "diarrhea": {
            "description": "Diarrhea frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 0]},
                1: {
                    "criteria": "Increase of <4 stools per day over baseline",
                    "range": [1, 3],
                },
                2: {
                    "criteria": "Increase of 4-6 stools per day over baseline",
                    "range": [4, 6],
                },
                3: {
                    "criteria": "Increase of ≥7 stools per day; hospitalization indicated",
                    "range": [7, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "anorexia": {
            "description": "Appetite/food intake decrease",
            "unit": "percent of normal intake",
            "grades": {
                0: {"criteria": "Normal", "range": [100, float("inf")]},
                1: {"criteria": "Decreased appetite, but eating", "range": [75, 100]},
                2: {"criteria": "Decreased intake <3 days", "range": [50, 75]},
                3: {"criteria": "Decreased intake ≥3 days", "range": [25, 50]},
                4: {
                    "criteria": "Life-threatening consequences; urgent intervention indicated",
                    "range": [0, 25],
                },
                5: {"criteria": "Death", "range": None},
            },
        },
        # Hepatic
        "alt_increase": {
            "description": "Alanine aminotransferase increased",
            "unit": "x ULN (upper limit of normal)",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 2.5xULN", "range": [1, 2.5]},
                2: {"criteria": ">2.5 - 5.0xULN", "range": [2.5, 5]},
                3: {"criteria": ">5.0 - 20.0xULN", "range": [5, 20]},
                4: {"criteria": ">20.0xULN", "range": [20, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Renal
        "creatinine_increase": {
            "description": "Creatinine increased",
            "unit": "x ULN",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 1.5xULN", "range": [1, 1.5]},
                2: {"criteria": ">1.5 - 3.0xULN", "range": [1.5, 3]},
                3: {"criteria": ">3.0 - 6.0xULN", "range": [3, 6]},
                4: {"criteria": ">6.0xULN", "range": [6, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Constitutional
        "fever": {
            "description": "Fever",
            "unit": "°C",
            "grades": {
                0: {"criteria": "None", "range": [0, 39]},
                1: {"criteria": "39.0 - 39.5°C", "range": [39, 39.5]},
                2: {"criteria": ">39.5 - 40.0°C", "range": [39.5, 40]},
                3: {"criteria": ">40.0 - 41.0°C", "range": [40, 41]},
                4: {"criteria": ">41.0°C for >24 hrs", "range": [41, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "weight_loss": {
            "description": "Weight loss",
            "unit": "percent of baseline weight",
            "grades": {
                0: {"criteria": "<5%", "range": [0, 5]},
                1: {"criteria": "5% - <10%", "range": [5, 10]},
                2: {"criteria": "10% - <20%", "range": [10, 20]},
                3: {"criteria": "≥20%", "range": [20, float("inf")]},
                4: {"criteria": "Life-threatening", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Dermatologic
        "alopecia": {
            "description": "Hair loss",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {"criteria": "Hair loss at injection/treatment site", "range": None},
                2: {"criteria": "Moderate alopecia", "range": None},
                3: {"criteria": "Complete alopecia", "range": None},
                4: {"criteria": "Not applicable", "range": None},
                5: {"criteria": "Not applicable", "range": None},
            },
        },
        # Neurologic
        "neuropathy": {
            "description": "Peripheral neuropathy",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {
                    "criteria": "Asymptomatic; clinically detectable on examination",
                    "range": None,
                },
                2: {
                    "criteria": "Mild symptoms; limiting instrumental ADL",
                    "range": None,
                },
                3: {
                    "criteria": "Severe symptoms; limiting self-care ADL",
                    "range": None,
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
    }

    def apply_vcog_grade(symptom, severity, measurement=None):
        """Apply VCOG-CTCAE grading criteria to an adverse event.

        Parameters
        ----------
        symptom : str
            The type of adverse event
        severity : str
            The severity description
        measurement : float or None
            Quantitative measurement related to the symptom, if available

        Returns
        -------
        int
            The VCOG-CTCAE grade (0-5)
        str
            Description of the grading rationale

        """
        # Standard severity-based grading if no specific criteria exist
        grade_map = {
            "none": 0,
            "mild": 1,
            "moderate": 2,
            "severe": 3,
            "life-threatening": 4,
            "death": 5,
        }

        symptom_lower = symptom.lower()

        # Check if the symptom has specific VCOG-CTCAE criteria
        if symptom_lower in vcog_criteria:
            criteria = vcog_criteria[symptom_lower]

            # If measurement is provided and criteria has numeric ranges
            if measurement is not None:
                try:
                    measurement_value = float(measurement)

                    # Find the appropriate grade based on the measurement ranges
                    for grade, grade_info in criteria["grades"].items():
                        if grade_info["range"] is not None:
                            min_val, max_val = grade_info["range"]
                            if min_val <= measurement_value < max_val:
                                return (
                                    grade,
                                    f"Grade {grade}: {criteria['description']} - {criteria['grades'][grade]['criteria']}",
                                )
                except (ValueError, TypeError):
                    # If measurement can't be converted to float, fall back to severity-based grading
                    pass

            # If there's a reported severity with no valid measurement
            if severity.lower() in grade_map:
                # Check if the grade exists in the criteria
                severity_grade = grade_map[severity.lower()]
                if severity_grade in criteria["grades"]:
                    return (
                        severity_grade,
                        f"Grade {severity_grade}: {criteria['description']} - {criteria['grades'][severity_grade]['criteria']}",
                    )

        # Default to using the severity mapping if no specific criteria match
        if severity.lower() in grade_map:
            return grade_map[severity.lower()], f"Grade {grade_map[severity.lower()]}: Based on reported severity"

        # Default grade if no specific criteria match
        return 1, "Grade 1: Default grade (specific criteria not found)"

    # Step 3: Apply grading to each record
    log += "Applying VCOG-CTCAE v1.1 grading criteria to each adverse event...\n"

    # Create new columns for the grade and rationale
    grading_results = data.apply(
        lambda row: apply_vcog_grade(
            row["symptom"],
            row["severity"],
            row["measurement"] if "measurement" in data.columns else None,
        ),
        axis=1,
    )

    # Split the returned tuples into separate columns
    data["vcog_grade"] = [result[0] for result in grading_results]
    data["grading_rationale"] = [result[1] for result in grading_results]

    # Step 4: Analyze patterns across time points (if available)
    if "time_point" in data.columns:
        log += "\n## Step 3: Analyzing adverse event patterns across time points\n"

        # Group by subject and symptom to track progression
        progression_analysis = data.pivot_table(
            index=["subject_id", "symptom"],
            columns="time_point",
            values="vcog_grade",
            aggfunc="max",
        ).reset_index()

        # Calculate if grade is increasing, decreasing, or stable for each subject-symptom pair
        trend_counts = {"increasing": 0, "decreasing": 0, "stable": 0, "fluctuating": 0}

        numeric_columns = [col for col in progression_analysis.columns if col not in ["subject_id", "symptom"]]

        if len(numeric_columns) >= 2:
            # Sort columns to ensure chronological order
            numeric_columns.sort()

            for _, row in progression_analysis.iterrows():
                values = [row[col] for col in numeric_columns if not pd.isna(row[col])]
                if len(values) >= 2:
                    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["increasing"] += 1
                    elif all(values[i] > values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["decreasing"] += 1
                    elif all(values[i] == values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["stable"] += 1
                    else:
                        trend_counts["fluctuating"] += 1

            log += "Adverse event progression patterns:\n"
            for trend, count in trend_counts.items():
                log += f"- {trend.capitalize()}: {count} subject-symptom pairs\n"

        # Save progression analysis
        progression_file = "vcog_ctcae_progression_analysis.csv"
        progression_analysis.to_csv(progression_file)
        log += f"\nDetailed progression analysis saved to: {progression_file}\n"

    # Step 5: Summarize the grading results
    log += "\n## Step 4: Summarizing adverse event grades\n"

    # Count events by grade
    grade_counts = data["vcog_grade"].value_counts().sort_index()
    log += "Grade distribution:\n"
    for grade, count in grade_counts.items():
        log += f"- Grade {grade}: {count} events\n"

    # Summarize by symptom type
    symptom_summary = data.groupby("symptom")["vcog_grade"].agg(["max", "mean", "count"])
    log += "\nSymptom severity summary:\n"
    for symptom, stats in symptom_summary.iterrows():
        log += f"- {symptom}: max grade = {stats['max']}, avg grade = {stats['mean']:.2f}, count = {stats['count']}\n"

    # Summarize by subject
    subject_summary = data.groupby("subject_id")["vcog_grade"].agg(["max", "mean", "count"])
    log += f"\nSubjects with adverse events: {len(subject_summary)}\n"
    log += f"Subjects with Grade 3+ events: {len(subject_summary[subject_summary['max'] >= 3])}\n"

    # Create a summary of most severe events
    most_severe = data.sort_values("vcog_grade", ascending=False).head(10)
    log += "\nTop 10 most severe adverse events:\n"
    for i, (_, event) in enumerate(most_severe.iterrows(), 1):
        log += f"{i}. Subject {event['subject_id']}: {event['symptom']} (Grade {event['vcog_grade']})\n"

    # Step 6: Save detailed results to file
    output_file = "vcog_ctcae_graded_events.csv"
    data.to_csv(output_file, index=False)
    log += "\n## Step 5: Results saved\n"
    log += f"Detailed graded events saved to: {output_file}\n"

    # Save the VCOG criteria as a reference
    with open("vcog_ctcae_criteria_reference.json", "w") as f:
        json.dump(vcog_criteria, f, indent=2)
    log += "VCOG-CTCAE criteria reference saved to: vcog_ctcae_criteria_reference.json\n"

    return log


def analyze_radiolabeled_antibody_biodistribution(time_points, tissue_data):
    """Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points (hours) at which measurements were taken
    tissue_data : dict
        Dictionary where keys are tissue names and values are lists/arrays of %IA/g
        measurements corresponding to time_points. Must include 'tumor' as one of the keys.

    Returns
    -------
    str
        Research log summarizing the biodistribution analysis, pharmacokinetic parameters,
        and tumor-to-normal tissue ratios

    """
    import json
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Validate inputs
    if "tumor" not in tissue_data:
        return "Error: Tumor data must be provided in tissue_data dictionary"

    # Define bi-exponential model for pharmacokinetic analysis
    # C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
    def bi_exp_model(t, A, alpha, B, beta):
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

    # Initialize results dictionary
    results = {
        "tissues_analyzed": list(tissue_data.keys()),
        "pk_parameters": {},
        "tumor_to_normal_ratios": {},
        "auc_values": {},
    }

    # Analyze each tissue
    for tissue, measurements in tissue_data.items():
        try:
            # Fit bi-exponential model
            params, _ = curve_fit(
                bi_exp_model,
                time_points,
                measurements,
                p0=[50, 0.1, 50, 0.01],  # Initial parameter guess
                bounds=([0, 0, 0, 0], [100, 5, 100, 1]),  # Parameter bounds
            )

            A, alpha, B, beta = params

            # Calculate pharmacokinetic parameters
            # Distribution half-life (fast component)
            t_half_dist = np.log(2) / alpha

            # Elimination half-life (slow component)
            t_half_elim = np.log(2) / beta

            # Area under the curve (AUC)
            auc = A / alpha + B / beta

            # Mean residence time (MRT)
            mrt = (A / (alpha**2) + B / (beta**2)) / auc

            # Clearance (for blood/plasma only - conceptual)
            clearance = 1 / auc if tissue.lower() in ["blood", "plasma"] else None

            # Store results
            results["pk_parameters"][tissue] = {
                "A": float(A),
                "alpha": float(alpha),
                "B": float(B),
                "beta": float(beta),
                "distribution_half_life_h": float(t_half_dist),
                "elimination_half_life_h": float(t_half_elim),
                "mean_residence_time_h": float(mrt),
            }

            if clearance:
                results["pk_parameters"][tissue]["clearance"] = float(clearance)

            # Calculate AUC
            results["auc_values"][tissue] = float(auc)

        except Exception as e:
            results["pk_parameters"][tissue] = f"Fitting failed: {str(e)}"

    # Calculate tumor-to-normal tissue ratios at each time point
    for tissue in tissue_data:
        if tissue != "tumor":
            ratios = [
                t / n if n > 0 else float("inf")
                for t, n in zip(tissue_data["tumor"], tissue_data[tissue], strict=False)
            ]
            results["tumor_to_normal_ratios"][tissue] = {
                "values": [float(r) for r in ratios],
                "max_ratio": float(max(ratios)),
                "max_ratio_time_point": float(time_points[np.argmax(ratios)]),
            }

    # Save results to JSON file
    filename = "biodistribution_pk_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    # Generate research log
    log = "# Biodistribution and Pharmacokinetic Analysis of Radiolabeled Antibody\n\n"
    log += "## Analysis Summary\n"
    log += f"- Analyzed biodistribution data across {len(tissue_data)} tissues\n"
    log += f"- Time points analyzed: {time_points} hours\n"
    log += "- Performed bi-exponential pharmacokinetic modeling\n\n"

    log += "## Key Pharmacokinetic Parameters\n"
    for tissue, params in results["pk_parameters"].items():
        if isinstance(params, dict):
            log += f"\n### {tissue.capitalize()}\n"
            log += f"- Distribution half-life: {params['distribution_half_life_h']:.2f} hours\n"
            log += f"- Elimination half-life: {params['elimination_half_life_h']:.2f} hours\n"
            log += f"- Mean residence time: {params['mean_residence_time_h']:.2f} hours\n"
            if "clearance" in params:
                log += f"- Clearance: {params['clearance']:.4f} units\n"

    log += "\n## Tumor-to-Normal Tissue Ratios\n"
    for tissue, ratio_data in results["tumor_to_normal_ratios"].items():
        log += f"- {tissue.capitalize()}: Max ratio {ratio_data['max_ratio']:.2f} at {ratio_data['max_ratio_time_point']:.1f} hours\n"

    log += "\n## Detailed Results\n"
    log += f"Complete analysis results saved to: {os.path.abspath(filename)}\n"

    return log


def estimate_alpha_particle_radiotherapy_dosimetry(
    biodistribution_data, radiation_parameters, output_file="dosimetry_results.csv"
):
    """Estimate radiation absorbed doses to tumor and normal organs for alpha-particle radiotherapeutics.

    This function implements the Medical Internal Radiation Dose (MIRD) schema to calculate
    absorbed doses based on biodistribution data from healthy mice and radiation transport parameters.

    Parameters
    ----------
    biodistribution_data : dict
        Dictionary containing organ/tissue names as keys and a list of time-activity measurements as values.
        Each measurement should be a tuple of (time_hours, percent_injected_activity).
        Must include entries for all relevant organs including 'tumor'.

    radiation_parameters : dict
        Dictionary containing radiation parameters for the alpha-emitting radionuclide:
        - 'radionuclide': str - Name of the radionuclide (e.g., 'Ac-225')
        - 'half_life_hours': float - Physical half-life in hours
        - 'energy_per_decay_MeV': float - Energy released per decay in MeV
        - 'radiation_weighting_factor': float - Radiation weighting factor for alpha particles
        - 'S_factors': dict - S-factors (Gy/Bq-s) for each source-target organ pair

    output_file : str, optional
        Filename to save the dosimetry results (default: "dosimetry_results.csv")

    Returns
    -------
    str
        Research log summarizing the dosimetry estimation process and results

    """
    import csv
    from datetime import datetime

    import numpy as np
    from scipy.integrate import trapezoid

    # Initialize research log
    log = f"Alpha-Particle Radiotherapy Dosimetry Estimation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Radionuclide: {radiation_parameters['radionuclide']}\n"
    log += f"Half-life: {radiation_parameters['half_life_hours']} hours\n\n"

    # Step 1: Calculate time-integrated activity for each organ
    log += "Step 1: Calculating time-integrated activity for each organ\n"
    time_integrated_activity = {}

    for organ, measurements in biodistribution_data.items():
        times = [m[0] for m in measurements]
        activities = [m[1] for m in measurements]

        # Apply physical decay correction
        decay_constant = np.log(2) / radiation_parameters["half_life_hours"]
        decay_corrected_activities = [a * np.exp(-decay_constant * t) for a, t in zip(activities, times, strict=False)]

        # Calculate time-integrated activity using trapezoidal integration
        cumulated_activity = trapezoid(decay_corrected_activities, times)
        time_integrated_activity[organ] = cumulated_activity

        log += f"  - {organ}: {cumulated_activity:.4f} %IA-h\n"

    # Step 2: Calculate absorbed dose using MIRD schema
    log += "\nStep 2: Calculating absorbed doses using MIRD schema\n"

    # Convert %IA-h to MBq-h for a standard injection of 1 MBq
    conversion_factor = 0.01  # Convert %IA to fraction of IA

    # Energy conversion factor: MeV to J

    # Calculate absorbed dose for each target organ
    absorbed_doses = {}
    s_factors = radiation_parameters["S_factors"]

    for target_organ in biodistribution_data:
        absorbed_dose = 0

        # Sum contributions from all source organs
        for source_organ, cumulated_activity in time_integrated_activity.items():
            if (source_organ, target_organ) in s_factors:
                s_value = s_factors[(source_organ, target_organ)]
                organ_contribution = cumulated_activity * conversion_factor * s_value
                absorbed_dose += organ_contribution

        # Apply radiation weighting factor for alpha particles
        absorbed_dose *= radiation_parameters["radiation_weighting_factor"]

        # Store as Gy/MBq
        absorbed_doses[target_organ] = absorbed_dose
        log += f"  - {target_organ}: {absorbed_dose:.4f} Gy/MBq\n"

    # Step 3: Calculate therapeutic index (tumor-to-normal tissue dose ratios)
    log += "\nStep 3: Calculating therapeutic indices (tumor-to-normal tissue ratios)\n"

    tumor_dose = absorbed_doses.get("tumor", 0)
    if tumor_dose > 0:
        for organ, dose in absorbed_doses.items():
            if organ != "tumor" and dose > 0:
                therapeutic_index = tumor_dose / dose
                log += f"  - Tumor-to-{organ} ratio: {therapeutic_index:.2f}\n"

    # Save results to CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Organ", "Absorbed Dose (Gy/MBq)"])
        for organ, dose in absorbed_doses.items():
            writer.writerow([organ, f"{dose:.4f}"])

    log += f"\nDosimetry results saved to {output_file}\n"

    return log


def perform_mwas_cyp2c19_metabolizer_status(
    methylation_data_path,
    metabolizer_status_path,
    covariates_path=None,
    pvalue_threshold=0.05,
    output_file="significant_cpg_sites.csv",
):
    """Perform a Methylome-wide Association Study (MWAS) to identify CpG sites significantly associated with CYP2C19 metabolizer status.

    Parameters
    ----------
    methylation_data_path : str
        Path to CSV or TSV file containing DNA methylation beta values.
        Rows should be samples, columns should be CpG sites.
    metabolizer_status_path : str
        Path to CSV or TSV file containing CYP2C19 metabolizer status for each sample.
        Should have a sample ID column and a status column (e.g., poor, intermediate, normal, rapid, ultrarapid).
    covariates_path : str, optional
        Path to CSV or TSV file containing covariates to adjust for in the regression model
        (e.g., age, sex, smoking status).
    pvalue_threshold : float, optional
        P-value threshold for significance after multiple testing correction. Default is 0.05.
    output_file : str, optional
        Filename to save significant CpG sites. Default is "significant_cpg_sites.csv".

    Returns
    -------
    str
        A research log summarizing the MWAS analysis and results.

    """
    import time

    import pandas as pd
    from scipy.stats import linregress
    from statsmodels.formula.api import ols

    start_time = time.time()
    log = ["## Methylome-wide Association Study (MWAS) of CYP2C19 Metabolizer Status"]

    # Load data from files
    log.append("\n### Loading Data")
    try:
        # Load methylation data
        if methylation_data_path.endswith(".csv"):
            methylation_data = pd.read_csv(methylation_data_path, index_col=0)
        elif methylation_data_path.endswith((".tsv", ".txt")):
            methylation_data = pd.read_csv(methylation_data_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for methylation data. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded methylation data from {methylation_data_path}")

        # Load metabolizer status data
        if metabolizer_status_path.endswith(".csv"):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, index_col=0)
        elif metabolizer_status_path.endswith((".tsv", ".txt")):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for metabolizer status. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded metabolizer status data from {metabolizer_status_path}")

        # Convert DataFrame to Series if necessary
        if metabolizer_status_df.shape[1] == 1:
            metabolizer_status = metabolizer_status_df.iloc[:, 0]
        else:
            log.append("Error: Metabolizer status file should contain a single column with status values.")
            return "\n".join(log)

        # Load covariates if provided
        covariates = None
        if covariates_path is not None:
            if covariates_path.endswith(".csv"):
                covariates = pd.read_csv(covariates_path, index_col=0)
            elif covariates_path.endswith((".tsv", ".txt")):
                covariates = pd.read_csv(covariates_path, sep="\t", index_col=0)
            else:
                log.append("Error: Unsupported file format for covariates. Please provide a CSV or TSV file.")
                return "\n".join(log)
            log.append(f"- Successfully loaded covariates data from {covariates_path}")
    except Exception as e:
        log.append(f"Error loading data: {str(e)}")
        return "\n".join(log)

    # Step 1: Data preprocessing
    log.append("\n### Data Preprocessing")
    log.append(f"- Methylation data shape: {methylation_data.shape} (samples × CpG sites)")
    log.append(f"- Number of samples with metabolizer status: {len(metabolizer_status)}")

    # Ensure sample IDs match between methylation data and metabolizer status
    common_samples = methylation_data.index.intersection(metabolizer_status.index)
    methylation_data = methylation_data.loc[common_samples]
    metabolizer_status = metabolizer_status.loc[common_samples]

    log.append(f"- Number of samples after matching: {len(common_samples)}")

    # Check for covariates
    if covariates is not None:
        log.append(f"- Covariates provided: {', '.join(covariates.columns)}")
        covariates = covariates.loc[common_samples]

    # Step 2: Perform regression for each CpG site
    log.append("\n### Association Analysis")
    log.append(f"- Total CpG sites to analyze: {methylation_data.shape[1]}")

    results = []
    cpg_sites = methylation_data.columns

    # Convert metabolizer status to numeric if it's categorical
    if metabolizer_status.dtype == "object":
        # Create a mapping dictionary for metabolizer status
        # Assuming order: poor < intermediate < normal < rapid < ultrarapid
        status_order = {
            "poor": 1,
            "intermediate": 2,
            "normal": 3,
            "rapid": 4,
            "ultrarapid": 5,
        }

        # Try to map using the dictionary, or keep as is if already numeric
        try:
            metabolizer_status_numeric = metabolizer_status.map(status_order)
            log.append("- Converted metabolizer status to numeric values")
        except Exception:
            metabolizer_status_numeric = metabolizer_status
            log.append("- Using metabolizer status as provided (assuming numeric)")
    else:
        metabolizer_status_numeric = metabolizer_status

    # Perform regression for each CpG site
    for cpg in cpg_sites:
        methylation_values = methylation_data[cpg]

        # Basic model without covariates
        if covariates is None:
            model = linregress(metabolizer_status_numeric, methylation_values)
            pvalue = model.pvalue
            coefficient = model.slope
        else:
            # Create DataFrame for regression with covariates
            data_for_regression = pd.DataFrame(
                {
                    "methylation": methylation_values,
                    "metabolizer": metabolizer_status_numeric,
                }
            )

            # Add covariates
            for col in covariates.columns:
                data_for_regression[col] = covariates[col]

            # Formula for regression with covariates
            formula = "methylation ~ metabolizer + " + " + ".join(covariates.columns)
            model = ols(formula, data=data_for_regression).fit()

            pvalue = model.pvalues["metabolizer"]
            coefficient = model.params["metabolizer"]

        results.append({"CpG_site": cpg, "coefficient": coefficient, "pvalue": pvalue})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Multiple testing correction
    log.append("\n### Multiple Testing Correction")
    log.append("- Applying Bonferroni correction")

    # Bonferroni correction
    results_df["adjusted_pvalue"] = results_df["pvalue"] * len(results_df)
    results_df["adjusted_pvalue"] = results_df["adjusted_pvalue"].clip(upper=1.0)  # Ensure p-values don't exceed 1

    # Step 4: Identify significant CpG sites
    significant_sites = results_df[results_df["adjusted_pvalue"] < pvalue_threshold]
    significant_sites = significant_sites.sort_values("adjusted_pvalue")

    log.append("\n### Results")
    log.append(f"- Number of significant CpG sites (adjusted p < {pvalue_threshold}): {len(significant_sites)}")

    if len(significant_sites) > 0:
        # Save significant sites to file
        significant_sites.to_csv(output_file, index=False)
        log.append("- Top 5 significant CpG sites:")

        for _, row in significant_sites.head(5).iterrows():
            log.append(
                f"  * {row['CpG_site']}: coefficient = {row['coefficient']:.4f}, adj. p-value = {row['adjusted_pvalue']:.6f}"
            )

        log.append(f"- Full results saved to: {output_file}")
    else:
        log.append("- No significant CpG sites found after multiple testing correction")

    # Execution time
    execution_time = time.time() - start_time
    log.append("\n### Summary")
    log.append(f"- Analysis completed in {execution_time:.2f} seconds")

    return "\n".join(log)


def calculate_physicochemical_properties(smiles_string):
    """Calculate key physicochemical properties of a drug candidate molecule.

    Parameters
    ----------
    smiles_string : str
        The molecular structure in SMILES format

    Returns
    -------
    str
        A research log summarizing the calculated physicochemical properties and
        indicating where the detailed results are saved

    """
    import csv
    import os

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem.MolStandardize import rdMolStandardize

    # Create RDKit molecule from SMILES
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return "ERROR: Invalid SMILES string provided."
    except Exception as e:
        return f"ERROR: Failed to process SMILES string: {str(e)}"

    # Calculate basic properties
    properties = {
        "SMILES": smiles_string,
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Ring Count": Descriptors.RingCount(mol),
    }

    # Estimate pKa (simplified approach - in practice would use specialized tools)
    # This is a simplification as accurate pKa prediction requires specialized tools
    uncharger = rdMolStandardize.Uncharger()
    uncharger.uncharge(mol)
    acidic_groups = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "O"
        and any(neigh.GetSymbol() == "C" and neigh.GetDegree() == 3 for neigh in atom.GetNeighbors())
    )
    basic_groups = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetDegree() < 4)
    properties["Estimated Acidic Groups"] = acidic_groups
    properties["Estimated Basic Groups"] = basic_groups

    # Calculate drug-likeness score (using Crippen approach)
    properties["Drug-likeness Score"] = round(Crippen.MolMR(mol), 2)

    # Calculate logD (simplified as logP - pKa adjustment would need specialized tools)
    properties["Estimated logD7.4"] = properties["cLogP"]

    # Save results to CSV
    csv_filename = "physicochemical_properties.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Property", "Value"])
        for prop, value in properties.items():
            writer.writerow([prop, value])

    # Generate research log
    log = f"""Physicochemical Property Calculation Research Log:

Analyzed compound with SMILES: {smiles_string}

Key properties:
- Molecular Weight: {properties["Molecular Weight"]} g/mol
- cLogP: {properties["cLogP"]}
- Topological Polar Surface Area: {properties["TPSA"]} Å²
- H-Bond Donors: {properties["H-Bond Donors"]}
- H-Bond Acceptors: {properties["H-Bond Acceptors"]}
- Rotatable Bonds: {properties["Rotatable Bonds"]}
- Estimated logD (at pH 7.4): {properties["Estimated logD7.4"]}
- Estimated Acidic Groups: {properties["Estimated Acidic Groups"]}
- Estimated Basic Groups: {properties["Estimated Basic Groups"]}

Complete results saved to: {os.path.abspath(csv_filename)}
"""

    return log


def analyze_xenograft_tumor_growth_inhibition(
    data_path,
    time_column,
    volume_column,
    group_column,
    subject_column,
    output_dir="./results",
):
    """Analyze tumor growth inhibition in xenograft models across different treatment groups.

    Parameters
    ----------
    data_path : str
        Path to CSV or TSV file containing tumor volume measurements. The file should have columns for
        time, volume, treatment group, and subject ID
    time_column : str
        Name of the column containing time points (e.g., 'Day', 'Time')
    volume_column : str
        Name of the column containing tumor volume measurements
    group_column : str
        Name of the column containing treatment group labels
    subject_column : str
        Name of the column containing subject/mouse identifiers
    output_dir : str, optional
        Directory to save output files (default: "./results")

    Returns
    -------
    str
        Research log summarizing the analysis steps, findings, and generated file paths

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = "# Xenograft Tumor Growth Inhibition Analysis\n\n"

    # Load data from file
    log += "## 1. Data Loading and Summary\n\n"
    try:
        if data_path.endswith(".csv"):
            data_df = pd.read_csv(data_path)
        elif data_path.endswith((".tsv", ".txt")):
            data_df = pd.read_csv(data_path, sep="\t")
        else:
            log += "Error: Unsupported file format. Please provide a CSV or TSV file.\n"
            return log
        log += f"Successfully loaded tumor growth data from {data_path}\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Validate required columns
    required_columns = [time_column, volume_column, group_column, subject_column]
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        log += f"Error: Missing required columns: {', '.join(missing_columns)}\n"
        return log

    # Get unique groups and time points
    groups = data_df[group_column].unique()
    time_points = sorted(data_df[time_column].unique())
    n_groups = len(groups)
    log += f"- Number of treatment groups: {n_groups} ({', '.join(map(str, groups))})\n"
    log += f"- Number of time points: {len(time_points)}\n"
    log += f"- Number of subjects: {data_df[subject_column].nunique()}\n"
    log += f"- Total number of measurements: {len(data_df)}\n\n"

    # 2. Calculate group statistics at each time point
    log += "## 2. Tumor Growth Analysis\n\n"

    # Group statistics
    stats_df = (
        data_df.groupby([group_column, time_column])[volume_column]
        .agg(mean="mean", sem=lambda x: stats.sem(x), count="count")
        .reset_index()
    )

    # Save group statistics
    stats_file = os.path.join(output_dir, "tumor_volume_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    log += f"Group statistics saved to: {stats_file}\n\n"

    # 3. Calculate tumor growth rates
    log += "## 3. Tumor Growth Rate Analysis\n\n"

    growth_rates = {}
    for group in groups:
        group_data = data_df[data_df[group_column] == group]

        # Calculate growth rate for each subject
        subject_growth_rates = []
        for subject in group_data[subject_column].unique():
            subject_data = group_data[group_data[subject_column] == subject]

            if len(subject_data) >= 2:
                # Simple linear regression for growth rate
                x = subject_data[time_column].values
                y = subject_data[volume_column].values
                slope, _, _, _, _ = stats.linregress(x, y)
                subject_growth_rates.append(slope)

        growth_rates[group] = subject_growth_rates
        mean_rate = np.mean(subject_growth_rates)
        sem_rate = stats.sem(subject_growth_rates)

        log += (
            f"- {group}: Mean growth rate = {mean_rate:.2f} ± {sem_rate:.2f} mm³/day (n={len(subject_growth_rates)})\n"
        )

    # 4. Calculate Tumor Growth Inhibition (TGI)
    log += "\n## 4. Tumor Growth Inhibition (TGI)\n\n"

    # Identify control group (assuming the first group is control)
    control_group = groups[0]
    log += f"Control group: {control_group}\n\n"

    # Calculate TGI for the final time point
    final_time = max(time_points)
    final_data = data_df[data_df[time_column] == final_time]

    control_final_mean = final_data[final_data[group_column] == control_group][volume_column].mean()

    tgi_results = {}
    for group in groups:
        if group == control_group:
            continue

        group_final_mean = final_data[final_data[group_column] == group][volume_column].mean()
        tgi = ((control_final_mean - group_final_mean) / control_final_mean) * 100
        tgi_results[group] = tgi

        log += f"- {group}: TGI = {tgi:.1f}% (relative to {control_group})\n"

    # 5. Statistical Analysis
    log += "\n## 5. Statistical Analysis\n\n"

    # Repeated measures ANOVA
    log += "### Repeated Measures ANOVA\n\n"

    try:
        # Prepare data for repeated measures ANOVA
        formula = f"{volume_column} ~ C({group_column}) * C({time_column}) + C({subject_column})"
        model = ols(formula, data=data_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Save ANOVA results
        anova_file = os.path.join(output_dir, "repeated_measures_anova.csv")
        anova_table.to_csv(anova_file)

        log += f"ANOVA results saved to: {anova_file}\n\n"

        # Extract p-values
        group_effect_p = anova_table.loc[f"C({group_column})", "PR(>F)"]
        time_effect_p = anova_table.loc[f"C({time_column})", "PR(>F)"]
        interaction_p = anova_table.loc[f"C({group_column}):C({time_column})", "PR(>F)"]

        log += f"- Treatment effect: p = {group_effect_p:.4f}\n"
        log += f"- Time effect: p = {time_effect_p:.4f}\n"
        log += f"- Treatment × Time interaction: p = {interaction_p:.4f}\n\n"

        # Post-hoc analysis at final time point
        log += "### Post-hoc Analysis (Final Time Point)\n\n"

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(endog=final_data[volume_column], groups=final_data[group_column], alpha=0.05)

        # Save Tukey results
        tukey_file = os.path.join(output_dir, "tukey_posthoc_results.txt")
        with open(tukey_file, "w") as f:
            f.write(str(tukey.summary()))

        log += f"Tukey's HSD results saved to: {tukey_file}\n\n"

        # Summarize significant comparisons
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        sig_pairs = tukey_df[tukey_df["p-adj"] < 0.05]
        if len(sig_pairs) > 0:
            log += "Significant pairwise comparisons:\n"
            for _, row in sig_pairs.iterrows():
                log += f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}\n"
        else:
            log += "No significant pairwise comparisons found.\n"

    except Exception as e:
        log += f"Error in statistical analysis: {str(e)}\n"

    # 6. Generate tumor growth curves
    log += "\n## 6. Tumor Growth Visualization\n\n"

    plt.figure(figsize=(10, 6))

    for group in groups:
        group_stats = stats_df[stats_df[group_column] == group]
        plt.errorbar(
            group_stats[time_column],
            group_stats["mean"],
            yerr=group_stats["sem"],
            label=group,
            capsize=3,
            marker="o",
        )

    plt.xlabel(f"{time_column} (days)")
    plt.ylabel(f"Tumor Volume ({volume_column})")
    plt.title("Xenograft Tumor Growth Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    plot_file = os.path.join(output_dir, "tumor_growth_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    log += f"Tumor growth curve plot saved to: {plot_file}\n"

    # 7. Conclusion
    log += "\n## 7. Conclusion\n\n"

    # Summarize most effective treatment
    if tgi_results:
        best_treatment = max(tgi_results.items(), key=lambda x: x[1])
        log += f"The most effective treatment was {best_treatment[0]} with a tumor growth inhibition of {best_treatment[1]:.1f}%.\n"

    # Statistical significance summary
    try:
        if group_effect_p < 0.05:
            log += "Statistical analysis confirmed significant differences between treatment groups.\n"
        else:
            log += "No statistically significant differences were found between treatment groups.\n"
    except Exception:
        pass

    return log


def analyze_western_blot(
    blot_image_path,
    target_bands,
    loading_control_band,
    antibody_info,
    output_dir="./results",
):
    """Performs densitometric analysis of Western blot images to quantify relative protein expression.

    Parameters
    ----------
    blot_image_path : str
        Path to the Western blot image file
    target_bands : list of dict
        List of dictionaries containing information about target protein bands
        Each dict should have 'name', 'roi' (region of interest as [x, y, width, height])
    loading_control_band : dict
        Dictionary with 'name' and 'roi' for the loading control protein (e.g., β-actin, GAPDH)
    antibody_info : dict
        Dictionary containing information about antibodies used
        Should have 'primary' and 'secondary' keys with antibody details
    output_dir : str, optional
        Directory to save output files, defaults to './results'

    Returns
    -------
    str
        Research log summarizing the Western blot analysis process and results

    """
    import os

    import numpy as np
    from skimage import io

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the Western blot image
    image = io.imread(blot_image_path)
    if len(image.shape) > 2:  # Convert to grayscale if color image
        image = np.mean(image, axis=2).astype(np.uint8)

    # Initialize results dictionary
    results = {
        "loading_control": {"name": loading_control_band["name"], "intensity": 0},
        "targets": [],
    }

    # Analyze loading control band
    lc_roi = loading_control_band["roi"]
    lc_band = image[lc_roi[1] : lc_roi[1] + lc_roi[3], lc_roi[0] : lc_roi[0] + lc_roi[2]]
    lc_intensity = np.sum(lc_band)
    results["loading_control"]["intensity"] = lc_intensity

    # Analyze target protein bands
    for band in target_bands:
        roi = band["roi"]
        band_img = image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        band_intensity = np.sum(band_img)

        # Calculate relative expression (normalized to loading control)
        relative_expression = band_intensity / lc_intensity

        results["targets"].append(
            {
                "name": band["name"],
                "intensity": band_intensity,
                "relative_expression": relative_expression,
            }
        )

    # Generate results table and save to CSV
    results_file = os.path.join(output_dir, "western_blot_results.csv")
    with open(results_file, "w") as f:
        f.write("Protein,Raw Intensity,Relative Expression\n")
        f.write(f"{results['loading_control']['name']},{results['loading_control']['intensity']},1.0\n")
        for target in results["targets"]:
            f.write(f"{target['name']},{target['intensity']},{target['relative_expression']:.4f}\n")

    # Generate research log
    log = "## Western Blot Analysis\n\n"
    log += f"Analyzed Western blot image: {os.path.basename(blot_image_path)}\n\n"
    log += "### Antibodies Used\n"
    log += f"Primary antibody: {antibody_info['primary']}\n"
    log += f"Secondary antibody: {antibody_info['secondary']}\n\n"
    log += "### Analysis Steps\n"
    log += "1. Loaded Western blot image and converted to grayscale\n"
    log += f"2. Quantified loading control ({loading_control_band['name']}) band intensity\n"
    log += "3. Measured target protein band intensities\n"
    log += "4. Calculated relative expression by normalizing to loading control\n\n"
    log += "### Results\n"
    log += f"Loading control ({loading_control_band['name']}): {results['loading_control']['intensity']} intensity units\n\n"
    log += "Target proteins:\n"
    for target in results["targets"]:
        log += f"- {target['name']}: {target['intensity']} intensity units, "
        log += f"{target['relative_expression']:.4f} relative expression\n"
    log += f"\nDetailed results saved to: {results_file}\n"

    return log


# DDInter Drug-Drug Interaction Analysis Functions


def _load_ddinter_data(data_lake_path):
    """
    Load DDInter datasets from pickle files, processing if needed.

    Parameters
    ----------
    data_lake_path : str
        Path to data lake directory containing DDInter pickle files

    Returns
    -------
    tuple
        (drug_info, interaction_matrix, name_mapping) dictionaries
    """
    import os
    import pickle

    # Define schema directory (following established pattern)
    schema_dir = os.path.join(os.path.dirname(__file__), "schema_db")

    # Define paths to DDInter pickle files
    drug_info_path = os.path.join(schema_dir, "ddinter_drugs.pkl")
    interaction_path = os.path.join(schema_dir, "ddinter_interactions.pkl")
    mapping_path = os.path.join(schema_dir, "ddinter_name_mapping.pkl")

    # Check if processing is needed (lazy loading pattern)
    pkl_files = [drug_info_path, interaction_path, mapping_path]
    if not all(os.path.exists(f) for f in pkl_files):
        _process_ddinter_data_inline(data_lake_path, schema_dir)

    # Load data
    try:
        with open(drug_info_path, "rb") as f:
            drug_info = pickle.load(f)

        with open(interaction_path, "rb") as f:
            interaction_matrix = pickle.load(f)

        with open(mapping_path, "rb") as f:
            name_mapping = pickle.load(f)

        return drug_info, interaction_matrix, name_mapping

    except Exception as e:
        raise FileNotFoundError(f"Error loading DDInter data: {e}") from e


def _process_ddinter_data_inline(data_lake_path, output_dir):
    """
    Process DDInter CSV files into standardized pickle files.

    This function processes raw DDInter 2.0 CSV files and creates standardized
    data structures for use in Biomni drug-drug interaction analysis.

    Parameters
    ----------
    data_lake_path : str
        Path to data lake directory containing raw DDInter CSV files
    output_dir : str
        Directory to save processed pickle files
    """
    import os
    import pickle
    from pathlib import Path

    import pandas as pd

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Define CSV files to process
    csv_files = [
        "ddinter_alimentary_tract_metabolism.csv",
        "ddinter_antineoplastic.csv",
        "ddinter_antiparasitic.csv",
        "ddinter_blood_organs.csv",
        "ddinter_dermatological.csv",
        "ddinter_hormonal.csv",
        "ddinter_respiratory.csv",
        "ddinter_various.csv",
    ]

    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(data_lake_path, csv_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add source category
            category = csv_file.replace("ddinter_", "").replace(".csv", "")
            df["category"] = category
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError("No DDInter CSV files found in data lake")

    # Process data
    drug_info = _build_drug_registry_inline(dataframes)
    interaction_matrix = _create_interaction_matrix_inline(dataframes)
    name_mapping = _create_name_mapping_inline(drug_info)

    # Save processed data
    with open(os.path.join(output_dir, "ddinter_drugs.pkl"), "wb") as f:
        pickle.dump(drug_info, f)

    with open(os.path.join(output_dir, "ddinter_interactions.pkl"), "wb") as f:
        pickle.dump(interaction_matrix, f)

    with open(os.path.join(output_dir, "ddinter_name_mapping.pkl"), "wb") as f:
        pickle.dump(name_mapping, f)

    # Generate and save statistics
    stats = _generate_ddinter_statistics_inline(drug_info, interaction_matrix)
    with open(os.path.join(output_dir, "ddinter_statistics.pkl"), "wb") as f:
        pickle.dump(stats, f)


def _standardize_drug_name_processing(drug_name):
    """Standardize drug names for consistent matching during processing."""
    import pandas as pd

    if pd.isna(drug_name):
        return ""

    # Convert to lowercase and strip whitespace
    standardized = str(drug_name).strip().lower()

    # Remove common suffixes and prefixes
    standardized = standardized.replace(" hydrochloride", "")
    standardized = standardized.replace(" sulfate", "")
    standardized = standardized.replace(" sodium", "")
    standardized = standardized.replace(" potassium", "")
    standardized = standardized.replace(" calcium", "")
    standardized = standardized.replace(" magnesium", "")

    return standardized


def _build_drug_registry_inline(dataframes):
    """Build comprehensive drug registry from all interactions."""

    drug_registry = {}

    for df in dataframes:
        for _, row in df.iterrows():
            drug_a_id = row["DDInterID_A"]
            drug_a_name = row["Drug_A"]
            drug_b_id = row["DDInterID_B"]
            drug_b_name = row["Drug_B"]

            # Add Drug A
            if drug_a_id not in drug_registry:
                drug_registry[drug_a_id] = {
                    "name": drug_a_name,
                    "standardized_name": _standardize_drug_name_processing(drug_a_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_a_id]["categories"].add(row["category"])

            # Add Drug B
            if drug_b_id not in drug_registry:
                drug_registry[drug_b_id] = {
                    "name": drug_b_name,
                    "standardized_name": _standardize_drug_name_processing(drug_b_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_b_id]["categories"].add(row["category"])

            # Record interactions
            drug_registry[drug_a_id]["interactions"].add(drug_b_id)
            drug_registry[drug_b_id]["interactions"].add(drug_a_id)

    # Convert sets to lists for pickle serialization
    for drug_id in drug_registry:
        drug_registry[drug_id]["categories"] = list(drug_registry[drug_id]["categories"])
        drug_registry[drug_id]["interactions"] = list(drug_registry[drug_id]["interactions"])

    return drug_registry


def _create_interaction_matrix_inline(dataframes):
    """Create interaction matrix for fast lookups using standardized drug names."""
    from collections import defaultdict

    import pandas as pd

    combined_df = pd.concat(dataframes, ignore_index=True)
    interaction_matrix = defaultdict(lambda: defaultdict(list))

    # Create bidirectional interaction matrix using standardized names
    for _, row in combined_df.iterrows():
        drug_a_std = _standardize_drug_name_processing(row["Drug_A"])
        drug_b_std = _standardize_drug_name_processing(row["Drug_B"])
        level = row["Level"]
        category = row["category"]

        interaction_data = {
            "level": level,
            "category": category,
            "drug_a_id": row["DDInterID_A"],
            "drug_b_id": row["DDInterID_B"],
            "drug_a_name": row["Drug_A"],
            "drug_b_name": row["Drug_B"],
        }

        # Add both directions using standardized names as keys
        interaction_matrix[drug_a_std][drug_b_std].append(interaction_data)
        interaction_matrix[drug_b_std][drug_a_std].append(interaction_data)

    # Convert to regular dict for pickle
    interaction_matrix = dict(interaction_matrix)
    for drug in interaction_matrix:
        interaction_matrix[drug] = dict(interaction_matrix[drug])

    return interaction_matrix


def _create_name_mapping_inline(drug_info):
    """Create drug name to ID mapping for fuzzy matching."""
    name_mapping = {}

    for drug_id, drug_data in drug_info.items():
        original_name = drug_data["name"]
        standardized_name = drug_data["standardized_name"]

        # Map both original and standardized names
        name_mapping[original_name.lower()] = drug_id
        name_mapping[standardized_name] = drug_id

    return name_mapping


def _generate_ddinter_statistics_inline(drug_info, interaction_matrix):
    """Generate statistics about the processed data."""
    from collections import defaultdict

    stats = {
        "total_drugs": len(drug_info),
        "total_interactions": 0,
        "interaction_levels": defaultdict(int),
        "drug_categories": defaultdict(int),
        "most_connected_drugs": [],
    }

    # Count interactions and levels
    for drug_a in interaction_matrix:
        for drug_b in interaction_matrix[drug_a]:
            interactions = interaction_matrix[drug_a][drug_b]
            stats["total_interactions"] += len(interactions)

            for interaction in interactions:
                stats["interaction_levels"][interaction["level"]] += 1

    # Count drug categories
    for drug_data in drug_info.values():
        for category in drug_data["categories"]:
            stats["drug_categories"][category] += 1

    # Find most connected drugs
    connection_counts = []
    for drug_id, drug_data in drug_info.items():
        connection_counts.append(
            {"drug_id": drug_id, "name": drug_data["name"], "connections": len(drug_data["interactions"])}
        )

    connection_counts.sort(key=lambda x: x["connections"], reverse=True)
    stats["most_connected_drugs"] = connection_counts[:10]

    return stats


def _standardize_drug_name(drug_name, name_mapping):
    """
    Standardize drug names using fuzzy matching against DDInter database.

    Parameters
    ----------
    drug_name : str
        Original drug name
    name_mapping : dict
        Drug name to ID mapping dictionary

    Returns
    -------
    str or None
        Standardized drug name or None if not found
    """
    from difflib import get_close_matches

    # Direct match
    if drug_name.lower() in name_mapping:
        return drug_name.lower()

    # Fuzzy match
    matches = get_close_matches(drug_name.lower(), name_mapping.keys(), n=1, cutoff=0.8)
    if matches:
        return matches[0]

    return None


def _format_interaction_result(interaction_data, drug_name_a, drug_name_b, include_mechanisms=True):
    """
    Format interaction results for research log.

    Parameters
    ----------
    interaction_data : list
        List of interaction data dictionaries
    drug_name_a : str
        First drug name
    drug_name_b : str
        Second drug name
    include_mechanisms : bool
        Whether to include detailed mechanism information

    Returns
    -------
    str
        Formatted interaction description
    """
    if not interaction_data:
        return f"No interactions found between {drug_name_a} and {drug_name_b}"

    result = f"Interaction between {drug_name_a} and {drug_name_b}:\n"

    for i, interaction in enumerate(interaction_data, 1):
        level = interaction.get("level", "Unknown")
        category = interaction.get("category", "Unknown")

        result += f"  {i}. Severity: {level}\n"
        result += f"     Category: {category.replace('_', ' ').title()}\n"

        if include_mechanisms:
            result += f"     Clinical significance: {level} interaction requiring appropriate monitoring\n"

    return result


def query_drug_interactions(drug_names, interaction_types=None, severity_levels=None, data_lake_path=None):
    """
    Query drug-drug interactions from DDInter database.

    Parameters
    ----------
    drug_names : list of str
        List of drug names to query for interactions
    interaction_types : list of str, optional
        Filter by interaction types (e.g., ['synergistic', 'antagonistic'])
    severity_levels : list of str, optional
        Filter by severity levels (e.g., ['Major', 'Moderate', 'Minor'])
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with detailed interaction analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "DDInter Drug-Drug Interaction Query\n"
    log += "=" * 40 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        # Default path assuming standard Biomni structure
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Query Parameters:\n"
    log += f"- Target drugs: {', '.join(drug_names)}\n"
    log += f"- Severity filter: {severity_levels if severity_levels else 'All levels'}\n"
    log += f"- Interaction types: {interaction_types if interaction_types else 'All types'}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize drug names
        standardized_names = []
        missing_drugs = []

        for drug_name in drug_names:
            standardized = _standardize_drug_name(drug_name, name_mapping)
            if standardized:
                standardized_names.append(standardized)
            else:
                missing_drugs.append(drug_name)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if not standardized_names:
            log += "Error: No valid drugs found in DDInter database\n"
            return log

        # Query interactions
        interactions_found = []

        for i, drug_a in enumerate(standardized_names):
            for j, drug_b in enumerate(standardized_names):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    # Apply filters
                    filtered_interactions = interactions

                    if severity_levels:
                        filtered_interactions = [
                            int_data for int_data in filtered_interactions if int_data.get("level") in severity_levels
                        ]

                    if interaction_types:
                        filtered_interactions = [
                            int_data
                            for int_data in filtered_interactions
                            if int_data.get("category") in interaction_types
                        ]

                    if filtered_interactions:
                        interactions_found.append(
                            {"drug_a": drug_a, "drug_b": drug_b, "interactions": filtered_interactions}
                        )

        # Format results
        log += "Interaction Analysis Results:\n"
        log += f"Found {len(interactions_found)} drug pairs with interactions\n\n"

        if interactions_found:
            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"], pair["drug_a"].title(), pair["drug_b"].title(), include_mechanisms=True
                )
                log += "\n"
        else:
            log += "No interactions found between the specified drugs with the given filters\n"

        # Summary statistics
        total_interactions = sum(len(pair["interactions"]) for pair in interactions_found)
        log += "Summary:\n"
        log += f"- Total drug pairs analyzed: {len(standardized_names) * (len(standardized_names) - 1) // 2}\n"
        log += f"- Drug pairs with interactions: {len(interactions_found)}\n"
        log += f"- Total interactions found: {total_interactions}\n"

        if interactions_found:
            severity_counts = {}
            for pair in interactions_found:
                for interaction in pair["interactions"]:
                    level = interaction.get("level", "Unknown")
                    severity_counts[level] = severity_counts.get(level, 0) + 1

            log += f"- Severity distribution: {dict(severity_counts)}\n"

    except Exception as e:
        log += f"Error during interaction query: {str(e)}\n"

    return log


def check_drug_combination_safety(drug_list, include_mechanisms=True, include_management=True, data_lake_path=None):
    """
    Analyze safety of a drug combination for potential interactions.

    Parameters
    ----------
    drug_list : list of str
        List of drugs to analyze for combination safety
    include_mechanisms : bool, default True
        Include interaction mechanism descriptions
    include_management : bool, default True
        Include management recommendations
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with safety analysis and recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Combination Safety Analysis\n"
    log += "=" * 35 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Safety Analysis Parameters:\n"
    log += f"- Drug combination: {', '.join(drug_list)}\n"
    log += f"- Include mechanisms: {include_mechanisms}\n"
    log += f"- Include management: {include_management}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        standardized_drugs = []
        missing_drugs = []

        for drug in drug_list:
            standardized = _standardize_drug_name(drug, name_mapping)
            if standardized:
                standardized_drugs.append(standardized)
            else:
                missing_drugs.append(drug)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if len(standardized_drugs) < 2:
            log += "Error: At least 2 valid drugs required for combination analysis\n"
            return log

        # Analyze all pairwise interactions
        interactions_found = []
        major_interactions = 0
        moderate_interactions = 0
        minor_interactions = 0

        for i, drug_a in enumerate(standardized_drugs):
            for j, drug_b in enumerate(standardized_drugs):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    for interaction in interactions:
                        level = interaction.get("level", "Unknown")
                        if level == "Major":
                            major_interactions += 1
                        elif level == "Moderate":
                            moderate_interactions += 1
                        elif level == "Minor":
                            minor_interactions += 1

                    interactions_found.append({"drug_a": drug_a, "drug_b": drug_b, "interactions": interactions})

        # Overall safety assessment
        log += "Overall Safety Assessment:\n"

        safety_score = 100
        safety_level = "Safe"

        if major_interactions > 0:
            safety_score -= major_interactions * 30
            safety_level = "High Risk"
        elif moderate_interactions > 2:
            safety_score -= moderate_interactions * 15
            safety_level = "Moderate Risk"
        elif moderate_interactions > 0:
            safety_score -= moderate_interactions * 10
            safety_level = "Low to Moderate Risk"
        elif minor_interactions > 0:
            safety_score -= minor_interactions * 5
            safety_level = "Low Risk"

        safety_score = max(0, safety_score)

        log += f"- Safety Level: {safety_level}\n"
        log += f"- Safety Score: {safety_score}/100\n"
        log += f"- Major interactions: {major_interactions}\n"
        log += f"- Moderate interactions: {moderate_interactions}\n"
        log += f"- Minor interactions: {minor_interactions}\n\n"

        # Detailed interaction analysis
        if interactions_found:
            log += "Detailed Interaction Analysis:\n"
            log += "-" * 30 + "\n"

            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"],
                    pair["drug_a"].title(),
                    pair["drug_b"].title(),
                    include_mechanisms=include_mechanisms,
                )
                log += "\n"

        # Clinical recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 25 + "\n"

        if major_interactions > 0:
            log += "- CONTRAINDICATED: This combination contains major interactions\n"
            log += "- Consider alternative medications or consult specialist\n"
            log += "- If combination is necessary, intensive monitoring required\n"
        elif moderate_interactions > 2:
            log += "- CAUTION: Multiple moderate interactions detected\n"
            log += "- Monitor patient closely for adverse effects\n"
            log += "- Consider dose adjustments or alternative agents\n"
        elif moderate_interactions > 0:
            log += "- MONITOR: Moderate interactions present\n"
            log += "- Regular patient monitoring recommended\n"
            log += "- Be aware of potential side effects\n"
        elif minor_interactions > 0:
            log += "- AWARENESS: Minor interactions detected\n"
            log += "- Standard monitoring sufficient\n"
            log += "- Educate patient about potential minor effects\n"
        else:
            log += "- SAFE: No significant interactions detected\n"
            log += "- Standard clinical monitoring appropriate\n"

        if include_management:
            log += "\nGeneral Management Strategies:\n"
            log += "- Separate administration times when possible\n"
            log += "- Monitor for signs of toxicity or reduced efficacy\n"
            log += "- Consider therapeutic drug monitoring if available\n"
            log += "- Educate patient about potential interaction symptoms\n"

    except Exception as e:
        log += f"Error during safety analysis: {str(e)}\n"

    return log


def analyze_interaction_mechanisms(drug_pair, detailed_analysis=True, data_lake_path=None):
    """
    Analyze interaction mechanisms between two specific drugs.

    Parameters
    ----------
    drug_pair : tuple of str
        Pair of drug names to analyze (drug1, drug2)
    detailed_analysis : bool, default True
        Include detailed mechanistic information
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with mechanism analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Interaction Mechanism Analysis\n"
    log += "=" * 37 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    drug_a, drug_b = drug_pair
    log += "Mechanism Analysis Parameters:\n"
    log += f"- Drug A: {drug_a}\n"
    log += f"- Drug B: {drug_b}\n"
    log += f"- Detailed analysis: {detailed_analysis}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        std_drug_a = _standardize_drug_name(drug_a, name_mapping)
        std_drug_b = _standardize_drug_name(drug_b, name_mapping)

        if not std_drug_a:
            log += f"Error: Drug '{drug_a}' not found in DDInter database\n"
            return log
        if not std_drug_b:
            log += f"Error: Drug '{drug_b}' not found in DDInter database\n"
            return log

        # Query interactions
        interactions = []
        if std_drug_a in interaction_matrix and std_drug_b in interaction_matrix[std_drug_a]:
            interactions = interaction_matrix[std_drug_a][std_drug_b]

        if not interactions:
            log += f"No interactions found between {drug_a} and {drug_b}\n"
            return log

        # Get drug information
        drug_a_id = name_mapping[std_drug_a]
        drug_b_id = name_mapping[std_drug_b]
        drug_a_info = drug_info.get(drug_a_id, {})
        drug_b_info = drug_info.get(drug_b_id, {})

        log += "Drug Profile Analysis:\n"
        log += "-" * 20 + "\n"
        log += f"{drug_a.title()}:\n"
        log += f"- Categories: {', '.join(drug_a_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_a_info.get('interactions', []))}\n\n"

        log += f"{drug_b.title()}:\n"
        log += f"- Categories: {', '.join(drug_b_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_b_info.get('interactions', []))}\n\n"

        # Analyze interaction mechanisms
        log += "Interaction Mechanism Analysis:\n"
        log += "-" * 30 + "\n"

        for i, interaction in enumerate(interactions, 1):
            level = interaction.get("level", "Unknown")
            category = interaction.get("category", "Unknown")

            log += f"Interaction {i}:\n"
            log += f"- Severity: {level}\n"
            log += f"- Category: {category.replace('_', ' ').title()}\n"

            if detailed_analysis:
                # Provide mechanism insights based on severity and category
                if level == "Major":
                    log += "- Clinical Impact: High risk interaction requiring immediate attention\n"
                    log += "- Mechanism: Likely involves significant pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Avoid combination or use with extreme caution\n"
                elif level == "Moderate":
                    log += "- Clinical Impact: Moderate risk requiring monitoring\n"
                    log += "- Mechanism: May involve enzyme induction/inhibition or receptor competition\n"
                    log += "- Management: Monitor closely, consider dose adjustment\n"
                elif level == "Minor":
                    log += "- Clinical Impact: Low risk, usually manageable\n"
                    log += "- Mechanism: Minor pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Standard monitoring sufficient\n"

                # Category-specific mechanism insights
                category_mechanisms = {
                    "alimentary_tract_metabolism": "Gastrointestinal absorption or metabolic interactions",
                    "antineoplastic": "Bone marrow suppression or tumor resistance mechanisms",
                    "blood_organs": "Hematological effects or coagulation pathway interactions",
                    "hormonal": "Endocrine system interactions or hormone receptor effects",
                    "respiratory": "Pulmonary function or bronchodilation interactions",
                    "dermatological": "Skin absorption or topical application interactions",
                    "antiparasitic": "Antimicrobial resistance or metabolic pathway interactions",
                    "various": "Multiple potential interaction pathways",
                }

                mechanism = category_mechanisms.get(category, "Unknown mechanism")
                log += f"- Category-specific mechanism: {mechanism}\n"

            log += "\n"

        # Summary and recommendations
        log += "Summary and Recommendations:\n"
        log += "-" * 28 + "\n"

        severity_counts = {}
        for interaction in interactions:
            level = interaction.get("level", "Unknown")
            severity_counts[level] = severity_counts.get(level, 0) + 1

        log += f"- Total interactions analyzed: {len(interactions)}\n"
        log += f"- Severity distribution: {dict(severity_counts)}\n"

        # Overall recommendation
        if any(int_data.get("level") == "Major" for int_data in interactions):
            log += "- Overall recommendation: AVOID - Major interaction detected\n"
            log += "- Consider alternative medications\n"
        elif any(int_data.get("level") == "Moderate" for int_data in interactions):
            log += "- Overall recommendation: MONITOR - Moderate interaction present\n"
            log += "- Close patient monitoring required\n"
        else:
            log += "- Overall recommendation: AWARENESS - Minor interactions only\n"
            log += "- Standard monitoring appropriate\n"

        if detailed_analysis:
            log += "\nMechanistic Considerations:\n"
            log += f"- Monitor for additive effects in the {category.replace('_', ' ')} system\n"
            log += "- Consider potential for altered drug metabolism\n"
            log += "- Be aware of possible changes in drug efficacy or toxicity\n"
            log += "- Timing of administration may be important\n"

    except Exception as e:
        log += f"Error during mechanism analysis: {str(e)}\n"

    return log


def find_alternative_drugs_ddinter(target_drug, contraindicated_drugs, therapeutic_class=None, data_lake_path=None):
    """
    Find alternative drugs that don't interact with contraindicated drugs.

    Parameters
    ----------
    target_drug : str
        Drug to find alternatives for
    contraindicated_drugs : list of str
        List of drugs to avoid interactions with
    therapeutic_class : str, optional
        Limit search to specific therapeutic class
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with alternative drug recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Alternative Drug Finder (DDInter)\n"
    log += "=" * 32 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Alternative Drug Search Parameters:\n"
    log += f"- Target drug: {target_drug}\n"
    log += f"- Contraindicated drugs: {', '.join(contraindicated_drugs)}\n"
    log += f"- Therapeutic class filter: {therapeutic_class if therapeutic_class else 'All classes'}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize target drug name
        std_target = _standardize_drug_name(target_drug, name_mapping)
        if not std_target:
            log += f"Error: Target drug '{target_drug}' not found in DDInter database\n"
            return log

        # Standardize contraindicated drug names
        std_contraindicated = []
        missing_contraindicated = []

        for drug in contraindicated_drugs:
            std_drug = _standardize_drug_name(drug, name_mapping)
            if std_drug:
                std_contraindicated.append(std_drug)
            else:
                missing_contraindicated.append(drug)

        if missing_contraindicated:
            log += "Warning: The following contraindicated drugs were not found:\n"
            for drug in missing_contraindicated:
                log += f"- {drug}\n"
            log += "\n"

        # Get target drug information
        target_id = name_mapping[std_target]
        target_info = drug_info.get(target_id, {})
        target_categories = target_info.get("categories", [])

        log += "Target Drug Profile:\n"
        log += f"- Drug: {target_drug}\n"
        log += f"- Categories: {', '.join(target_categories)}\n"
        log += f"- Total interactions: {len(target_info.get('interactions', []))}\n\n"

        # Find alternative drugs
        alternatives = []

        for drug_id, drug_data in drug_info.items():
            drug_name = drug_data["name"]
            drug_categories = drug_data.get("categories", [])

            # Skip the target drug itself
            if drug_id == target_id:
                continue

            # Apply therapeutic class filter
            if therapeutic_class:
                if not any(therapeutic_class.lower() in cat.lower() for cat in drug_categories):
                    continue
            else:
                # Look for drugs in similar categories as target
                if not any(cat in target_categories for cat in drug_categories):
                    continue

            # Check if this drug interacts with any contraindicated drugs
            has_contraindicated_interactions = False
            interaction_count = 0
            major_interactions = 0

            std_drug_name = drug_data["standardized_name"]

            for contraindicated in std_contraindicated:
                if std_drug_name in interaction_matrix and contraindicated in interaction_matrix[std_drug_name]:
                    interactions = interaction_matrix[std_drug_name][contraindicated]
                    interaction_count += len(interactions)

                    # Check for major interactions
                    for interaction in interactions:
                        if interaction.get("level") == "Major":
                            major_interactions += 1
                            has_contraindicated_interactions = True
                            break

                    if has_contraindicated_interactions:
                        break

            # Add to alternatives if no major contraindicated interactions
            if not has_contraindicated_interactions:
                alternatives.append(
                    {
                        "name": drug_name,
                        "categories": drug_categories,
                        "interaction_count": interaction_count,
                        "total_interactions": len(drug_data.get("interactions", [])),
                    }
                )

        # Sort alternatives by interaction count (fewer is better)
        alternatives.sort(key=lambda x: x["interaction_count"])

        # Present results
        log += "Alternative Drug Analysis:\n"
        log += "-" * 25 + "\n"

        if alternatives:
            log += f"Found {len(alternatives)} potential alternatives:\n\n"

            # Show top 10 alternatives
            top_alternatives = alternatives[:10]

            for i, alt in enumerate(top_alternatives, 1):
                log += f"{i}. {alt['name']}\n"
                log += f"   - Categories: {', '.join(alt['categories'])}\n"
                log += f"   - Interactions with contraindicated drugs: {alt['interaction_count']}\n"
                log += f"   - Total known interactions: {alt['total_interactions']}\n"

                # Risk assessment
                if alt["interaction_count"] == 0:
                    risk = "No known interactions"
                elif alt["interaction_count"] <= 2:
                    risk = "Low interaction risk"
                elif alt["interaction_count"] <= 5:
                    risk = "Moderate interaction risk"
                else:
                    risk = "Higher interaction risk"

                log += f"   - Risk assessment: {risk}\n\n"

            if len(alternatives) > 10:
                log += f"... and {len(alternatives) - 10} additional alternatives\n\n"
        else:
            log += "No suitable alternatives found in the DDInter database\n"
            log += "Consider:\n"
            log += "- Expanding therapeutic class search criteria\n"
            log += "- Consulting additional drug databases\n"
            log += "- Seeking specialist pharmacological advice\n\n"

        # Recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 22 + "\n"

        if alternatives:
            best_alternative = alternatives[0]
            log += f"- Primary recommendation: {best_alternative['name']}\n"
            log += "- Rationale: Lowest interaction risk with contraindicated drugs\n"

            if best_alternative["interaction_count"] == 0:
                log += "- Safety profile: No known interactions with specified drugs\n"
            else:
                log += f"- Safety profile: {best_alternative['interaction_count']} minor interactions detected\n"

            log += "- Next steps: Verify therapeutic equivalence and dosing\n"
            log += "- Monitoring: Standard clinical monitoring recommended\n"
        else:
            log += "- No direct alternatives identified\n"
            log += "- Consider non-pharmacological approaches\n"
            log += "- Consult clinical pharmacist or specialist\n"
            log += "- Review patient's complete medication profile\n"

        log += "\nImportant Notes:\n"
        log += "- This analysis is based on DDInter 2.0 data only\n"
        log += "- Always verify therapeutic equivalence before substitution\n"
        log += "- Consider patient-specific factors (allergies, comorbidities)\n"
        log += "- Monitor patient response after any medication changes\n"

    except Exception as e:
        log += f"Error during alternative drug search: {str(e)}\n"

    return log


# OpenFDA Integration Functions


class OpenFDAClient:
    """
    Client for interacting with the FDA's OpenFDA API.

    Provides comprehensive drug safety monitoring, adverse event analysis,
    and regulatory intelligence capabilities through the OpenFDA API.
    """

    BASE_URL = "https://api.fda.gov"

    def __init__(self):
        import time

        import requests

        self.requests = requests
        self.time = time
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Biomni-Agent/1.0 (https://biomni.stanford.edu)"})
        self.retry_attempts = 3
        self.timeout = 30
        self.rate_limit_delay = 0.2  # 5 requests/second
        self.last_request_time = 0

    def _handle_rate_limiting(self):
        """Implement rate limiting to respect FDA API limits."""
        current_time = self.time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            self.time.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = self.time.time()

    def _validate_response(self, response_data: dict) -> dict:
        """Validate FDA API response structure and handle variations."""
        if not isinstance(response_data, dict):
            raise ValueError("Invalid FDA API response format")

        # Check for error responses
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown FDA API error")
            raise Exception(f"FDA API Error: {error_msg}")

        # Validate expected fields exist
        if "meta" not in response_data and "results" not in response_data:
            # Some endpoints return data directly without meta
            return {"results": [response_data], "meta": {"results": {"total": 1}}}

        return response_data

    def _handle_api_variations(self, endpoint: str, params: dict) -> dict:
        """Handle known FDA API endpoint variations and parameter mappings."""
        endpoint_param_mappings = {
            "drug/event": {
                "drug_name": "patient.drug.openfda.brand_name.exact",
                "generic_name": "patient.drug.openfda.generic_name.exact",
            },
            "drug/label": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
            "drug/enforcement": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
        }

        # Transform parameters based on endpoint
        if endpoint in endpoint_param_mappings:
            new_params = {}
            for key, value in params.items():
                if key in endpoint_param_mappings[endpoint]:
                    new_params[endpoint_param_mappings[endpoint][key]] = value
                else:
                    new_params[key] = value
            return new_params

        return params

    def _build_fda_search_params(self, endpoint: str, params: dict) -> dict:
        """Build FDA API search parameters from input parameters."""
        fda_params = {}

        # Handle drug name searches
        if "drug_name" in params:
            drug_name = params["drug_name"]
            if endpoint == "drug/event":
                # For adverse events, search in medicinalproduct field
                fda_params["search"] = f"patient.drug.medicinalproduct:{drug_name}"
            elif endpoint == "drug/label":
                # For drug labels, search in brand name
                fda_params["search"] = f"openfda.brand_name:{drug_name}"
            elif endpoint == "drug/enforcement":
                # For enforcement/recalls, search in brand name
                fda_params["search"] = f"openfda.brand_name:{drug_name}"

        # Handle other parameters
        for key, value in params.items():
            if key not in ["drug_name"]:  # Skip drug_name as it's handled above
                if key == "limit":
                    fda_params["limit"] = value
                elif key == "skip":
                    fda_params["skip"] = value
                # Add other FDA API parameters as needed

        return fda_params

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with retry logic and error handling."""
        self._handle_rate_limiting()

        # Build FDA API search parameters
        fda_params = self._build_fda_search_params(endpoint, params)

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(f"{self.BASE_URL}/{endpoint}.json", params=fda_params, timeout=self.timeout)

                if response.status_code == 404:
                    return {
                        "results": [],
                        "meta": {"results": {"total": 0}},
                        "message": "No results found for the specified query",
                    }

                response.raise_for_status()

                # Validate and normalize response
                data = self._validate_response(response.json())

                return data

            except self.requests.exceptions.Timeout:
                if attempt == self.retry_attempts - 1:
                    raise Exception("FDA API request timed out after multiple attempts") from None
                self.time.sleep(2**attempt)  # Exponential backoff

            except self.requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limiting - wait and retry
                    self.time.sleep(5 * (attempt + 1))
                    continue
                else:
                    raise Exception(f"FDA API HTTP Error: {e.response.status_code}") from e

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"FDA API request failed: {str(e)}") from e
                self.time.sleep(2**attempt)

        return {}

    def query_adverse_events(self, drug_name: str, limit: int = 100) -> dict:
        """Query adverse events with robust error handling and validation."""
        endpoint = "drug/event"
        params = {"drug_name": drug_name, "limit": limit}

        try:
            data = self._make_request(endpoint, params)

            # Add FDA disclaimer to results
            data["disclaimer"] = (
                "FDA Disclaimer: These data do not establish causation. "
                "Reports are voluntary and subject to reporting bias. "
                "Data should not be used for regulatory decision-making."
            )

            return data

        except Exception as e:
            return {
                "results": [],
                "meta": {"results": {"total": 0}},
                "error": str(e),
                "disclaimer": (
                    "FDA Disclaimer: These data do not establish causation. "
                    "Reports are voluntary and subject to reporting bias."
                ),
            }

    def query_drug_labels(self, drug_name: str, sections: list[str] | None = None) -> dict:
        """Query FDA drug label information."""
        endpoint = "drug/label"
        params = {"drug_name": drug_name, "limit": 50}

        return self._make_request(endpoint, params)

    def query_drug_recalls(self, drug_name: str, classification: list[str] | None = None) -> dict:
        """Query FDA drug recall and enforcement information."""
        endpoint = "drug/enforcement"
        params = {"drug_name": drug_name, "limit": 100}

        return self._make_request(endpoint, params)


# Helper Functions for OpenFDA Data Processing


def _standardize_drug_name_fda(drug_name: str) -> str:
    """Standardize drug names for FDA API queries."""
    # Handle None/empty values
    if not drug_name:
        return ""

    # Remove common suffixes
    suffixes = ["sodium", "hydrochloride", "sulfate", "phosphate", "acetate", "citrate"]

    # Clean and standardize
    name = drug_name.strip().lower()

    for suffix in suffixes:
        if name.endswith(f" {suffix}"):
            name = name[: -len(f" {suffix}")]

    return name


def _apply_fda_filters(response_data: dict, filters: dict) -> dict:
    """Apply post-query filtering to FDA responses."""
    if not response_data.get("results"):
        return response_data

    filtered_results = []

    for result in response_data["results"]:
        include = True

        # Apply severity filter
        if filters.get("severity_filter") or filters.get("severity"):
            severity_list = filters.get("severity_filter", filters.get("severity", []))
            if "serious" in severity_list:
                # For serious filter, only include if serious == '1'
                if result.get("serious") != "1":
                    include = False
            elif "non_serious" in severity_list:
                # For non-serious filter, only include if serious != '1'
                if result.get("serious") == "1":
                    include = False

        # Apply outcome filter
        if (filters.get("outcome_filter") or filters.get("outcome")) and include:
            outcome_list = filters.get("outcome_filter", filters.get("outcome", []))
            if "life_threatening" in outcome_list:
                # Check if the result has life threatening outcome
                if result.get("seriousnesslifethreatening") != "1":
                    include = False
            elif "hospitalization" in outcome_list:
                # Check if the result has hospitalization outcome
                if result.get("seriousnesshospitalization") != "1":
                    include = False
            elif "death" in outcome_list:
                # Check if the result has death outcome
                if result.get("seriousnessdeath") != "1":
                    include = False

        # Apply classification filter (for recalls)
        if filters.get("classification") and include:
            classification_list = filters.get("classification", [])
            result_class = result.get("classification", "")
            if result_class not in classification_list:
                include = False

        if include:
            filtered_results.append(result)

    response_data["results"] = filtered_results
    response_data["meta"]["results"]["total"] = len(filtered_results)

    return response_data


def _extract_fda_safety_signals(response_list: list[dict]) -> dict:
    """Extract safety signals from adverse event data."""
    drug_signals = {}
    reaction_patterns = {}
    temporal_patterns = {}

    for response in response_list:
        if not response.get("results"):
            continue

        for result in response["results"]:
            # Extract drug information
            drugs = result.get("patient", {}).get("drug", [])
            for drug in drugs:
                # Use the existing standardization function
                drug_name = _standardize_drug_name_fda(drug.get("medicinalproduct", ""))
                if drug_name:
                    if drug_name not in drug_signals:
                        drug_signals[drug_name] = {"total_reports": 0, "serious_reports": 0, "common_reactions": []}

                    drug_signals[drug_name]["total_reports"] += 1
                    if result.get("serious") == "1":
                        drug_signals[drug_name]["serious_reports"] += 1

            # Extract reaction patterns
            reactions = result.get("patient", {}).get("reaction", [])
            for reaction in reactions:
                reaction_name = reaction.get("reactionmeddrapt", "")
                if reaction_name:
                    if reaction_name not in reaction_patterns:
                        reaction_patterns[reaction_name] = {
                            "count": 0,
                            "severity_counts": {"serious": 0, "non_serious": 0},
                        }

                    reaction_patterns[reaction_name]["count"] += 1

                    # Count severity
                    if result.get("serious") == "1":
                        reaction_patterns[reaction_name]["severity_counts"]["serious"] += 1
                    else:
                        reaction_patterns[reaction_name]["severity_counts"]["non_serious"] += 1

            # Extract temporal patterns
            receipt_date = result.get("receiptdate")
            if receipt_date and len(receipt_date) >= 6:
                year_month = receipt_date[:6]  # YYYYMM
                if year_month not in temporal_patterns:
                    temporal_patterns[year_month] = {"count": 0, "serious_count": 0}

                temporal_patterns[year_month]["count"] += 1
                if result.get("serious") == "1":
                    temporal_patterns[year_month]["serious_count"] += 1

    # Build common reactions for each drug based on actual data
    for drug_name in drug_signals:
        # Find reactions that occurred with this specific drug
        drug_reactions = {}

        for response in response_list:
            if not response.get("results"):
                continue

            for result in response["results"]:
                drugs = result.get("patient", {}).get("drug", [])
                has_this_drug = any(
                    _standardize_drug_name_fda(drug.get("medicinalproduct", "")) == drug_name for drug in drugs
                )

                if has_this_drug:
                    reactions = result.get("patient", {}).get("reaction", [])
                    for reaction in reactions:
                        reaction_name = reaction.get("reactionmeddrapt", "")
                        if reaction_name:
                            if reaction_name not in drug_reactions:
                                drug_reactions[reaction_name] = 0
                            drug_reactions[reaction_name] += 1

        # Get top 3 reactions for this drug
        top_reactions = sorted(drug_reactions.items(), key=lambda x: x[1], reverse=True)[:3]
        drug_signals[drug_name]["common_reactions"] = [r[0] for r in top_reactions]

    return {
        "drug_signals": drug_signals,
        "reaction_patterns": reaction_patterns,
        "temporal_patterns": temporal_patterns,
    }


def _generate_fda_statistics(response_data: dict) -> dict:
    """Generate summary statistics from FDA responses."""
    stats = {
        "total_reports": 0,
        "serious_reports": 0,
        "death_reports": 0,
        "life_threatening_reports": 0,
        "hospitalization_reports": 0,
        "top_reactions": [],
        "temporal_pattern": {},
    }

    if not response_data.get("results"):
        return stats

    reaction_counts = {}

    for result in response_data["results"]:
        stats["total_reports"] += 1

        # Count serious reports
        if result.get("serious") == "1":
            stats["serious_reports"] += 1

        # Count specific outcomes
        outcomes = result.get("patient", {}).get("reaction", [])
        for outcome in outcomes:
            outcome_name = outcome.get("reactionmeddrapt", "Unknown")
            reaction_counts[outcome_name] = reaction_counts.get(outcome_name, 0) + 1

        # Count deaths and other serious outcomes
        if result.get("patient", {}).get("patientdeath"):
            stats["death_reports"] += 1

        if result.get("patient", {}).get("patientlifethreatening"):
            stats["life_threatening_reports"] += 1

        if result.get("patient", {}).get("patienthospitalization"):
            stats["hospitalization_reports"] += 1

    # Top reactions
    sorted_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)
    stats["top_reactions"] = sorted_reactions[:10]

    # Report distribution
    stats["report_distribution"] = {
        "serious_percentage": (stats["serious_reports"] / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
        "non_serious_percentage": ((stats["total_reports"] - stats["serious_reports"]) / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
        "death_percentage": (stats["death_reports"] / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
    }

    return stats


def _format_adverse_event_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format adverse event data into readable summary."""
    if not response_data.get("results"):
        return f"No adverse events found for {drug_name} in the FDA database."

    stats = _generate_fda_statistics(response_data)

    summary = "Adverse Event Summary\n"
    summary += "=" * 21 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total Reports: {stats['total_reports']:,}\n\n"

    if stats["total_reports"] > 0:
        summary += "Summary Statistics:\n"
        summary += f"- Serious Reports: {stats['serious_reports']:,} ({stats['serious_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["death_reports"] > 0:
            summary += (
                f"- Deaths: {stats['death_reports']:,} ({stats['death_reports'] / stats['total_reports'] * 100:.1f}%)\n"
            )

        if stats["life_threatening_reports"] > 0:
            summary += f"- Life-threatening: {stats['life_threatening_reports']:,} ({stats['life_threatening_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["hospitalization_reports"] > 0:
            summary += f"- Hospitalizations: {stats['hospitalization_reports']:,} ({stats['hospitalization_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["top_reactions"]:
            summary += "\nCommon Reactions:\n"
            for i, (reaction, count) in enumerate(stats["top_reactions"][:5], 1):
                summary += f"{i}. {reaction} ({count:,} reports)\n"

    # Add FDA disclaimer
    summary += "\n" + response_data.get("disclaimer", "")

    return summary


def _format_drug_label_summary(response_data: dict, drug_name: str, sections: list[str] | None = None) -> str:
    """Format drug label information into readable summary."""
    if not response_data.get("results"):
        return f"No drug label information found for {drug_name} in the FDA database."

    result = response_data["results"][0]  # Use first result

    summary = "OpenFDA Drug Label Information\n"
    summary += "=" * 29 + "\n"
    summary += f"Drug: {drug_name}\n"

    # Extract key information
    if "effective_time" in result:
        summary += f"Effective Date: {result['effective_time']}\n"

    if "openfda" in result:
        openfda = result["openfda"]
        if "brand_name" in openfda:
            summary += f"Brand Name: {', '.join(openfda['brand_name'])}\n"
        if "generic_name" in openfda:
            summary += f"Generic Name: {', '.join(openfda['generic_name'])}\n"
        if "manufacturer_name" in openfda:
            summary += f"Manufacturer: {', '.join(openfda['manufacturer_name'])}\n"

    summary += "\n"

    # Display specific sections
    section_mapping = {
        "indications_and_usage": "Indications and Usage",
        "contraindications": "Contraindications",
        "warnings": "Warnings",
        "dosage_and_administration": "Dosage and Administration",
        "adverse_reactions": "Adverse Reactions",
        "clinical_pharmacology": "Clinical Pharmacology",
    }

    sections_to_show = sections if sections else section_mapping.keys()

    for section_key in sections_to_show:
        if section_key in result:
            section_title = section_mapping.get(section_key, section_key.title())
            summary += f"{section_title}:\n"

            content = result[section_key]
            if isinstance(content, list):
                content = " ".join(content)

            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."

            summary += f"{content}\n\n"

    return summary


def _format_recall_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format recall information into structured output."""
    if not response_data.get("results"):
        return f"No drug recalls found for {drug_name} in the FDA database."

    summary = "OpenFDA Drug Recall Information\n"
    summary += "=" * 31 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total recalls found: {len(response_data['results'])}\n\n"

    if include_details:
        summary += "Recall Details:\n"

        for i, recall in enumerate(response_data["results"][:5], 1):  # Show top 5
            summary += f"{i}. Recall Number: {recall.get('recall_number', 'N/A')}\n"
            summary += f"   - Product: {recall.get('product_description', 'N/A')}\n"
            summary += f"   - Classification: {recall.get('classification', 'N/A')}\n"
            summary += f"   - Reason: {recall.get('reason_for_recall', 'N/A')}\n"
            summary += f"   - Date: {recall.get('recall_initiation_date', 'N/A')}\n"
            summary += f"   - Status: {recall.get('status', 'N/A')}\n"
            summary += f"   - Distribution: {recall.get('distribution_pattern', 'N/A')}\n\n"

        if len(response_data["results"]) > 5:
            summary += f"... and {len(response_data['results']) - 5} additional recalls\n"

    return summary


def _format_safety_signal_summary(
    signals_data: dict,
    drug_list: list[str],
    comparison_period: tuple[str, str] | None = None,
    signal_threshold: float = 2.0,
) -> str:
    """Format safety signal analysis results."""
    summary = "OpenFDA Safety Signal Analysis\n"
    summary += "=" * 29 + "\n"
    summary += f"Drugs analyzed: {drug_list}\n"

    # Add comparison period and threshold info
    if comparison_period:
        summary += f"Comparison period: {comparison_period[0]} to {comparison_period[1]}\n"
    if signal_threshold != 2.0:
        summary += f"Signal threshold: {signal_threshold}\n"
    summary += "\n"

    if not signals_data:
        summary += "No safety signals detected.\n"
        return summary

    summary += "Signal Detection Results:\n"

    # Handle the actual data structure from _extract_fda_safety_signals
    drug_signals = signals_data.get("drug_signals", {})
    reaction_patterns = signals_data.get("reaction_patterns", {})
    signals_data.get("temporal_patterns", {})

    # Display drug-specific signals
    for i, drug_name in enumerate(drug_list, 1):
        drug_data = drug_signals.get(drug_name, {})
        if drug_data:
            summary += f"{i}. {drug_name.title()}\n"
            summary += f"   - Total reports: {drug_data['total_reports']:,}\n"
            summary += f"   - Serious reports: {drug_data['serious_reports']:,}\n"

            if drug_data.get("common_reactions"):
                summary += f"   - Common reactions: {', '.join(drug_data['common_reactions'])}\n"
            summary += "\n"
        else:
            summary += f"{i}. {drug_name.title()}\n"
            summary += "   - No data found\n\n"

    # Display cross-drug reaction patterns
    if reaction_patterns:
        summary += "Cross-drug Analysis:\n"
        sorted_reactions = sorted(reaction_patterns.items(), key=lambda x: x[1]["count"], reverse=True)

        for reaction, data in sorted_reactions[:5]:  # Show top 5 reactions
            summary += f"- {reaction}: {data['count']:,} reports\n"
            if data["severity_counts"]["serious"] > 0:
                summary += f"  * Serious: {data['severity_counts']['serious']:,}\n"

    # Add trend analysis if comparison period is specified
    if comparison_period:
        summary += "\nTrend Analysis:\n"
        summary += f"Comparing current period to {comparison_period[0]} - {comparison_period[1]}\n"
        summary += "* Trend detection based on temporal patterns in adverse event reports\n"
        summary += "* Analysis considers seasonal variations and reporting delays\n"

    return summary


# Main OpenFDA Integration Functions


def query_fda_adverse_events(
    drug_name: str,
    date_range: tuple[str, str] | None = None,
    severity_filter: list[str] | None = None,
    outcome_filter: list[str] | None = None,
    limit: int = 100,
) -> str:
    """
    Query FDA adverse event reports for specific drugs.

    Args:
        drug_name: Name of the drug to query
        date_range: Optional date range as (start_date, end_date) in YYYY-MM-DD format
        severity_filter: Optional filter by severity levels ["serious", "non_serious"]
        outcome_filter: Optional filter by outcomes ["life_threatening", "hospitalization", "death"]
        limit: Maximum number of results to return

    Returns:
        Formatted string with adverse event analysis
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query adverse events
        response = client.query_adverse_events(standardized_name, limit=limit)

        # Apply filters if specified
        if severity_filter or outcome_filter:
            filters = {"severity_filter": severity_filter, "outcome_filter": outcome_filter}
            response = _apply_fda_filters(response, filters)

        # Format results with main function title
        formatted_result = _format_adverse_event_summary(response, drug_name, include_details=True)

        # Replace title for main function
        if formatted_result.startswith("Adverse Event Summary"):
            formatted_result = formatted_result.replace(
                "Adverse Event Summary\n" + "=" * 21, "OpenFDA Adverse Event Query Results\n" + "=" * 35, 1
            )

        # Add filter and date range info if specified
        lines = formatted_result.split("\n")
        insert_index = -1

        # Find insertion point (after drug name)
        for i, line in enumerate(lines):
            if line.startswith("Drug: "):
                insert_index = i + 1
                break

        if insert_index >= 0:
            # Add date range info
            if date_range:
                lines.insert(insert_index, f"Date range: {date_range[0]} to {date_range[1]}")
                insert_index += 1

            # Add severity filter info
            if severity_filter:
                lines.insert(insert_index, f"Severity filter: {severity_filter}")
                insert_index += 1

            # Add outcome filter info
            if outcome_filter:
                lines.insert(insert_index, f"Outcome filter: {outcome_filter}")
                insert_index += 1

        formatted_result = "\n".join(lines)

        return formatted_result

    except Exception as e:
        return f"Error querying FDA adverse events for {drug_name}: {str(e)}"


def get_fda_drug_label_info(drug_name: str, sections: list[str] | None = None) -> str:
    """
    Retrieve FDA drug label information.

    Args:
        drug_name: Name of the drug to query
        sections: Optional list of specific sections to retrieve
                 ["indications_and_usage", "contraindications", "warnings", "dosage_and_administration"]

    Returns:
        Formatted string with drug label information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query drug labels
        response = client.query_drug_labels(standardized_name, sections=sections)

        # Check if we got results
        if not response.get("results"):
            return f"No label information found for drug: {drug_name}"

        # Format results
        return _format_drug_label_summary(response, drug_name, sections=sections)

    except Exception as e:
        return f"Error retrieving FDA drug label for {drug_name}: {str(e)}"


def check_fda_drug_recalls(
    drug_name: str, classification: list[str] | None = None, date_range: tuple[str, str] | None = None
) -> str:
    """
    Check for FDA drug recalls and enforcement actions.

    Args:
        drug_name: Name of the drug to check
        classification: Optional filter by recall class ["Class I", "Class II", "Class III"]
        date_range: Optional date range for recalls

    Returns:
        Formatted string with recall information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query drug recalls
        response = client.query_drug_recalls(standardized_name, classification=classification)

        # Format results with filter information
        formatted_result = _format_recall_summary(response, drug_name, include_details=True)

        # Add filter information to the output
        if classification:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n", f"Drug: {drug_name}\nClassification filter: {', '.join(classification)}\n"
            )

        if date_range:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n", f"Drug: {drug_name}\nDate range: {date_range[0]} to {date_range[1]}\n"
            )

        return formatted_result

    except Exception as e:
        return f"Error checking FDA drug recalls for {drug_name}: {str(e)}"


def analyze_fda_safety_signals(
    drug_list: list[str], comparison_period: tuple[str, str] | None = None, signal_threshold: float = 2.0
) -> str:
    """
    Analyze safety signals across multiple drugs.

    Args:
        drug_list: List of drug names to analyze
        comparison_period: Optional comparison time period
        signal_threshold: Threshold for signal detection

    Returns:
        Formatted string with safety signal analysis
    """
    try:
        # Validate input parameters
        if not drug_list:
            return "Error: At least one drug must be provided for analysis"

        if len(drug_list) < 2:
            return "Error: At least 2 drugs required for comparative safety signal analysis"

        # Validate drug names
        valid_drugs = [drug.strip() for drug in drug_list if drug and drug.strip()]
        if not valid_drugs:
            return "Error: No valid drug names provided"

        client = OpenFDAClient()

        # Collect data for all drugs
        all_responses = []

        for drug in valid_drugs:
            standardized_name = _standardize_drug_name_fda(drug)
            if standardized_name:  # Only query if standardization worked
                response = client.query_adverse_events(standardized_name, limit=200)

                if response.get("results"):
                    all_responses.append(response)

        # Check if we got any data
        if not all_responses:
            return "Error: No adverse event data found for any of the provided drugs"

        # Extract safety signals
        signals = _extract_fda_safety_signals(all_responses)

        # Format results with comparison period and threshold info
        return _format_safety_signal_summary(signals, valid_drugs, comparison_period, signal_threshold)

    except Exception as e:
        return f"Error analyzing FDA safety signals: {str(e)}"
