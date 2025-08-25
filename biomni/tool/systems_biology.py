def query_chatnt(question, sequence, device=-1):
    """
    Call ChatNT to answer a question about a DNA sequence.

    Parameters:
    -----------
    question : str
        Question to ask about the DNA sequence
    sequence : str
        DNA sequence to analyze
    device : int, optional
        Device to use for the ChatNT model. Default is -1 (CPU).

    Returns:
    --------
    str
        Answer to the question
    """
    from transformers import pipeline

    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True, device=device)

    # Define custom inputs (note that the number of <DNA> token in the english sequence must be equal to len(dna_sequences))
    english_sequence = f"{question} <DNA> ?"
    dna_sequences = [sequence]

    # Generate sequence
    generated_english_sequence = pipe(inputs={"english_sequence": english_sequence, "dna_sequences": dna_sequences})

    return generated_english_sequence


def perform_flux_balance_analysis(model_file, constraints=None, objective_reaction=None, output_file="fba_results.csv"):
    """
    Perform Flux Balance Analysis (FBA) on a genome-scale metabolic network model.

    FBA is a computational technique that predicts metabolic flux distributions
    by formulating and solving a linear optimization problem.

    Parameters:
    -----------
    model_file : str
        Path to the metabolic model file (SBML or JSON format)
    constraints : dict, optional
        Dictionary of reaction constraints where keys are reaction IDs and
        values are tuples of (lower_bound, upper_bound)
    objective_reaction : str, optional
        Reaction ID to use as the objective function (e.g., biomass reaction)
        If None, uses the model's default objective
    output_file : str, optional
        File name to save the flux distribution results

    Returns:
    --------
    str
        Research log summarizing the FBA process and results
    """

    import cobra
    import pandas as pd

    # Initialize research log
    log = "# Flux Balance Analysis (FBA) Research Log\n\n"

    # Step 1: Load the metabolic network model
    log += "## Step 1: Loading metabolic model\n"
    try:
        if model_file.endswith(".xml") or model_file.endswith(".sbml"):
            model = cobra.io.read_sbml_model(model_file)
        elif model_file.endswith(".json"):
            model = cobra.io.load_json_model(model_file)
        else:
            model = cobra.io.load_model(model_file)
        log += f"- Successfully loaded model from {model_file}\n"
        log += f"- Model contains {len(model.reactions)} reactions and {len(model.metabolites)} metabolites\n\n"
    except Exception as e:
        log += f"- Error loading model: {str(e)}\n"
        return log

    # Step 2: Set constraints
    log += "## Step 2: Setting constraints\n"
    if constraints:
        log += "- Applied the following constraints:\n"
        for reaction_id, (lb, ub) in constraints.items():
            try:
                reaction = model.reactions.get_by_id(reaction_id)
                reaction.bounds = (lb, ub)
                log += f"  * {reaction_id}: lower_bound={lb}, upper_bound={ub}\n"
            except Exception as e:
                log += f"  * Error setting constraint for {reaction_id}: {str(e)}\n"
    else:
        log += "- No additional constraints specified, using model defaults\n"
    log += "\n"

    # Step 3: Set objective function
    log += "## Step 3: Setting objective function\n"
    if objective_reaction:
        try:
            model.objective = objective_reaction
            log += f"- Set objective function to maximize {objective_reaction}\n\n"
        except Exception as e:
            log += f"- Error setting objective function: {str(e)}\n"
            log += "- Using model's default objective function\n\n"
    else:
        log += f"- Using model's default objective function: {model.objective.expression}\n\n"

    # Step 4: Solve the FBA problem
    log += "## Step 4: Solving FBA optimization problem\n"
    try:
        solution = model.optimize()
        log += f"- Optimization status: {solution.status}\n"
        log += f"- Objective value: {solution.objective_value:.6f}\n\n"
    except Exception as e:
        log += f"- Error during optimization: {str(e)}\n"
        return log

    # Step 5: Save and report results
    log += "## Step 5: Analyzing flux distribution\n"

    # Create a dataframe with the flux distribution
    flux_distribution = pd.DataFrame(
        {
            "reaction_id": [r.id for r in model.reactions],
            "reaction_name": [r.name for r in model.reactions],
            "flux": [solution.fluxes[r.id] for r in model.reactions],
            "lower_bound": [r.lower_bound for r in model.reactions],
            "upper_bound": [r.upper_bound for r in model.reactions],
        }
    )

    # Save to file
    flux_distribution.to_csv(output_file, index=False)
    log += f"- Flux distribution saved to {output_file}\n"

    # Report active reactions (non-zero flux)
    active_reactions = flux_distribution[abs(flux_distribution["flux"]) > 1e-6]
    log += f"- Number of active reactions (flux > 1e-6): {len(active_reactions)}\n"

    # Report top reactions by absolute flux
    top_reactions = flux_distribution.iloc[abs(flux_distribution["flux"]).argsort()[::-1]].head(10)
    log += "- Top 10 reactions by absolute flux magnitude:\n"
    for _, row in top_reactions.iterrows():
        log += f"  * {row['reaction_id']} ({row['reaction_name']}): {row['flux']:.6f}\n"

    log += f"\nFBA analysis complete. Full results available in {output_file}\n"

    return log


def model_protein_dimerization_network(monomer_concentrations, dimerization_affinities, network_topology):
    """
    Model protein dimerization networks to find equilibrium concentrations of dimers.

    Parameters:
    -----------
    monomer_concentrations : dict
        Dictionary mapping monomer names to their initial concentrations (in arbitrary units)
    dimerization_affinities : dict
        Dictionary mapping dimer names (as 'A-B' strings) to their association constants (Ka)
    network_topology : list of tuples
        List of (monomer1, monomer2) pairs that can form dimers

    Returns:
    --------
    str
        Research log summarizing the modeling process and results
    """
    import time

    import numpy as np
    from scipy.integrate import solve_ivp

    # Start timing
    start_time = time.time()

    # Extract unique monomers and create mapping to indices
    all_monomers = list(monomer_concentrations.keys())
    monomer_to_idx = {monomer: i for i, monomer in enumerate(all_monomers)}

    # Create mapping from dimer indices to names
    dimer_pairs = []
    dimer_names = []
    dimer_affinities = []

    for m1, m2 in network_topology:
        dimer_name = f"{m1}-{m2}"
        if dimer_name in dimerization_affinities:
            affinity = dimerization_affinities[dimer_name]
        else:
            # Try reverse order
            dimer_name = f"{m2}-{m1}"
            if dimer_name in dimerization_affinities:
                affinity = dimerization_affinities[dimer_name]
            else:
                raise ValueError(f"No affinity found for dimer pair {m1}-{m2}")

        dimer_pairs.append((monomer_to_idx[m1], monomer_to_idx[m2]))
        dimer_names.append(dimer_name)
        dimer_affinities.append(affinity)

    # Initial conditions (monomer concentrations followed by dimer concentrations)
    num_monomers = len(all_monomers)
    num_dimers = len(dimer_pairs)
    y0 = np.zeros(num_monomers + num_dimers)

    for monomer, conc in monomer_concentrations.items():
        y0[monomer_to_idx[monomer]] = conc

    # Define ODE system
    def dimerization_odes(t, y):
        dydt = np.zeros_like(y)

        # Extract current concentrations
        monomer_concs = y[:num_monomers]
        dimer_concs = y[num_monomers:]

        # Calculate rate of change for each species
        for i, ((m1_idx, m2_idx), affinity) in enumerate(zip(dimer_pairs, dimer_affinities, strict=False)):
            # Formation rate: kon * [A] * [B]
            # Dissociation rate: koff * [AB]
            # At equilibrium: kon/koff = Ka (affinity)
            # For simplicity, set kon = Ka and koff = 1

            kon = affinity
            koff = 1.0

            # Dimer formation/dissociation
            dimer_idx = num_monomers + i

            # Rate of change
            formation_rate = kon * monomer_concs[m1_idx] * monomer_concs[m2_idx]
            dissociation_rate = koff * dimer_concs[i]
            net_rate = formation_rate - dissociation_rate

            # Update rates for monomers and dimers
            dydt[m1_idx] -= net_rate
            dydt[m2_idx] -= net_rate
            dydt[dimer_idx] += net_rate

        return dydt

    # Solve ODE system to equilibrium
    # Using a long enough time to reach equilibrium
    t_span = (0, 1000)  # Arbitrary long time to reach equilibrium

    sol = solve_ivp(
        dimerization_odes,
        t_span,
        y0,
        method="BDF",  # Good for stiff problems
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract final concentrations (equilibrium)
    final_monomer_concs = {monomer: sol.y[idx][-1] for monomer, idx in monomer_to_idx.items()}
    final_dimer_concs = {dimer_name: sol.y[num_monomers + i][-1] for i, dimer_name in enumerate(dimer_names)}

    # Calculate time taken
    elapsed_time = time.time() - start_time

    # Generate research log
    log = "Protein Dimerization Network Modeling\n"
    log += "=====================================\n\n"
    log += "Network summary:\n"
    log += f"- Number of monomers: {num_monomers}\n"
    log += f"- Number of possible dimers: {num_dimers}\n\n"

    log += "Initial conditions:\n"
    for monomer, conc in monomer_concentrations.items():
        log += f"- {monomer}: {conc:.4f}\n"
    log += "\n"

    log += "Dimerization affinities:\n"
    for dimer, affinity in zip(dimer_names, dimer_affinities, strict=False):
        log += f"- {dimer}: {affinity:.4e}\n"
    log += "\n"

    log += "Equilibrium concentrations:\n"
    log += "Monomers:\n"
    for monomer, conc in final_monomer_concs.items():
        log += f"- {monomer}: {conc:.4f}\n"

    log += "\nDimers:\n"
    for dimer, conc in final_dimer_concs.items():
        log += f"- {dimer}: {conc:.4f}\n"
    log += "\n"

    log += f"Simulation completed in {elapsed_time:.2f} seconds.\n"

    return log


def simulate_metabolic_network_perturbation(
    model_file, initial_concentrations, perturbation_params, simulation_time=100, time_points=1000
):
    """
    Construct and simulate kinetic models of metabolic networks and analyze their responses to perturbations.

    Parameters:
    -----------
    model_file : str
        Path to the COBRA model file (SBML format)
    initial_concentrations : dict
        Dictionary mapping metabolite IDs to their initial concentrations
    perturbation_params : dict
        Dictionary with the following keys:
        - 'time': float, time at which perturbation is applied
        - 'metabolite': str, ID of the metabolite to perturb
        - 'factor': float, multiplication factor for the metabolite concentration
    simulation_time : float, optional
        Total simulation time (default: 100)
    time_points : int, optional
        Number of time points to simulate (default: 1000)

    Returns:
    --------
    str
        Research log summarizing the steps taken and results obtained
    """

    import cobra
    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Step 1: Load the metabolic model
    try:
        model = cobra.io.read_sbml_model(model_file)
        log = f"Loaded metabolic model from {model_file} with {len(model.reactions)} reactions and {len(model.metabolites)} metabolites.\n\n"
    except Exception as e:
        return f"Error loading model: {str(e)}"

    # Step 2: Set up metabolite list and initial concentrations
    metabolites = list(model.metabolites)
    metabolite_ids = [m.id for m in metabolites]

    # Set default initial concentrations for metabolites not specified
    conc_array = np.ones(len(metabolites))
    for i, m_id in enumerate(metabolite_ids):
        if m_id in initial_concentrations:
            conc_array[i] = initial_concentrations[m_id]

    log += "Initial concentrations set up for all metabolites.\n\n"

    # Step 3: Define kinetic model using simple mass-action kinetics
    def mass_action_kinetics(reaction, concentrations, metabolite_ids):
        # Simple mass-action kinetics: product of substrate concentrations
        rate = 1.0
        for metabolite, coeff in reaction.metabolites.items():
            if coeff < 0:  # Substrate
                idx = metabolite_ids.index(metabolite.id)
                rate *= concentrations[idx] ** abs(coeff)
        return rate

    # Step 4: Define ODE system
    def ode_system(t, concentrations, metabolite_ids, reactions, perturbation):
        dC_dt = np.zeros(len(concentrations))

        # Apply perturbation if we're at or past the perturbation time
        if t >= perturbation["time"]:
            perturb_idx = metabolite_ids.index(perturbation["metabolite"])
            if not perturbation.get("applied", False):
                concentrations[perturb_idx] *= perturbation["factor"]
                perturbation["applied"] = True

        # Calculate reaction rates
        rates = {}
        for reaction in reactions:
            rates[reaction.id] = mass_action_kinetics(reaction, concentrations, metabolite_ids)

        # Update concentration changes based on stoichiometry and rates
        for i, metabolite_id in enumerate(metabolite_ids):
            for reaction in reactions:
                if metabolite_id in [m.id for m in reaction.metabolites]:
                    coeff = reaction.metabolites[model.metabolites.get_by_id(metabolite_id)]
                    dC_dt[i] += coeff * rates[reaction.id]

        return dC_dt

    log += "Kinetic model defined using mass-action kinetics for all reactions.\n\n"

    # Step 5: Simulate the system
    perturbation = perturbation_params.copy()
    perturbation["applied"] = False

    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    log += f"Starting simulation for {simulation_time} time units with perturbation of {perturbation['metabolite']} "
    log += f"by factor {perturbation['factor']} at time {perturbation['time']}.\n\n"

    try:
        solution = solve_ivp(
            lambda t, y: ode_system(t, y, metabolite_ids, model.reactions, perturbation),
            t_span,
            conc_array,
            t_eval=t_eval,
            method="LSODA",
        )

        log += f"Simulation completed successfully with {len(solution.t)} time points.\n\n"
    except Exception as e:
        return f"Error during simulation: {str(e)}"

    # Step 6: Calculate fluxes at each time point
    fluxes = np.zeros((len(solution.t), len(model.reactions)))
    reaction_ids = [r.id for r in model.reactions]

    for t_idx, t in enumerate(solution.t):
        concentrations = solution.y[:, t_idx]
        perturbation_temp = perturbation.copy()
        if t >= perturbation["time"]:
            perturbation_temp["applied"] = True

        for r_idx, reaction in enumerate(model.reactions):
            fluxes[t_idx, r_idx] = mass_action_kinetics(reaction, concentrations, metabolite_ids)

    log += "Calculated reaction fluxes for all time points.\n\n"

    # Step 7: Save results to files
    conc_df = pd.DataFrame(solution.y.T, columns=metabolite_ids)
    conc_df["time"] = solution.t
    conc_file = "metabolite_concentrations.csv"
    conc_df.to_csv(conc_file, index=False)

    flux_df = pd.DataFrame(fluxes, columns=reaction_ids)
    flux_df["time"] = solution.t
    flux_file = "reaction_fluxes.csv"
    flux_df.to_csv(flux_file, index=False)

    log += f"Results saved to {conc_file} and {flux_file}.\n\n"

    # Step 8: Analyze perturbation response
    perturb_time_idx = np.where(solution.t >= perturbation["time"])[0][0]
    pre_perturb = solution.y[:, perturb_time_idx - 1]
    post_perturb = solution.y[:, perturb_time_idx]

    # Find metabolites with significant changes
    significant_changes = []
    for i, m_id in enumerate(metabolite_ids):
        rel_change = abs(post_perturb[i] - pre_perturb[i]) / pre_perturb[i] if pre_perturb[i] > 0 else 0
        if rel_change > 0.05:  # 5% change threshold
            significant_changes.append((m_id, rel_change))

    log += "Perturbation Analysis Results:\n"
    log += f"Perturbation of {perturbation['metabolite']} at time {perturbation['time']} by factor {perturbation['factor']}.\n"
    log += f"Number of metabolites with significant immediate response: {len(significant_changes)}.\n"

    if significant_changes:
        log += "Top 5 most affected metabolites (by relative concentration change):\n"
        for m_id, change in sorted(significant_changes, key=lambda x: x[1], reverse=True)[:5]:
            log += f"  - {m_id}: {change * 100:.2f}% change\n"

    log += "\nSimulation and perturbation analysis completed successfully."

    return log


def simulate_protein_signaling_network(
    network_structure, reaction_params, species_params, simulation_time=100, time_points=1000
):
    """
    Simulate protein signaling network dynamics using ODE-based logic modeling with normalized Hill functions.

    Parameters:
    -----------
    network_structure : dict
        Dictionary defining the network topology. Each key is a target protein and its value is a list of tuples
        (regulator, regulation_type) where regulation_type is 1 for activation and -1 for inhibition.

    reaction_params : dict
        Dictionary of reaction parameters. Keys are tuples (regulator, target) and values are dictionaries
        with keys 'W' (weight), 'n' (Hill coefficient), and 'EC50' (half-maximal effective concentration).

    species_params : dict
        Dictionary of species parameters. Keys are protein names and values are dictionaries
        with keys 'tau' (time constant), 'y0' (initial concentration), and 'ymax' (maximum concentration).

    simulation_time : float, optional
        Total simulation time in arbitrary time units. Default is 100.

    time_points : int, optional
        Number of time points for the simulation. Default is 1000.

    Returns:
    --------
    str
        Research log summarizing the simulation process and results.
    """
    import csv

    import numpy as np
    from scipy.integrate import solve_ivp

    # Extract all unique protein species
    all_proteins = set(network_structure.keys())
    for _target, regulators in network_structure.items():
        for regulator, _ in regulators:
            all_proteins.add(regulator)

    # Create ordered list of proteins for indexing
    proteins = sorted(all_proteins)
    protein_indices = {protein: i for i, protein in enumerate(proteins)}

    # Define the normalized Hill function
    def hill_function(x, n, ec50):
        return x**n / (x**n + ec50**n)

    # Define the ODE system
    def ode_system(t, y):
        dydt = np.zeros_like(y)

        for target, regulators in network_structure.items():
            target_idx = protein_indices[target]
            target_params = species_params[target]

            # Calculate regulation term for each regulator
            regulation_terms = []
            for regulator, reg_type in regulators:
                regulator_idx = protein_indices[regulator]
                params = reaction_params.get((regulator, target), {})

                if not params:
                    continue

                x = y[regulator_idx]
                n = params["n"]
                ec50 = params["EC50"]
                weight = params["W"]

                # Apply Hill function based on regulation type
                if reg_type == 1:  # Activation
                    term = weight * hill_function(x, n, ec50)
                else:  # Inhibition
                    term = weight * (1 - hill_function(x, n, ec50))

                regulation_terms.append(term)

            # Combine regulation terms (if any)
            if regulation_terms:
                # Simple summation of regulation terms
                regulation = sum(regulation_terms) / len(regulation_terms)

                # Ensure regulation stays within [0, 1]
                regulation = max(0, min(1, regulation))

                # Calculate rate of change
                tau = target_params["tau"]
                ymax = target_params["ymax"]
                dydt[target_idx] = (1 / tau) * (regulation * ymax - y[target_idx])

        return dydt

    # Set up initial conditions
    y0 = np.zeros(len(proteins))
    for protein, params in species_params.items():
        if protein in protein_indices:
            y0[protein_indices[protein]] = params["y0"]

    # Set up time points
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    # Solve the ODE system
    solution = solve_ivp(ode_system, t_span, y0, method="LSODA", t_eval=t_eval, rtol=1e-6, atol=1e-9)

    # Save results to CSV
    output_file = "protein_signaling_simulation_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        header = ["Time"] + proteins
        writer.writerow(header)

        # Write data
        for i in range(len(solution.t)):
            row = [solution.t[i]] + list(solution.y[:, i])
            writer.writerow(row)

    # Generate research log
    log = "ODE-based Logic Modeling of Protein Signaling Networks\n"
    log += "=============================================\n\n"
    log += f"Network structure: {len(network_structure)} target proteins, {len(proteins)} total proteins\n"
    log += f"Simulation time: {simulation_time} time units\n"
    log += f"Number of time points: {time_points}\n\n"

    log += "Simulation steps:\n"
    log += "1. Initialized protein concentrations based on provided initial values\n"
    log += "2. Set up ODE system using normalized Hill functions for protein interactions\n"
    log += "3. Solved the ODE system using LSODA integration method\n"
    log += "4. Saved time-series data for all protein concentrations\n\n"

    log += f"Results saved to: {output_file}\n\n"

    # Add summary statistics
    log += "Summary of final protein concentrations:\n"
    for i, protein in enumerate(proteins):
        final_conc = solution.y[i, -1]
        log += f"- {protein}: {final_conc:.4f}\n"

    return log


def compare_protein_structures(pdb_file1, pdb_file2, chain_id1="A", chain_id2="A", output_prefix="protein_comparison"):
    """
    Compares two protein structures to identify structural differences and conformational changes.

    Parameters:
    -----------
    pdb_file1 : str
        Path to the first PDB file
    pdb_file2 : str
        Path to the second PDB file
    chain_id1 : str, optional
        Chain ID to analyze in the first structure (default: 'A')
    chain_id2 : str, optional
        Chain ID to analyze in the second structure (default: 'A')
    output_prefix : str, optional
        Prefix for output files (default: "protein_comparison")

    Returns:
    --------
    str
        A research log summarizing the structural comparison analysis
    """
    import warnings

    import numpy as np
    from Bio.PDB import PDBIO, PDBParser, Select, Superimposer
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    # Suppress PDB parsing warnings
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)

    research_log = []
    research_log.append("# Protein Structure Comparison Analysis\n")
    research_log.append(f"Comparing structures from {pdb_file1} and {pdb_file2}\n")

    # Initialize parser
    parser = PDBParser()

    # Parse structures
    research_log.append("## Loading PDB structures")
    structure1 = parser.get_structure("structure1", pdb_file1)
    structure2 = parser.get_structure("structure2", pdb_file2)

    # Get specified chains
    try:
        chain1 = structure1[0][chain_id1]
        chain2 = structure2[0][chain_id2]
        research_log.append(f"Successfully loaded chain {chain_id1} from {pdb_file1}")
        research_log.append(f"Successfully loaded chain {chain_id2} from {pdb_file2}")
    except KeyError as e:
        return f"Error: Chain not found: {str(e)}"

    # Get CA atoms for alignment
    ca_atoms1 = []
    ca_atoms2 = []

    # Create residue mappings based on residue ID
    residues1 = {residue.id[1]: residue for residue in chain1 if residue.has_id("CA")}
    residues2 = {residue.id[1]: residue for residue in chain2 if residue.has_id("CA")}

    # Find common residues
    common_residues = sorted(set(residues1.keys()) & set(residues2.keys()))

    if len(common_residues) == 0:
        return "Error: No common residues with CA atoms found between the structures"

    research_log.append(f"Found {len(common_residues)} common residues for alignment\n")

    # Get paired atoms for alignment
    for res_id in common_residues:
        ca_atoms1.append(residues1[res_id]["CA"])
        ca_atoms2.append(residues2[res_id]["CA"])

    # Perform structural alignment
    research_log.append("## Structural Alignment")
    super_imposer = Superimposer()
    super_imposer.set_atoms(ca_atoms1, ca_atoms2)
    super_imposer.apply(structure2.get_atoms())

    # Calculate RMSD
    rmsd = super_imposer.rms
    research_log.append(f"Overall RMSD: {rmsd:.4f} Å\n")

    # Calculate per-residue distance after alignment
    research_log.append("## Conformational Differences Analysis")

    # Find regions with significant differences
    distance_data = []
    significant_changes = []
    threshold = 2.0  # Angstroms threshold for significant difference

    for res_id in common_residues:
        res1 = residues1[res_id]
        res2 = residues2[res_id]

        # Calculate distance between CA atoms
        ca1 = res1["CA"]
        ca2 = res2["CA"]
        distance = np.linalg.norm(ca1.coord - ca2.coord)

        distance_data.append((res_id, distance))

        if distance > threshold:
            significant_changes.append((res_id, res1.get_resname(), distance))

    # Save alignment as PDB files
    aligned_file1 = f"{output_prefix}_ref.pdb"
    aligned_file2 = f"{output_prefix}_aligned.pdb"

    io = PDBIO()
    io.set_structure(structure1)
    io.save(aligned_file1, select=Select())

    io.set_structure(structure2)
    io.save(aligned_file2, select=Select())

    research_log.append(f"Aligned structures saved as {aligned_file1} and {aligned_file2}")

    # Report on significant differences
    if significant_changes:
        research_log.append(
            f"\nIdentified {len(significant_changes)} residues with significant conformational changes:"
        )
        for res_id, res_name, distance in significant_changes:
            research_log.append(f"  - Residue {res_name}{res_id}: {distance:.2f} Å displacement")
    else:
        research_log.append("\nNo significant conformational changes detected (threshold: 2.0 Å)")

    # Save per-residue distance data
    distance_file = f"{output_prefix}_residue_distances.csv"
    with open(distance_file, "w") as f:
        f.write("Residue_ID,Distance(Å)\n")
        for res_id, distance in distance_data:
            f.write(f"{res_id},{distance:.4f}\n")

    research_log.append(f"\nPer-residue distance data saved to {distance_file}")

    # Identify regions with continuous changes
    regions = []
    current_region = []

    for res_id, _, distance in significant_changes:
        if not current_region or res_id == current_region[-1][0] + 1:
            current_region.append((res_id, distance))
        else:
            if len(current_region) >= 3:  # Consider regions with at least 3 consecutive residues
                regions.append(current_region)
            current_region = [(res_id, distance)]

    if current_region and len(current_region) >= 3:
        regions.append(current_region)

    if regions:
        research_log.append("\n## Continuous Regions with Conformational Changes")
        for i, region in enumerate(regions, 1):
            start_res = region[0][0]
            end_res = region[-1][0]
            avg_dist = sum(r[1] for r in region) / len(region)
            research_log.append(f"Region {i}: Residues {start_res}-{end_res} (Average displacement: {avg_dist:.2f} Å)")

    research_log.append("\n## Summary")
    research_log.append(f"- Compared structures from {pdb_file1} and {pdb_file2}")
    research_log.append(f"- Overall RMSD: {rmsd:.4f} Å")
    research_log.append(f"- {len(significant_changes)} residues with significant conformational changes")
    research_log.append(f"- {len(regions)} continuous regions of conformational change")
    research_log.append(f"- Files generated: {aligned_file1}, {aligned_file2}, {distance_file}")

    return "\n".join(research_log)


def simulate_renin_angiotensin_system_dynamics(
    initial_concentrations, rate_constants, feedback_params, simulation_time=48, time_points=100
):
    """
    Simulate the time-dependent concentrations of renin-angiotensin system (RAS) components.

    Parameters:
    -----------
    initial_concentrations : dict
        Initial concentrations of RAS components with keys:
        'renin', 'angiotensinogen', 'angiotensin_I', 'angiotensin_II',
        'ACE2_angiotensin_II', 'angiotensin_1_7'

    rate_constants : dict
        Kinetic rate constants with keys:
        'k_ren' (renin production), 'k_agt' (angiotensinogen production),
        'k_ace' (ACE conversion rate), 'k_ace2' (ACE2 conversion rate),
        'k_at1r' (AT1R binding rate), 'k_mas' (Mas receptor binding rate)

    feedback_params : dict
        Parameters controlling feedback mechanisms with keys:
        'fb_ang_II' (angiotensin II feedback), 'fb_ace2' (ACE2 feedback)

    simulation_time : float, optional
        Total simulation time in hours (default: 48)

    time_points : int, optional
        Number of time points to evaluate (default: 100)

    Returns:
    --------
    str
        Research log summarizing the simulation steps and results
    """

    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp

    # Extract initial concentrations
    y0 = [
        initial_concentrations["renin"],
        initial_concentrations["angiotensinogen"],
        initial_concentrations["angiotensin_I"],
        initial_concentrations["angiotensin_II"],
        initial_concentrations["ACE2_angiotensin_II"],
        initial_concentrations["angiotensin_1_7"],
    ]

    # Define the system of ODEs
    def ras_ode_system(t, y):
        renin, agt, ang_I, ang_II, ace2_ang_II, ang_1_7 = y

        # Production rates with feedback
        renin_production = rate_constants["k_ren"] * (1 / (1 + feedback_params["fb_ang_II"] * ang_II))
        agt_production = rate_constants["k_agt"]

        # Conversion rates
        ang_I_formation = renin * agt
        ang_II_formation = rate_constants["k_ace"] * ang_I
        ace2_binding = rate_constants["k_ace2"] * ang_II
        ang_1_7_formation = ace2_ang_II

        # Clearance/degradation (simplified as proportional to concentration)
        renin_clearance = 0.1 * renin
        agt_clearance = 0.05 * agt
        ang_I_clearance = 0.2 * ang_I
        ang_II_clearance = 0.3 * ang_II + rate_constants["k_at1r"] * ang_II
        ace2_ang_II_clearance = 0.15 * ace2_ang_II
        ang_1_7_clearance = 0.25 * ang_1_7 + rate_constants["k_mas"] * ang_1_7

        # ODEs
        drenin_dt = renin_production - renin_clearance
        dagt_dt = agt_production - agt_clearance - ang_I_formation
        dang_I_dt = ang_I_formation - ang_I_clearance - ang_II_formation
        dang_II_dt = ang_II_formation - ang_II_clearance - ace2_binding
        dace2_ang_II_dt = ace2_binding - ace2_ang_II_clearance - ang_1_7_formation
        dang_1_7_dt = ang_1_7_formation - ang_1_7_clearance

        return [drenin_dt, dagt_dt, dang_I_dt, dang_II_dt, dace2_ang_II_dt, dang_1_7_dt]

    # Time points for simulation
    t_span = (0, simulation_time)
    t_eval = np.linspace(0, simulation_time, time_points)

    # Solve the ODE system
    solution = solve_ivp(ras_ode_system, t_span, y0, method="RK45", t_eval=t_eval, rtol=1e-6)

    # Create DataFrame with results
    component_names = [
        "Renin",
        "Angiotensinogen",
        "Angiotensin I",
        "Angiotensin II",
        "ACE2-Angiotensin II",
        "Angiotensin 1-7",
    ]
    results_df = pd.DataFrame(solution.y.T, columns=component_names)
    results_df.insert(0, "Time (hours)", solution.t)

    # Save results to CSV
    results_file = "ras_simulation_results.csv"
    results_df.to_csv(results_file, index=False)

    # Create research log
    log = f"""RAS ODE Modeling Simulation Log:

1. Initialized RAS component concentrations:
   - Renin: {initial_concentrations["renin"]}
   - Angiotensinogen: {initial_concentrations["angiotensinogen"]}
   - Angiotensin I: {initial_concentrations["angiotensin_I"]}
   - Angiotensin II: {initial_concentrations["angiotensin_II"]}
   - ACE2-Angiotensin II complex: {initial_concentrations["ACE2_angiotensin_II"]}
   - Angiotensin 1-7: {initial_concentrations["angiotensin_1_7"]}

2. Applied rate constants:
   - Renin production (k_ren): {rate_constants["k_ren"]}
   - Angiotensinogen production (k_agt): {rate_constants["k_agt"]}
   - ACE conversion rate (k_ace): {rate_constants["k_ace"]}
   - ACE2 conversion rate (k_ace2): {rate_constants["k_ace2"]}
   - AT1R binding rate (k_at1r): {rate_constants["k_at1r"]}
   - Mas receptor binding rate (k_mas): {rate_constants["k_mas"]}

3. Feedback parameters:
   - Angiotensin II feedback (fb_ang_II): {feedback_params["fb_ang_II"]}
   - ACE2 feedback (fb_ace2): {feedback_params["fb_ace2"]}

4. Simulation parameters:
   - Total simulation time: {simulation_time} hours
   - Number of time points: {time_points}

5. Solved system of ODEs using SciPy's solve_ivp with RK45 method.

6. Final concentrations at {simulation_time} hours:
   - Renin: {results_df["Renin"].iloc[-1]:.4f}
   - Angiotensinogen: {results_df["Angiotensinogen"].iloc[-1]:.4f}
   - Angiotensin I: {results_df["Angiotensin I"].iloc[-1]:.4f}
   - Angiotensin II: {results_df["Angiotensin II"].iloc[-1]:.4f}
   - ACE2-Angiotensin II complex: {results_df["ACE2-Angiotensin II"].iloc[-1]:.4f}
   - Angiotensin 1-7: {results_df["Angiotensin 1-7"].iloc[-1]:.4f}

7. Results saved to file: {results_file}

8. Key observations:
   - The model captures the conversion of Angiotensinogen to Angiotensin I via Renin
   - Angiotensin I is converted to Angiotensin II via ACE
   - Angiotensin II binds with ACE2 to form the ACE2-Angiotensin II complex
   - The complex is converted to Angiotensin 1-7
   - Feedback mechanisms regulate the production rates
"""

    return log
