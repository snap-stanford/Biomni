description = [
    {
        "description": "Perform Flux Balance Analysis (FBA) on a genome-scale "
        "metabolic network model and return a research log of the "
        "process and results.",
        "name": "perform_flux_balance_analysis",
        "optional_parameters": [
            {
                "default": None,
                "description": "Dictionary of reaction constraints "
                "where keys are reaction IDs and "
                "values are tuples of (lower_bound, "
                "upper_bound)",
                "name": "constraints",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Reaction ID to use as the objective function (e.g., biomass reaction)",
                "name": "objective_reaction",
                "type": "str",
            },
            {
                "default": "fba_results.csv",
                "description": "File name to save the flux distribution results",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the metabolic model file (SBML or JSON format)",
                "name": "model_file",
                "type": "str",
            }
        ],
    },
    {
        "description": "Model protein dimerization networks to find equilibrium concentrations of dimers.",
        "name": "model_protein_dimerization_network",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary mapping monomer names to their initial concentrations (in arbitrary units)",
                "name": "monomer_concentrations",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary mapping dimer names (as 'A-B' strings) to their association constants (Ka)",
                "name": "dimerization_affinities",
                "type": "dict",
            },
            {
                "default": None,
                "description": "List of (monomer1, monomer2) pairs that can form dimers",
                "name": "network_topology",
                "type": "list",
            },
        ],
    },
    {
        "description": "Construct and simulate kinetic models of metabolic networks "
        "and analyze their responses to perturbations.",
        "name": "simulate_metabolic_network_perturbation",
        "optional_parameters": [
            {"default": 100, "description": "Total simulation time", "name": "simulation_time", "type": "float"},
            {"default": 1000, "description": "Number of time points to simulate", "name": "time_points", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the COBRA model file (SBML format)",
                "name": "model_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Dictionary mapping metabolite IDs to their initial concentrations",
                "name": "initial_concentrations",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary with keys 'time' "
                "(float), 'metabolite' (str), and "
                "'factor' (float) for perturbation "
                "details",
                "name": "perturbation_params",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Simulate protein signaling network dynamics using ODE-based "
        "logic modeling with normalized Hill functions.",
        "name": "simulate_protein_signaling_network",
        "optional_parameters": [
            {
                "default": 100,
                "description": "Total simulation time in arbitrary time units.",
                "name": "simulation_time",
                "type": "float",
            },
            {
                "default": 1000,
                "description": "Number of time points for the simulation.",
                "name": "time_points",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary defining the network "
                "topology. Each key is a target "
                "protein and its value is a list of "
                "tuples (regulator, regulation_type) "
                "where regulation_type is 1 for "
                "activation and -1 for inhibition.",
                "name": "network_structure",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary of reaction parameters. "
                "Keys are tuples (regulator, target) "
                "and values are dictionaries with "
                "keys 'W' (weight), 'n' (Hill "
                "coefficient), and 'EC50' "
                "(half-maximal effective "
                "concentration).",
                "name": "reaction_params",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary of species parameters. "
                "Keys are protein names and values "
                "are dictionaries with keys 'tau' "
                "(time constant), 'y0' (initial "
                "concentration), and 'ymax' (maximum "
                "concentration).",
                "name": "species_params",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Compares two protein structures to identify structural differences and conformational changes.",
        "name": "compare_protein_structures",
        "optional_parameters": [
            {
                "default": "A",
                "description": "Chain ID to analyze in the first structure",
                "name": "chain_id1",
                "type": "str",
            },
            {
                "default": "A",
                "description": "Chain ID to analyze in the second structure",
                "name": "chain_id2",
                "type": "str",
            },
            {
                "default": "protein_comparison",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            },
        ],
        "required_parameters": [
            {"default": None, "description": "Path to the first PDB file", "name": "pdb_file1", "type": "str"},
            {"default": None, "description": "Path to the second PDB file", "name": "pdb_file2", "type": "str"},
        ],
    },
    {
        "description": "Simulate the time-dependent concentrations of renin-angiotensin system (RAS) components.",
        "name": "simulate_renin_angiotensin_system_dynamics",
        "optional_parameters": [
            {
                "default": 48,
                "description": "Total simulation time in hours",
                "name": "simulation_time",
                "type": "float",
            },
            {"default": 100, "description": "Number of time points to evaluate", "name": "time_points", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Initial concentrations of RAS "
                "components with keys: 'renin', "
                "'angiotensinogen', 'angiotensin_I', "
                "'angiotensin_II', "
                "'ACE2_angiotensin_II', "
                "'angiotensin_1_7'",
                "name": "initial_concentrations",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Kinetic rate constants with keys: "
                "'k_ren', 'k_agt', 'k_ace', "
                "'k_ace2', 'k_at1r', 'k_mas'",
                "name": "rate_constants",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Parameters controlling feedback mechanisms with keys: 'fb_ang_II', 'fb_ace2'",
                "name": "feedback_params",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Answer functions and properties questions for DNA sequences ",
        "name": "query_chatnt",
        "optional_parameters": [
            {
                "default": -1,
                "description": "Device to use for the ChatNT model. Default is -1 (CPU).",
                "name": "device",
                "type": "int",
            }
        ],
        "required_parameters": [
            {"default": "A", "description": "Questions about the DNA sequence", "name": "question", "type": "str"},
            {"default": "A", "description": "DNA sequence with potential functions", "name": "sequence", "type": "str"},
        ],
    },
    {
        "name": "predict_metabolic_engineering_targets",
        "description": (
            "Predict genetic overexpression (amplification), competing-pathway knockout/downregulation, "
            "and kinetically bottlenecked enzyme targets for metabolic engineering using Flux Scanning "
            "based on Enforced Objective Flux (FSEOF), parsimonious FBA (pFBA), Flux Variability Analysis "
            "(FVA), and optional sMOMENT/GECKO-style global enzyme mass-pool constraints."
        ),
        "required_parameters": [
            {
                "name": "model_file",
                "type": "str",
                "description": "Path to the genome-scale metabolic model file (.xml/.sbml, .json, or .mat).",
                "default": None,
            },
            {
                "name": "target_reaction",
                "type": "str",
                "description": "Reaction ID of the target product synthesis or secretion reaction.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "biomass_reaction",
                "type": "str",
                "description": "Biomass objective reaction ID. Inferred from model objective if None.",
                "default": None,
            },
            {
                "name": "n_steps",
                "type": "int",
                "description": "Number of enforced product flux steps scanned between WT and 90% max yield.",
                "default": 10,
            },
            {
                "name": "min_growth_fraction",
                "type": "float",
                "description": "Minimum fraction of WT growth required for viable knockout candidates.",
                "default": 0.2,
            },
            {
                "name": "fva_fraction",
                "type": "float",
                "description": "Fraction of optimal growth maintained during FVA robustness checks.",
                "default": 0.9,
            },
            {
                "name": "enzyme_kcat_mw",
                "type": "dict",
                "description": "Optional dict mapping reaction_id -> {'kcat_per_s': float, 'mw_kda': float}.",
                "default": None,
            },
            {
                "name": "total_enzyme_pool",
                "type": "float",
                "description": "Available metabolic enzyme mass budget in g_enzyme/gDW.",
                "default": None,
            },
            {
                "name": "output_file",
                "type": "str",
                "description": "Output CSV file path for the ranked metabolic engineering targets.",
                "default": "metabolic_engineering_targets.csv",
            },
        ],
    },
]
