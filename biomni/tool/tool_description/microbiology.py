description = [
    {
        "description": "Optimize anaerobic digestion process conditions to maximize VFA production or methane yield.",
        "name": "optimize_anaerobic_digestion_process",
        "optional_parameters": [
            {
                "default": "methane_yield",
                "description": "Target output to maximize, either 'vfa_production' or 'methane_yield'",
                "name": "target_output",
                "type": "str",
            },
            {
                "default": "rsm",
                "description": "Method used for optimization, "
                "either 'rsm' (Response Surface "
                "Methodology) or 'genetic' (Genetic "
                "Algorithm)",
                "name": "optimization_method",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary containing waste "
                "characteristics such as "
                "total_solids, volatile_solids, and "
                "cod",
                "name": "waste_characteristics",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing operational "
                "parameters and their ranges for "
                "hrt, olr, if_ratio, temperature, "
                "and ph",
                "name": "operational_parameters",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Analyzes arsenic speciation in liquid samples using "
        "HPLC-ICP-MS technique. Returns a research log summarizing "
        "analysis steps and results.",
        "name": "analyze_arsenic_speciation_hplc_icpms",
        "optional_parameters": [
            {
                "default": "Unknown Sample",
                "description": "Name of the sample being analyzed",
                "name": "sample_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Dictionary containing calibration "
                "standards data with known "
                "concentrations for each arsenic "
                "species",
                "name": "calibration_data",
                "type": "dict",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary containing sample data "
                "with keys as sample IDs and values "
                "as dictionaries with retention "
                "times (in minutes) as keys and "
                "signal intensities as values",
                "name": "sample_data",
                "type": "dict",
            }
        ],
    },
    {
        "description": "Count bacterial colonies from an image of agar plate using computer vision techniques.",
        "name": "count_bacterial_colonies",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Dilution factor of the plated sample",
                "name": "dilution_factor",
                "type": "float",
            },
            {
                "default": 65.0,
                "description": "Area of the agar plate in square centimeters",
                "name": "plate_area_cm2",
                "type": "float",
            },
            {
                "default": "./output",
                "description": "Directory to save output images and results",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the image file containing bacterial colonies on agar plate",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Annotate a bacterial genome using Prokka to identify genes, proteins, and functional features.",
        "name": "annotate_bacterial_genome",
        "optional_parameters": [
            {
                "default": "annotation_results",
                "description": "Directory where annotation results will be saved",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": "",
                "description": "Genus name for the organism",
                "name": "genus",
                "type": "str",
            },
            {
                "default": "",
                "description": "Species name for the organism",
                "name": "species",
                "type": "str",
            },
            {
                "default": "",
                "description": "Strain identifier",
                "name": "strain",
                "type": "str",
            },
            {
                "default": "",
                "description": "Prefix for output files",
                "name": "prefix",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the assembled genome sequence file in FASTA format",
                "name": "genome_file_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Quantify bacterial concentration (CFU/mL) using serial dilutions and spot plating.",
        "name": "enumerate_bacterial_cfu_by_serial_dilution",
        "optional_parameters": [
            {
                "default": 1.0,
                "description": "Volume of the initial bacterial sample in milliliters",
                "name": "initial_sample_volume_ml",
                "type": "float",
            },
            {
                "default": 100000000.0,
                "description": "Estimated concentration of bacteria in the initial sample (CFU/mL)",
                "name": "estimated_concentration",
                "type": "float",
            },
            {
                "default": 10,
                "description": "Factor by which each dilution reduces the concentration",
                "name": "dilution_factor",
                "type": "int",
            },
            {
                "default": 8,
                "description": "Number of serial dilutions to perform",
                "name": "num_dilutions",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Number of replicate spots to plate for each dilution",
                "name": "spots_per_dilution",
                "type": "int",
            },
            {
                "default": "cfu_enumeration_results.csv",
                "description": "Filename to save the CFU enumeration results",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Model bacterial population dynamics over time using ordinary differential equations.",
        "name": "model_bacterial_growth_dynamics",
        "optional_parameters": [
            {
                "default": 24,
                "description": "Total simulation time in hours",
                "name": "simulation_time",
                "type": "float",
            },
            {
                "default": 0.1,
                "description": "Time step for simulation output",
                "name": "time_step",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Initial bacterial population size (CFU/ml or cells)",
                "name": "initial_population",
                "type": "float",
            },
            {
                "default": None,
                "description": "Bacterial growth rate (per hour)",
                "name": "growth_rate",
                "type": "float",
            },
            {
                "default": None,
                "description": "Rate at which bacteria are cleared from the system (per hour)",
                "name": "clearance_rate",
                "type": "float",
            },
            {
                "default": None,
                "description": "Maximum carrying capacity of the environment (CFU/ml or cells)",
                "name": "niche_size",
                "type": "float",
            },
        ],
    },
    {
        "description": "Quantifies biofilm biomass using crystal violet staining "
        "assay data and returns a detailed research log.",
        "name": "quantify_biofilm_biomass_crystal_violet",
        "optional_parameters": [
            {
                "default": None,
                "description": "Names of the biofilm samples corresponding to od_values",
                "name": "sample_names",
                "type": "array",
                "items": {"type": "string"},
            },
            {
                "default": 0,
                "description": "Index of the negative control sample in od_values",
                "name": "control_index",
                "type": "int",
            },
            {
                "default": None,
                "description": "Path to save the results as CSV file",
                "name": "save_path",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Optical density measurements from "
                "crystal violet staining "
                "representing absorbance readings "
                "for samples",
                "name": "od_values",
                "type": "array",
                "items": {"type": "number"},
            }
        ],
    },
    {
        "description": "Perform automated cell segmentation and quantify "
        "morphological metrics from fluorescence microscopy images.",
        "name": "segment_and_analyze_microbial_cells",
        "optional_parameters": [
            {
                "default": "./output",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 50,
                "description": "Minimum cell size in pixels to filter noise",
                "name": "min_cell_size",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the fluorescence microscopy image file",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Perform cell segmentation on fluorescence microscopy images "
        "using deep learning with pre-trained models from the "
        "Cellpose/Omnipose library.",
        "name": "segment_cells_with_deep_learning",
        "optional_parameters": [
            {
                "default": "bact_fluor_omni",
                "description": "Name of the pre-trained model to "
                "use (Options include: "
                "'bact_fluor_omni', 'cyto', "
                "'nuclei', etc.)",
                "name": "model_type",
                "type": "str",
            },
            {
                "default": None,
                "description": "Expected diameter of cells in pixels. If None, diameter is automatically estimated",
                "name": "diameter",
                "type": "float",
            },
            {
                "default": "segmentation_results",
                "description": "Directory to save segmentation results",
                "name": "save_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the fluorescence microscopy image file",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Simulate microbial community dynamics using the Generalized Lotka-Volterra (gLV) model.",
        "name": "simulate_generalized_lotka_volterra_dynamics",
        "optional_parameters": [
            {
                "default": "glv_simulation_results.csv",
                "description": "Filename to save the simulation results",
                "name": "output_file",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Initial abundances of each microbial species (1D array)",
                "name": "initial_abundances",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Intrinsic growth rates for each microbial species (1D array)",
                "name": "growth_rates",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Matrix of interaction coefficients "
                "where A[i,j] represents the effect "
                "of species j on species i (2D "
                "array)",
                "name": "interaction_matrix",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Time points at which to evaluate the model",
                "name": "time_points",
                "type": "numpy.ndarray",
            },
        ],
    },
    {
        "description": "Predict the secondary structure of an RNA molecule using ViennaRNA.",
        "name": "predict_rna_secondary_structure",
        "optional_parameters": [
            {
                "default": "rna_structure",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The RNA sequence (consisting of A, U, G, C nucleotides)",
                "name": "rna_sequence",
                "type": "str",
            }
        ],
    },
    {
        "description": "Performs stochastic simulation of microbial population dynamics using the Gillespie algorithm.",
        "name": "simulate_microbial_population_dynamics",
        "optional_parameters": [
            {
                "default": 100,
                "description": "Maximum simulation time",
                "name": "max_time",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Number of stochastic simulations to run",
                "name": "num_simulations",
                "type": "int",
            },
            {
                "default": 100,
                "description": "Number of time points to record for trajectories",
                "name": "time_points",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Initial population sizes for each microbial species",
                "name": "initial_populations",
                "type": "array",
                "items": {"type": "integer"},
            },
            {
                "default": None,
                "description": "Per capita growth rates for each species",
                "name": "growth_rates",
                "type": "array",
                "items": {"type": "number"},
            },
            {
                "default": None,
                "description": "Per capita death/clearance rates for each species",
                "name": "clearance_rates",
                "type": "array",
                "items": {"type": "number"},
            },
            {
                "default": None,
                "description": "Maximum sustainable population for each species",
                "name": "carrying_capacities",
                "type": "array",
                "items": {"type": "number"},
            },
        ],
    },
]
