description = [
    {
        "description": "Analyze cell migration metrics from time-lapse microscopy images.",
        "name": "analyze_cell_migration_metrics",
        "optional_parameters": [
            {
                "default": 1.0,
                "description": "Conversion factor from pixels to micrometers",
                "name": "pixel_size_um",
                "type": "float",
            },
            {
                "default": 1.0,
                "description": "Time interval between consecutive frames in minutes",
                "name": "time_interval_min",
                "type": "float",
            },
            {
                "default": 10,
                "description": "Minimum number of frames a cell must be tracked to be included in analysis",
                "name": "min_track_length",
                "type": "int",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the directory containing time-lapse images or path to a multi-frame TIFF file",
                "name": "image_sequence_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Simulates CRISPR-Cas9 genome editing process including guide "
        "RNA design, delivery, and analysis.",
        "name": "perform_crispr_cas9_genome_editing",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "List of guide RNA sequences (20 "
                "nucleotides each) targeting the "
                "genomic region of interest",
                "name": "guide_rna_sequences",
                "type": "array",
                "items": {"type": "string"},
            },
            {
                "default": None,
                "description": "Target genomic sequence to be "
                "edited (should be longer than guide "
                "RNA and contain the target sites)",
                "name": "target_genomic_loci",
                "type": "str",
            },
            {
                "default": None,
                "description": "Type of cell or tissue being edited (affects delivery efficiency and editing outcomes)",
                "name": "cell_tissue_type",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze calcium imaging data to quantify neuronal activity "
        "metrics including cell counts, event rates, decay times, and "
        "signal-to-noise ratios.",
        "name": "analyze_calcium_imaging_data",
        "optional_parameters": [
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the time-series stack of fluorescence microscopy images (TIFF format)",
                "name": "image_stack_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyzes in vitro drug release kinetics from biomaterial formulations.",
        "name": "analyze_in_vitro_drug_release_kinetics",
        "optional_parameters": [
            {
                "default": "Drug",
                "description": "Name of the drug being analyzed",
                "name": "drug_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Total amount of drug initially "
                "loaded in the formulation. If None, "
                "the maximum concentration is used "
                "as 100%",
                "name": "total_drug_loaded",
                "type": "float",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Time points at which drug concentrations were measured (in hours)",
                "name": "time_points",
                "type": "array",
                "items": {"type": "number"},
            },
            {
                "default": None,
                "description": "Measured drug concentration at each time point",
                "name": "concentration_data",
                "type": "array",
                "items": {"type": "number"},
            },
        ],
    },
    {
        "description": "Quantifies morphological properties of myofibers in microscopy images of tissue sections.",
        "name": "analyze_myofiber_morphology",
        "optional_parameters": [
            {
                "default": 2,
                "description": "Channel index containing nuclei staining (DAPI, Hoechst, etc.)",
                "name": "nuclei_channel",
                "type": "int",
            },
            {
                "default": 1,
                "description": "Channel index containing myofiber staining (α-Actinin, etc.)",
                "name": "myofiber_channel",
                "type": "int",
            },
            {
                "default": "otsu",
                "description": "Method for thresholding ('otsu', 'adaptive', or 'manual')",
                "name": "threshold_method",
                "type": "str",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the microscopy image file "
                "(typically a multichannel image "
                "with nuclei and myofiber staining)",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Model neural activity trajectories and decode behavioral variables.",
        "name": "decode_behavior_from_neural_trajectories",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Number of principal components to use for dimensionality reduction",
                "name": "n_components",
                "type": "int",
            },
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Neural spiking activity data, shape (n_timepoints, n_neurons)",
                "name": "neural_data",
                "type": "array",
            },
            {
                "default": None,
                "description": "Behavioral data, shape (n_timepoints, n_behavioral_variables)",
                "name": "behavioral_data",
                "type": "array",
            },
        ],
    },
    {
        "description": "Simulate a whole-cell model represented as a system of ordinary differential equations (ODEs).",
        "name": "simulate_whole_cell_ode_model",
        "optional_parameters": [
            {
                "default": None,
                "description": "Function defining the system of "
                "ODEs. Should take arguments (t, y, "
                "*args) where t is time, y is the "
                "state vector, and args contains "
                "additional parameters. If None, a "
                "simple example whole-cell model "
                "will be used.",
                "name": "ode_function",
                "type": "object",
            },
            {
                "default": "(0, 100)",
                "description": "Tuple of (start_time, end_time) for the simulation.",
                "name": "time_span",
                "type": "array",
                "items": {"type": "number"},
            },
            {
                "default": 1000,
                "description": "Number of time points to evaluate.",
                "name": "time_points",
                "type": "int",
            },
            {
                "default": "'LSODA'",
                "description": "Numerical integration method to use (e.g., 'RK45', 'LSODA', 'BDF').",
                "name": "method",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Initial values for each state "
                "variable in the model. If dict, "
                "keys are variable names and values "
                "are initial concentrations/values. "
                "If array-like, order must match the "
                "order expected by the ODE function.",
                "name": "initial_conditions",
                "type": "object",
            },
            {
                "default": None,
                "description": "Model parameters required by the "
                "ODE function. Keys are parameter "
                "names and values are parameter "
                "values.",
                "name": "parameters",
                "type": "dict",
            },
        ],
    },
]
