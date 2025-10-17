description = [
    {
        "description": "Quantify the percentage of cells in each cell cycle phase "
        "using Calcofluor white stained microscopy images.",
        "name": "quantify_cell_cycle_phases_from_microscopy",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save results",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of file paths to microscopy images of cells stained with Calcofluor white",
                "name": "image_paths",
                "type": "array",
                "items": {"type": "string"},
            }
        ],
    },
    {
        "description": "Quantify cell motility features from time-lapse microscopy "
        "images and cluster cells based on motility patterns.",
        "name": "quantify_and_cluster_cell_motility",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 3,
                "description": "Number of motility pattern clusters to identify",
                "name": "num_clusters",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to directory containing time-lapse microscopy images in sequential order",
                "name": "image_sequence_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Performs Fluorescence-Activated Cell Sorting (FACS) to "
        "enrich cell populations based on fluorescence "
        "characteristics.",
        "name": "perform_facs_cell_sorting",
        "optional_parameters": [
            {
                "default": None,
                "description": "Minimum threshold for the "
                "fluorescence parameter. Cells below "
                "this value will be excluded",
                "name": "threshold_min",
                "type": "float",
            },
            {
                "default": None,
                "description": "Maximum threshold for the "
                "fluorescence parameter. Cells above "
                "this value will be excluded",
                "name": "threshold_max",
                "type": "float",
            },
            {
                "default": "sorted_cells.csv",
                "description": "Filename to save the sorted cell population data",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the FCS file containing flow cytometry data",
                "name": "cell_suspension_data",
                "type": "str",
            },
            {
                "default": None,
                "description": "The fluorescence parameter to use for sorting (e.g., 'GFP', 'FITC', 'PE')",
                "name": "fluorescence_parameter",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze flow cytometry data to identify and quantify "
        "specific cell populations based on surface markers.",
        "name": "analyze_flow_cytometry_immunophenotyping",
        "optional_parameters": [
            {
                "default": None,
                "description": "Spillover/compensation matrix to correct for fluorescence overlap",
                "name": "compensation_matrix",
                "type": "numpy.ndarray",
            },
            {
                "default": "./results",
                "description": "Directory to save the results",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the FCS file containing flow cytometry data",
                "name": "fcs_file_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Dictionary defining the gating "
                "strategy. Each key is a population "
                "name, and each value is a list of "
                "tuples (marker, operator, "
                "threshold)",
                "name": "gating_strategy",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Quantifies metrics of mitochondrial morphology and membrane "
        "potential from fluorescence microscopy images.",
        "name": "analyze_mitochondrial_morphology_and_potential",
        "optional_parameters": [
            {
                "default": "./output",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the fluorescence microscopy "
                "image showing mitochondrial "
                "morphology (e.g., MTS-GFP)",
                "name": "morphology_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the fluorescence microscopy "
                "image showing mitochondrial "
                "membrane potential (e.g., TMRE "
                "staining)",
                "name": "potential_image_path",
                "type": "str",
            },
        ],
    },
]
