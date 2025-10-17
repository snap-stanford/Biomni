description = [
    {
        "description": "Analyze aortic diameter and geometry from cardiovascular "
        "imaging data to measure aortic root diameter, ascending "
        "aorta diameter, and calculate geometric parameters such as "
        "tortuosity and dilation indices.",
        "name": "analyze_aortic_diameter_and_geometry",
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
                "description": "Path to the cardiovascular imaging data (DICOM, JPG, PNG)",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze luminescence-based ATP assay data to determine intracellular ATP concentration.",
        "name": "analyze_atp_luminescence_assay",
        "optional_parameters": [
            {
                "default": "cell_count",
                "description": "Method used to normalize ATP values, either cell_count or protein_content",
                "name": "normalization_method",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to CSV file with normalization "
                "data or dictionary with sample IDs "
                "as keys and normalization values",
                "name": "normalization_data",
                "type": "str or dict",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV file containing "
                "luminescence readings from samples "
                "with columns for Sample_ID and "
                "Luminescence_Value",
                "name": "data_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to CSV file containing "
                "standard curve data with columns "
                "for ATP_Concentration (in nM) and "
                "Luminescence_Value",
                "name": "standard_curve_file",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze histological images of thrombus samples stained with "
        "H&E to identify and quantify different thrombus components "
        "(fresh, cellular lysis, endothelialization, fibroblastic "
        "reaction).",
        "name": "analyze_thrombus_histology",
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
                "description": "Path to the histological image of thrombus sample stained with H&E",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyzes intracellular calcium concentration using Rhod-2 "
        "fluorescent indicator from microscopy images.",
        "name": "analyze_intracellular_calcium_with_rhod2",
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
                "description": "Path to the background image (no cells, just media)",
                "name": "background_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the control image (cells without calcium stimulus)",
                "name": "control_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the sample image (cells with calcium stimulus)",
                "name": "sample_image_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Quantify the volume/density of immunofluorescence-labeled corneal nerve fibers.",
        "name": "quantify_corneal_nerve_fibers",
        "optional_parameters": [
            {
                "default": "./output",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": "otsu",
                "description": "Method for thresholding ('otsu', 'adaptive', 'manual')",
                "name": "threshold_method",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the immunofluorescence microscopy image file",
                "name": "image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Type of nerve fiber marker (e.g., 'βIII-tubulin', 'SP', 'L1CAM')",
                "name": "marker_type",
                "type": "str",
            },
        ],
    },
    {
        "description": "Segment cells and quantify protein expression levels from multichannel tissue images.",
        "name": "segment_and_quantify_cells_in_multiplexed_images",
        "optional_parameters": [
            {
                "default": 0,
                "description": "Index of the nuclear marker channel (typically DAPI)",
                "name": "nuclear_channel_index",
                "type": "int",
            },
            {
                "default": "./output",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the multichannel image file (tiff stack or similar format)",
                "name": "image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of marker names corresponding to each channel in the image",
                "name": "markers_list",
                "type": "array",
                "items": {"type": "string"},
            },
        ],
    },
    {
        "description": "Analyze bone microarchitecture parameters from 3D micro-CT "
        "images to calculate bone mineral density, bone volume, "
        "trabecular number, thickness, and separation.",
        "name": "analyze_bone_microct_morphometry",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Threshold value for bone segmentation. If None, Otsu's method will be used",
                "name": "threshold_value",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the micro-CT scan data file (TIFF stack or similar 3D format)",
                "name": "input_file_path",
                "type": "str",
            }
        ],
    },
]
