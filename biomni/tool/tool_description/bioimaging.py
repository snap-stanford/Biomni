description = [
    {
        "description": "Perform comprehensive image segmentation using the nnU-Net deep-learning framework. Includes automatic input preparation, modality splitting for 4D NIfTI files, advanced preprocessing, verification, configurable hyperparameters, and automated label postprocessing.",
        "name": "segment_with_nn_unet",
        "required_parameters": [
            {
                "default": None,
                "description": "Path to input image file (4D NIfTI) or directory containing images for segmentation.",
                "name": "image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to store segmentation outputs.",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Task ID defined by nnU-Net (e.g., 'Task001_BrainTumour').",
                "name": "task_id",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": "3d_fullres",
                "description": "nnU-Net model type ('2d', '3d_fullres', '3d_lowres', or '3d_cascade_fullres').",
                "name": "model_type",
                "type": "str",
            },
            {
                "default": [0, 1, 2, 3, 4],
                "description": "Model folds to use for ensemble predictions.",
                "name": "folds",
                "type": "list",
            },
            {
                "default": False,
                "description": "Enable Test-Time Augmentation for improved accuracy.",
                "name": "use_tta",
                "type": "bool",
            },
            {
                "default": 1,
                "description": "Number of CPU threads to utilize for preprocessing and saving.",
                "name": "num_threads",
                "type": "int",
            },
            {
                "default": True,
                "description": "Use mixed precision for faster inference.",
                "name": "mixed_precision",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Enable detailed logging and progress output.",
                "name": "verbose",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Automatically prepare input data for nnUNet (split 4D files, handle modalities).",
                "name": "auto_prepare_input",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to nnUNet results folder. If not specified, will use environment variable nnUNet_RESULTS_FOLDER or default locations.",
                "name": "results_folder",
                "type": "str",
            },
        ],
    },
    {
        "description": "Split a 4D NIfTI file into separate modality files (FLAIR, T1w, t1gd, T2w) for nnUNet processing.",
        "name": "split_modalities",
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the 4D NIfTI file to split.",
                "name": "input_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save the split modality files.",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": "BRAT",
                "description": "Base name for the case (will create files like BRAT_0000.nii.gz, BRAT_0001.nii.gz, etc.).",
                "name": "case_name",
                "type": "str",
            }
        ],
    },
    {
        "description": "Prepare input data for nnUNet by automatically handling both 4D and pre-split modality files.",
        "name": "prepare_input_for_nnunet",
        "required_parameters": [
            {
                "default": None,
                "description": "Path to input file or directory.",
                "name": "input_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save prepared files.",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": "BRAT",
                "description": "Base name for the case.",
                "name": "case_name",
                "type": "str",
            }
        ],
    },
    {
        "description": "Create and save visualization of segmentation results using nilearn. Generates multiple views (overlay, axial, sagittal, coronal) and saves them as PNG files.",
        "name": "create_segmentation_visualization",
        "required_parameters": [
            {
                "default": None,
                "description": "Path to original MRI file for background image.",
                "name": "original_mri",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to segmentation file to overlay on the MRI.",
                "name": "segmentation",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": "./visualization_output",
                "description": "Directory to save visualization images (PNG files).",
                "name": "output_dir",
                "type": "str",
            }
        ],
    },
    {
        "description": "Setup nnU-Net environment variables and create necessary directories.",
        "name": "setup_nnunet_environment",
        "required_parameters": [],
        "optional_parameters": [],
    },
]