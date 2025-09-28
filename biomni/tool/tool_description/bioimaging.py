description = [
    # Segmentation functions
    {
        "description": "Split a 4D NIfTI file into separate modality files for nnUNet processing. "
        "Handles BRATS dataset format with FLAIR, T1w, t1gd, and T2w modalities.",
        "name": "split_modalities",
        "optional_parameters": [
            {
                "default": "BRAT",
                "description": "Base name for the case files",
                "name": "case_name",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the 4D NIfTI file to split",
                "name": "input_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save the split modality files",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Prepare input data for nnUNet by handling both 4D and pre-split modality files. "
        "Automatically detects file format and prepares data accordingly.",
        "name": "prepare_input_for_nnunet",
        "optional_parameters": [
            {
                "default": "BRAT",
                "description": "Base name for the case files",
                "name": "case_name",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to input file or directory",
                "name": "input_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save prepared files",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Segment images using nnUNet with proper environment setup. "
        "Supports brain tumor segmentation and other medical image segmentation tasks.",
        "name": "segment_with_nn_unet",
        "optional_parameters": [
            {
                "default": "3d_fullres",
                "description": "Model type for segmentation",
                "name": "model_type",
                "type": "str",
            },
            {
                "default": [0, 1, 2, 3, 4],
                "description": "Model folds to use for ensemble prediction",
                "name": "folds",
                "type": "list",
            },
            {
                "default": False,
                "description": "Use test time augmentation",
                "name": "use_tta",
                "type": "bool",
            },
            {
                "default": 1,
                "description": "Number of threads for preprocessing",
                "name": "num_threads",
                "type": "int",
            },
            {
                "default": True,
                "description": "Use mixed precision for faster inference",
                "name": "mixed_precision",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Enable verbose logging",
                "name": "verbose",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Automatically prepare input for nnUNet",
                "name": "auto_prepare_input",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to nnUNet results folder",
                "name": "results_folder",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to input image file or directory",
                "name": "image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save segmentation results",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Task identifier (e.g., 'Task001_BrainTumour')",
                "name": "task_id",
                "type": "str",
            },
        ],
    },
    {
        "description": "Create and save visualization of segmentation results using nilearn. "
        "Generates overlay plots and multiple anatomical views.",
        "name": "create_segmentation_visualization",
        "optional_parameters": [
            {
                "default": "./visualization_output",
                "description": "Directory to save visualization images",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to original MRI file",
                "name": "original_mri",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to segmentation file",
                "name": "segmentation",
                "type": "str",
            },
        ],
    },
    # Image registration functions
    {
        "description": "Perform rigid image registration between two medical images using SimpleITK. "
        "Rigid registration handles translation and rotation only, preserving shape and size. "
        "Includes preprocessing, similarity metrics calculation, and visualization generation.",
        "name": "quick_rigid_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform affine image registration between two medical images using SimpleITK. "
        "Affine registration handles translation, rotation, scaling, and shearing. "
        "More flexible than rigid registration but still preserves parallel lines.",
        "name": "quick_affine_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform deformable (B-spline) image registration between two medical images using SimpleITK. "
        "Deformable registration allows for local non-linear transformations, handling complex deformations. "
        "Most flexible but computationally intensive registration method.",
        "name": "quick_deformable_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
            {
                "default": 4,
                "description": "Number of B-spline control points per dimension for deformable registration",
                "name": "number_of_control_points",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform batch registration of multiple images to a single reference image. "
        "Automatically processes all medical image files in a directory and registers them to the fixed reference. "
        "Supports rigid, affine, or deformable registration for all images.",
        "name": "batch_register_images",
        "optional_parameters": [
            {
                "default": "rigid",
                "description": "Type of registration to perform: 'rigid', 'affine', or 'deformable'",
                "name": "transform_type",
                "type": "str",
            },
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots for each registration",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path containing multiple images to register (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "moving_images_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results for all images",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Calculate similarity metrics between two medical images. "
        "Supports mutual information, mean squared error, correlation, and normalized cross correlation.",
        "name": "calculate_similarity_metrics",
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the first image file",
                "name": "image1_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the second image file",
                "name": "image2_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Create visualization plots for registration results. "
        "Generates comparison plots, difference images, overlays, and metric charts.",
        "name": "create_registration_visualization",
        "optional_parameters": [
            {
                "default": "registration",
                "description": "Prefix for output files",
                "name": "prefix",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the original moving image file",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the registered image file",
                "name": "registered_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory to save visualization files",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
]
