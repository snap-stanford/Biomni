description = [
    {
        "description": "Predicts intrinsically disordered regions (IDRs) in a protein sequence using IUPred2A.",
        "name": "predict_protein_disorder_regions",
        "optional_parameters": [
            {
                "default": 0.5,
                "description": "The disorder score threshold above which a residue is considered disordered",
                "name": "threshold",
                "type": "float",
            },
            {
                "default": "disorder_prediction_results.csv",
                "description": "Filename to save the per-residue disorder scores",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The amino acid sequence of the protein to analyze",
                "name": "protein_sequence",
                "type": "str",
            }
        ],
    },
    {
        "description": "Quantifies cell morphology and cytoskeletal organization from fluorescence microscopy images.",
        "name": "analyze_cell_morphology_and_cytoskeleton",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": "otsu",
                "description": "Method for cell segmentation ('otsu', 'adaptive', or 'manual')",
                "name": "threshold_method",
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
        "description": "Quantify tissue deformation and flow dynamics from microscopy image sequence.",
        "name": "analyze_tissue_deformation_flow",
        "optional_parameters": [
            {
                "default": "results",
                "description": "Directory to save results",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 1.0,
                "description": "Physical scale of pixels (e.g., μm/pixel) for proper scaling of metrics",
                "name": "pixel_scale",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Sequence of microscopy images "
                "(either a list of file paths or a "
                "3D numpy array [time, height, "
                "width])",
                "name": "image_sequence",
                "type": "list or numpy.ndarray",
            }
        ],
    },
    {
        "name": "extract_frap_curve_from_image_stack",
        "description": (
            "Extract a normalized FRAP recovery curve from a multi-frame TIFF "
            "microscopy time-series. Performs background subtraction (optional), "
            "reference-ROI photobleach correction (optional), and double "
            "normalization against the pre-bleach mean intensity in the bleach "
            "ROI. Auto-detects the bleach frame from the largest single-frame "
            "intensity drop unless one is provided. Writes both the full curve "
            "and the post-bleach-only curve as CSV; the post-bleach CSV is the "
            "input expected by `fit_frap_recovery_curve`."
        ),
        "required_parameters": [
            {
                "name": "image_stack_path",
                "type": "str",
                "description": "Path to a TIFF time-series with shape (T, Y, X) or (T, C, Y, X). For multi-channel data, channel 0 is used.",
            },
            {
                "name": "bleach_roi",
                "type": "list",
                "description": "Three-element list [x_center_px, y_center_px, radius_px] specifying the bleached circular ROI in pixel coordinates.",
            },
        ],
        "optional_parameters": [
            {
                "name": "reference_roi",
                "type": "list",
                "default": None,
                "description": "Optional [x, y, r] for an unbleached reference region used to correct ongoing acquisition photobleaching.",
            },
            {
                "name": "background_roi",
                "type": "list",
                "default": None,
                "description": "Optional [x, y, r] outside any cell, for camera/background subtraction.",
            },
            {
                "name": "bleach_frame_index",
                "type": "int",
                "default": None,
                "description": "Index of the first post-bleach frame. If None, auto-detected as the frame with the largest intensity drop.",
            },
            {
                "name": "frame_interval_s",
                "type": "float",
                "default": 1.0,
                "description": "Time between successive frames in seconds.",
            },
            {
                "name": "output_folder",
                "type": "str",
                "default": "./tmp/",
                "description": "Directory where CSV outputs are written.",
            },
        ],
    },
    {
        "name": "fit_frap_recovery_curve",
        "description": (
            "Fit a normalized FRAP recovery curve and extract biophysical "
            "parameters: mobile fraction, half-time of recovery, diffusion "
            "coefficient (Soumpasis only), R-squared, and 95% confidence "
            "intervals from the covariance matrix. Supports the Soumpasis 1983 "
            "Bessel-function model for 2D diffusion in a uniform circular "
            "bleach spot, a single-exponential phenomenological model, and a "
            "two-component (fast + slow pool) double-exponential model. "
            "Writes the fit plot as PNG and the parameters as JSON."
        ),
        "required_parameters": [
            {
                "name": "curve_csv_path",
                "type": "str",
                "description": "Path to a CSV with columns 'time_s' (post-bleach time) and 'intensity_norm' (normalized to pre-bleach mean).",
            },
            {
                "name": "bleach_radius_um",
                "type": "float",
                "description": "Radius of the bleached spot in micrometers; required to convert tau_D into a diffusion coefficient.",
            },
        ],
        "optional_parameters": [
            {
                "name": "fit_model",
                "type": "str",
                "default": "soumpasis",
                "description": "One of 'soumpasis', 'exponential', or 'double_exponential'.",
            },
            {
                "name": "initial_guess",
                "type": "dict",
                "default": None,
                "description": "Optional dict with key 'p0' overriding the inferred starting parameters for scipy.optimize.curve_fit.",
            },
            {
                "name": "output_folder",
                "type": "str",
                "default": "./tmp/",
                "description": "Directory where the fit plot (PNG) and parameter file (JSON) are written.",
            },
        ],
    },
    {
        "name": "classify_condensate_material_state",
        "description": (
            "Classify FRAP fit parameters into liquid, gel-like, or arrested "
            "(glass-like) condensate material state using community-standard "
            "rule-of-thumb thresholds over mobile fraction, recovery half-time, "
            "and (for double-exponential fits) the slow-pool amplitude ratio. "
            "Optionally produces a 1-paragraph LLM-generated biological "
            "interpretation grounded in the classification and any user "
            "context note. Writes the verdict and reasoning as JSON."
        ),
        "required_parameters": [
            {
                "name": "fit_parameters_json",
                "type": "str",
                "description": "Path to the JSON file written by fit_frap_recovery_curve.",
            },
        ],
        "optional_parameters": [
            {
                "name": "output_folder",
                "type": "str",
                "default": "./tmp/",
                "description": "Directory for the output JSON.",
            },
            {
                "name": "use_llm_interpretation",
                "type": "bool",
                "default": False,
                "description": "If True, call an LLM via biomni.llm.get_llm to produce a prose interpretation. Default False keeps the tool deterministic.",
            },
            {
                "name": "llm",
                "type": "str",
                "default": "claude-sonnet-4-5",
                "description": "Model identifier passed to biomni.llm.get_llm when use_llm_interpretation=True.",
            },
            {
                "name": "context_note",
                "type": "str",
                "default": "",
                "description": "Optional free-text context (protein name, salt concentration, time after droplet formation, etc.) used by the LLM to ground its interpretation.",
            },
        ],
    },
    {
        "name": "run_frap_analysis_pipeline",
        "description": (
            "End-to-end FRAP analysis convenience wrapper. Takes a TIFF "
            "time-series and a bleach ROI, calls extract_frap_curve_from_image_stack, "
            "fit_frap_recovery_curve, and classify_condensate_material_state in "
            "sequence, then writes a single Markdown report alongside all "
            "intermediate artifacts. Use this as the default entry point for "
            "FRAP experiments; the agent should reach for the underlying tools "
            "directly only when fine-grained control over each step is needed."
        ),
        "required_parameters": [
            {
                "name": "image_stack_path",
                "type": "str",
                "description": "Path to a TIFF time-series with shape (T, Y, X) or (T, C, Y, X).",
            },
            {
                "name": "bleach_roi",
                "type": "list",
                "description": "[x_center_px, y_center_px, radius_px] for the bleached circular ROI.",
            },
            {
                "name": "bleach_radius_um",
                "type": "float",
                "description": "Radius of the bleached spot in micrometers.",
            },
        ],
        "optional_parameters": [
            {
                "name": "pixel_size_um",
                "type": "float",
                "default": None,
                "description": "Pixel size in micrometers. Reserved; not required when bleach_radius_um is provided directly.",
            },
            {
                "name": "reference_roi",
                "type": "list",
                "default": None,
                "description": "Optional [x, y, r] reference ROI for ongoing photobleach correction.",
            },
            {
                "name": "background_roi",
                "type": "list",
                "default": None,
                "description": "Optional [x, y, r] background ROI for camera/background subtraction.",
            },
            {
                "name": "bleach_frame_index",
                "type": "int",
                "default": None,
                "description": "Index of the first post-bleach frame; auto-detected if None.",
            },
            {
                "name": "frame_interval_s",
                "type": "float",
                "default": 1.0,
                "description": "Seconds between successive frames.",
            },
            {
                "name": "fit_model",
                "type": "str",
                "default": "soumpasis",
                "description": "One of 'soumpasis', 'exponential', 'double_exponential'.",
            },
            {
                "name": "use_llm_interpretation",
                "type": "bool",
                "default": False,
                "description": "Whether to call an LLM for a prose interpretation in the final report.",
            },
            {
                "name": "llm",
                "type": "str",
                "default": "claude-sonnet-4-5",
                "description": "LLM identifier when use_llm_interpretation=True.",
            },
            {
                "name": "context_note",
                "type": "str",
                "default": "",
                "description": "Optional context for the LLM interpretation.",
            },
            {
                "name": "output_folder",
                "type": "str",
                "default": "./tmp/",
                "description": "Directory for all output artifacts.",
            },
        ],
    },
]
