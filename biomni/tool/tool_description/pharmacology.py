description = [
    {
        "description": "Run DiffDock molecular docking using a protein PDB file and "
        "a SMILES string for the ligand, executing the process in a "
        "Docker container.",
        "name": "run_diffdock_with_smiles",
        "optional_parameters": [
            {
                "default": 0,
                "description": "GPU device ID to use for computation",
                "name": "gpu_device",
                "type": "int",
            },
            {
                "default": True,
                "description": "Whether to use GPU acceleration for docking",
                "name": "use_gpu",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the protein PDB file for docking",
                "name": "pdb_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "SMILES string representation of the ligand molecule",
                "name": "smiles_string",
                "type": "str",
            },
            {
                "default": None,
                "description": "Local directory path where docking results will be saved",
                "name": "local_output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs molecular docking using AutoDock Vina to predict "
        "binding affinities between small molecules and a receptor "
        "protein.",
        "name": "docking_autodock_vina",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Number of CPU cores to use for docking",
                "name": "ncpu",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing small molecules to dock",
                "name": "smiles_list",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Path to the receptor protein structure PDB file",
                "name": "receptor_pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "3D coordinates [x, y, z] of the docking box center",
                "name": "box_center",
                "type": "List[float]",
            },
            {
                "default": None,
                "description": "Dimensions [x, y, z] of the docking box",
                "name": "box_size",
                "type": "List[float]",
            },
        ],
    },
    {
        "description": "Runs AutoSite on a PDB file to identify potential binding "
        "sites and returns a research log with the results.",
        "name": "run_autosite",
        "optional_parameters": [
            {
                "default": 1.0,
                "description": "Grid spacing parameter for AutoSite calculation",
                "name": "spacing",
                "type": "float",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input PDB file",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory where AutoSite results will be saved",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Computes TxGNN model predictions for drug repurposing and "
        "returns the top predicted drugs with their scores for a "
        "given disease.",
        "name": "retrieve_topk_repurposing_drugs_from_disease_txgnn",
        "optional_parameters": [
            {
                "default": 5,
                "description": "The number of top drug predictions to return",
                "name": "k",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The name of the disease for which to retrieve drug predictions",
                "name": "disease_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the data lake",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Predicts ADMET (Absorption, Distribution, Metabolism, "
        "Excretion, Toxicity) properties for a list of compounds "
        "using pretrained models.",
        "name": "predict_admet_properties",
        "optional_parameters": [
            {
                "default": "MPNN",
                "description": "Type of model to use for ADMET prediction (options: 'MPNN', 'CNN', 'Morgan')",
                "name": "ADMET_model_type",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing chemical compounds to analyze",
                "name": "smiles_list",
                "type": "List[str]",
            }
        ],
    },
    {
        "description": "Predicts binding affinity between small molecules and a "
        "protein sequence using pre-trained deep learning models.",
        "name": "predict_binding_affinity_protein_1d_sequence",
        "optional_parameters": [
            {
                "default": "MPNN-CNN",
                "description": "Deep learning model architecture to "
                "use for binding affinity prediction "
                "(options: CNN-CNN, MPNN-CNN, "
                "Morgan-CNN, Morgan-AAC, "
                "Daylight-AAC)",
                "name": "affinity_model_type",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing chemical compounds",
                "name": "smiles_list",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Protein sequence in amino acid format",
                "name": "amino_acid_sequence",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyzes the stability of pharmaceutical formulations under accelerated storage conditions.",
        "name": "analyze_accelerated_stability_of_pharmaceutical_formulations",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "List of formulation dictionaries "
                "containing name, active ingredient, "
                "concentration, and excipients",
                "name": "formulations",
                "type": "List[dict]",
            },
            {
                "default": None,
                "description": "List of storage condition "
                "dictionaries containing "
                "temperature, humidity (optional), "
                "and description",
                "name": "storage_conditions",
                "type": "List[dict]",
            },
            {
                "default": None,
                "description": "List of time points in days to evaluate stability",
                "name": "time_points",
                "type": "List[int]",
            },
        ],
    },
    {
        "description": "Generates a detailed protocol for performing a 3D "
        "chondrogenic aggregate culture assay to evaluate compounds' "
        "effects on chondrogenesis.",
        "name": "run_3d_chondrogenic_aggregate_assay",
        "optional_parameters": [
            {
                "default": 21,
                "description": "Total duration of the culture period in days",
                "name": "culture_duration_days",
                "type": "int",
            },
            {
                "default": 7,
                "description": "Interval in days between measurements",
                "name": "measurement_intervals",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary with cell information "
                "including 'source', "
                "'passage_number', and "
                "'cell_density'",
                "name": "chondrocyte_cells",
                "type": "dict",
            },
            {
                "default": None,
                "description": "List of compounds to test, each with 'name', 'concentration', and 'vehicle' keys",
                "name": "test_compounds",
                "type": "list of dict",
            },
        ],
    },
    {
        "description": "Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.",
        "name": "grade_adverse_events_using_vcog_ctcae",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to a CSV file containing "
                "clinical evaluation data with "
                "columns: subject_id, time_point, "
                "symptom, severity, measurement "
                "(optional)",
                "name": "clinical_data_file",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.",
        "name": "analyze_radiolabeled_antibody_biodistribution",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Time points (hours) at which measurements were taken",
                "name": "time_points",
                "type": "List[float] or numpy.ndarray",
            },
            {
                "default": None,
                "description": "Dictionary where keys are tissue "
                "names and values are lists/arrays "
                "of %IA/g measurements corresponding "
                "to time_points. Must include "
                "'tumor' as one of the keys",
                "name": "tissue_data",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Estimate radiation absorbed doses to tumor and normal organs "
        "for alpha-particle radiotherapeutics using the Medical "
        "Internal Radiation Dose (MIRD) schema.",
        "name": "estimate_alpha_particle_radiotherapy_dosimetry",
        "optional_parameters": [
            {
                "default": "dosimetry_results.csv",
                "description": "Filename to save the dosimetry results",
                "name": "output_file",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary containing organ/tissue "
                "names as keys and a list of "
                "time-activity measurements as "
                "values. Each measurement should be "
                "a tuple of (time_hours, "
                "percent_injected_activity). Must "
                "include entries for all relevant "
                "organs including 'tumor'.",
                "name": "biodistribution_data",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing radiation "
                "parameters for the alpha-emitting "
                "radionuclide including "
                "'radionuclide', 'half_life_hours', "
                "'energy_per_decay_MeV', "
                "'radiation_weighting_factor', and "
                "'S_factors'.",
                "name": "radiation_parameters",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Perform a Methylome-wide Association Study (MWAS) to "
        "identify CpG sites significantly associated with CYP2C19 "
        "metabolizer status.",
        "name": "perform_mwas_cyp2c19_metabolizer_status",
        "optional_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "covariates to adjust for in the "
                "regression model (e.g., age, sex, "
                "smoking status).",
                "name": "covariates_path",
                "type": "str",
            },
            {
                "default": 0.05,
                "description": "P-value threshold for significance after multiple testing correction.",
                "name": "pvalue_threshold",
                "type": "float",
            },
            {
                "default": "significant_cpg_sites.csv",
                "description": "Filename to save significant CpG sites.",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "DNA methylation beta values. Rows "
                "should be samples, columns should "
                "be CpG sites.",
                "name": "methylation_data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "CYP2C19 metabolizer status for each "
                "sample. Should have a sample ID "
                "column and a status column.",
                "name": "metabolizer_status_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Calculate key physicochemical properties of a drug candidate molecule.",
        "name": "calculate_physicochemical_properties",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The molecular structure in SMILES format",
                "name": "smiles_string",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze tumor growth inhibition in xenograft models across different treatment groups.",
        "name": "analyze_xenograft_tumor_growth_inhibition",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing tumor volume measurements",
                "name": "data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing time points",
                "name": "time_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing tumor volume measurements",
                "name": "volume_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing treatment group labels",
                "name": "group_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing subject/mouse identifiers",
                "name": "subject_column",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs densitometric analysis of Western blot images to "
        "quantify relative protein expression.",
        "name": "analyze_western_blot",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the Western blot image file",
                "name": "blot_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of dictionaries containing "
                "information about target protein "
                "bands, each with 'name' and 'roi' "
                "(region of interest as [x, y, "
                "width, height])",
                "name": "target_bands",
                "type": "list of dict",
            },
            {
                "default": None,
                "description": "Dictionary with 'name' and 'roi' "
                "for the loading control protein "
                "(e.g., β-actin, GAPDH)",
                "name": "loading_control_band",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing information "
                "about antibodies used with "
                "'primary' and 'secondary' keys",
                "name": "antibody_info",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Query drug-drug interactions from DDInter database to identify potential interactions, mechanisms, and severity levels between specified drugs.",
        "name": "query_drug_interactions",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drug names to query for interactions",
                "name": "drug_names",
                "type": "List[str]",
            }
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Filter results by specific interaction types",
                "name": "interaction_types",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Filter results by severity levels (Major, Moderate, Minor)",
                "name": "severity_levels",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze safety of a drug combination for potential interactions using DDInter database with comprehensive risk assessment and clinical recommendations.",
        "name": "check_drug_combination_safety",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drugs to analyze for combination safety",
                "name": "drug_list",
                "type": "List[str]",
            }
        ],
        "optional_parameters": [
            {
                "default": True,
                "description": "Include interaction mechanism descriptions in results",
                "name": "include_mechanisms",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Include management recommendations in results",
                "name": "include_management",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze interaction mechanisms between two specific drugs providing detailed mechanistic insights and clinical significance assessment.",
        "name": "analyze_interaction_mechanisms",
        "required_parameters": [
            {
                "default": None,
                "description": "Pair of drug names to analyze (drug1, drug2)",
                "name": "drug_pair",
                "type": "Tuple[str, str]",
            }
        ],
        "optional_parameters": [
            {
                "default": True,
                "description": "Include detailed mechanistic information in analysis",
                "name": "detailed_analysis",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find alternative drugs that don't interact with contraindicated drugs using DDInter database for safer therapeutic substitutions.",
        "name": "find_alternative_drugs_ddinter",
        "required_parameters": [
            {
                "default": None,
                "description": "Drug to find alternatives for",
                "name": "target_drug",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of drugs to avoid interactions with",
                "name": "contraindicated_drugs",
                "type": "List[str]",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Limit search to specific therapeutic class",
                "name": "therapeutic_class",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Query FDA adverse event reports for specific drugs from the OpenFDA database to identify potential safety signals, reaction patterns, and regulatory intelligence.",
        "name": "query_fda_adverse_events",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to query for adverse events",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional date range as (start_date, end_date) in YYYY-MM-DD format",
                "name": "date_range",
                "type": "Tuple[str, str]",
            },
            {
                "default": None,
                "description": "Optional filter by severity levels ['serious', 'non_serious']",
                "name": "severity_filter",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Optional filter by outcomes ['life_threatening', 'hospitalization', 'death']",
                "name": "outcome_filter",
                "type": "List[str]",
            },
            {
                "default": 100,
                "description": "Maximum number of results to return",
                "name": "limit",
                "type": "int",
            },
        ],
    },
    {
        "description": "Retrieve FDA drug label information including indications, contraindications, warnings, and dosage information from the OpenFDA database.",
        "name": "get_fda_drug_label_info",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to query for label information",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional list of specific sections to retrieve ['indications_and_usage', 'contraindications', 'warnings', 'dosage_and_administration']",
                "name": "sections",
                "type": "List[str]",
            },
        ],
    },
    {
        "description": "Check for FDA drug recalls and enforcement actions from the OpenFDA database to identify safety concerns and regulatory actions.",
        "name": "check_fda_drug_recalls",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to check for recalls",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional filter by recall class ['Class I', 'Class II', 'Class III']",
                "name": "classification",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Optional date range for recalls as (start_date, end_date)",
                "name": "date_range",
                "type": "Tuple[str, str]",
            },
        ],
    },
    {
        "description": "Analyze safety signals across multiple drugs using OpenFDA adverse event data to identify patterns and comparative risk profiles.",
        "name": "analyze_fda_safety_signals",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drug names to analyze for safety signals",
                "name": "drug_list",
                "type": "List[str]",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional comparison time period as (start_date, end_date)",
                "name": "comparison_period",
                "type": "Tuple[str, str]",
            },
            {
                "default": 2.0,
                "description": "Threshold for signal detection",
                "name": "signal_threshold",
                "type": "float",
            },
        ],
    },
]
