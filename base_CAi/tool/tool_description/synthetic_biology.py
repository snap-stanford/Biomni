description = [
    {
        "description": "Engineer a bacterial genome by integrating therapeutic genetic parts for therapeutic delivery.",
        "name": "engineer_bacterial_genome_for_therapeutic_delivery",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the file containing the bacterial genome sequence in FASTA format",
                "name": "bacterial_genome_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Dictionary containing genetic parts "
                "to be integrated (promoters, genes, "
                "terminators, cargo)",
                "name": "genetic_parts",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Analyze bacterial growth data and extract growth parameters from OD600 measurements.",
        "name": "analyze_bacterial_growth_rate",
        "optional_parameters": [
            {
                "default": "Unknown strain",
                "description": "Name of the bacterial strain being analyzed",
                "name": "strain_name",
                "type": "str",
            },
            {
                "default": "./",
                "description": "Directory where to save the output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Time points at which OD600 measurements were taken (in hours)",
                "name": "time_points",
                "type": "List or numpy.ndarray",
            },
            {
                "default": None,
                "description": "Optical density (OD600) measurements corresponding to each time point",
                "name": "od_measurements",
                "type": "List or numpy.ndarray",
            },
        ],
    },
    {
        "description": "Analyze sequencing data to extract, quantify and determine lineage relationships of barcodes.",
        "name": "analyze_barcode_sequencing_data",
        "optional_parameters": [
            {
                "default": None,
                "description": "Regular expression pattern to identify barcodes. If None, will use flanking sequences",
                "name": "barcode_pattern",
                "type": "str",
            },
            {
                "default": None,
                "description": "5' flanking sequence of the barcode region",
                "name": "flanking_seq_5prime",
                "type": "str",
            },
            {
                "default": None,
                "description": "3' flanking sequence of the barcode region",
                "name": "flanking_seq_3prime",
                "type": "str",
            },
            {
                "default": 5,
                "description": "Minimum count threshold for considering a barcode",
                "name": "min_count",
                "type": "int",
            },
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input sequencing file in FASTQ or FASTA format",
                "name": "input_file",
                "type": "str",
            }
        ],
    },
    {
        "description": "Performs bifurcation analysis on a dynamical system and generates a bifurcation diagram.",
        "name": "analyze_bifurcation_diagram",
        "optional_parameters": [
            {
                "default": "Dynamical System",
                "description": "Name of the dynamical system being analyzed, used for plot titles.",
                "name": "system_name",
                "type": "str",
            },
            {
                "default": "./",
                "description": "Directory to save the output files.",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "A 2D array where each row "
                "represents a time series for a "
                "specific parameter value. Shape "
                "should be (n_parameter_values, "
                "n_time_points).",
                "name": "time_series_data",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "1D array of parameter values "
                "corresponding to each time series. "
                "Shape should be "
                "(n_parameter_values,).",
                "name": "parameter_values",
                "type": "numpy.ndarray",
            },
        ],
    },
    {
        "description": "Generate a mathematical model of a biochemical network in SBML format.",
        "name": "create_biochemical_network_sbml_model",
        "optional_parameters": [
            {
                "default": "biochemical_model.xml",
                "description": "File path to save the SBML model",
                "name": "output_file",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of dictionaries representing "
                "reactions with id, name, reactants, "
                "products, and reversible properties",
                "name": "reaction_network",
                "type": "List[dict]",
            },
            {
                "default": None,
                "description": "Dictionary mapping reaction IDs to kinetic law parameters with law_type and parameters",
                "name": "kinetic_parameters",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Analyzes and optimizes a DNA/RNA sequence for improved "
        "expression in a heterologous host organism.",
        "name": "optimize_codons_for_heterologous_expression",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The DNA or RNA sequence of the "
                "target gene to be optimized. Should "
                "contain complete codons (length "
                "divisible by 3).",
                "name": "target_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Dictionary mapping codons to their "
                "usage frequency in the host "
                "organism. Format: {'AUG': 0.8, "
                "'GCC': 0.6, ...} or {'ATG': 0.8, "
                "'GCC': 0.6, ...}",
                "name": "host_codon_usage",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Simulate gene regulatory circuit dynamics with growth feedback.",
        "name": "simulate_gene_circuit_with_growth_feedback",
        "optional_parameters": [
            {
                "default": 100,
                "description": "Total simulation time",
                "name": "simulation_time",
                "type": "float",
            },
            {
                "default": 1000,
                "description": "Number of time points to sample",
                "name": "time_points",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Adjacency matrix representing the "
                "gene circuit topology. Positive "
                "values indicate activation, "
                "negative values indicate "
                "repression. Shape should be "
                "(n_genes, n_genes) where n_genes is "
                "the number of genes in the circuit.",
                "name": "circuit_topology",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Dictionary containing kinetic "
                "parameters: 'basal_rates', "
                "'degradation_rates', "
                "'hill_coefficients', and "
                "'threshold_constants' for the gene "
                "circuit.",
                "name": "kinetic_params",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing "
                "growth-related parameters: "
                "'max_growth_rate', "
                "'growth_inhibition', and "
                "'gene_growth_weights'.",
                "name": "growth_params",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Identifies functional domains within a Fatty Acid Synthase "
        "(FAS) sequence and predicts their roles.",
        "name": "identify_fas_functional_domains",
        "optional_parameters": [
            {
                "default": "protein",
                "description": 'Type of sequence provided - "protein" or "nucleotide"',
                "name": "sequence_type",
                "type": "str",
            },
            {
                "default": "fas_domains_report.txt",
                "description": "Name of the output file to save the detailed domain report",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The nucleotide or protein sequence of a FAS gene",
                "name": "sequence",
                "type": "str",
            }
        ],
    },
]
