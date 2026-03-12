description = [
    {
        "description": "Perform liftover of genomic coordinates between hg19 and "
        "hg38 formats with detailed intermediate steps.",
        "name": "liftover_coordinates",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Chromosome number (e.g., '1', 'X')",
                "name": "chromosome",
                "type": "str",
            },
            {
                "default": None,
                "description": "Genomic position",
                "name": "position",
                "type": "int",
            },
            {
                "default": None,
                "description": "Input genome build ('hg19' or 'hg38')",
                "name": "input_format",
                "type": "str",
            },
            {
                "default": None,
                "description": "Output genome build ('hg19' or 'hg38')",
                "name": "output_format",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to liftover chain files",
                "name": "data_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs Bayesian fine-mapping from GWAS summary statistics "
        "using deep variational inference to compute posterior "
        "inclusion probabilities and credible sets for putative "
        "causal variants.",
        "name": "bayesian_finemapping_with_deep_vi",
        "optional_parameters": [
            {
                "default": 5000,
                "description": "Number of training iterations for the variational inference algorithm",
                "name": "n_iterations",
                "type": "int",
            },
            {
                "default": 0.01,
                "description": "Learning rate for the optimization algorithm",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 64,
                "description": "Hidden dimension size for the neural network",
                "name": "hidden_dim",
                "type": "int",
            },
            {
                "default": 0.95,
                "description": "Threshold for defining the credible set (e.g., 0.95 for a 95% credible set)",
                "name": "credible_threshold",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "GWAS summary statistics with "
                "variant_id, effect_size, pvalue, "
                "and optional se columns",
                "name": "gwas_summary_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Linkage disequilibrium matrix with pairwise correlations between variants",
                "name": "ld_matrix",
                "type": "numpy.ndarray",
            },
        ],
    },
    {
        "description": "Analyzes and categorizes mutations induced by Cas9 at target sites.",
        "name": "analyze_cas9_mutation_outcomes",
        "optional_parameters": [
            {
                "default": None,
                "description": "Dictionary mapping sequence IDs to "
                "cell line information (e.g., "
                "wildtype, knockout gene)",
                "name": "cell_line_info",
                "type": "dict",
            },
            {
                "default": "cas9_mutation_analysis",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary mapping sequence IDs to reference DNA sequences (strings)",
                "name": "reference_sequences",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Nested dictionary: {sequence_id: "
                "{read_id: sequence}} containing the "
                "edited/mutated sequences for each "
                "reference",
                "name": "edited_sequences",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Analyzes CRISPR-Cas9 genome editing results by comparing original and edited sequences.",
        "name": "analyze_crispr_genome_editing",
        "optional_parameters": [
            {
                "default": None,
                "description": "The homology-directed repair template sequence, if used",
                "name": "repair_template",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The original DNA sequence before CRISPR-Cas9 editing",
                "name": "original_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "The DNA sequence after CRISPR-Cas9 editing",
                "name": "edited_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "The CRISPR guide RNA (crRNA) sequence used for targeting",
                "name": "guide_rna",
                "type": "str",
            },
        ],
    },
    {
        "description": "Simulate DNA sequences with specified demographic and coalescent histories using msprime.",
        "name": "simulate_demographic_history",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Number of sample sequences to simulate",
                "name": "num_samples",
                "type": "int",
            },
            {
                "default": 100000,
                "description": "Length of the simulated sequence in base pairs",
                "name": "sequence_length",
                "type": "int",
            },
            {
                "default": 1e-08,
                "description": "Per-base recombination rate",
                "name": "recombination_rate",
                "type": "float",
            },
            {
                "default": 1e-08,
                "description": "Per-base mutation rate",
                "name": "mutation_rate",
                "type": "float",
            },
            {
                "default": "constant",
                "description": "Type of demographic model to "
                "simulate (constant, bottleneck, "
                "expansion, contraction, sawtooth)",
                "name": "demographic_model",
                "type": "str",
            },
            {
                "default": None,
                "description": "Parameters specific to the chosen demographic model",
                "name": "demographic_params",
                "type": "dict",
            },
            {
                "default": "kingman",
                "description": "Type of coalescent model to use (kingman, beta)",
                "name": "coalescent_model",
                "type": "str",
            },
            {
                "default": None,
                "description": "Parameter for beta-coalescent model",
                "name": "beta_coalescent_param",
                "type": "float",
            },
            {
                "default": None,
                "description": "Seed for random number generator",
                "name": "random_seed",
                "type": "int",
            },
            {
                "default": "simulated_sequences.vcf",
                "description": "Filename to save the simulated sequences in VCF format",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Identifies binding sites for a specific transcription factor in a genomic sequence.",
        "name": "identify_transcription_factor_binding_sites",
        "optional_parameters": [
            {
                "default": 0.8,
                "description": "Minimum score threshold for reporting binding sites (0.0-1.0)",
                "name": "threshold",
                "type": "float",
            },
            {
                "default": None,
                "description": "Path to save the results",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The genomic DNA sequence to analyze",
                "name": "sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the transcription factor to search for (e.g., 'Hsf1', 'GATA1')",
                "name": "tf_name",
                "type": "str",
            },
        ],
    },
    {
        "description": "Fit a linear mixed model for genomic prediction using genotype and phenotype data.",
        "name": "fit_genomic_prediction_model",
        "optional_parameters": [
            {
                "default": None,
                "description": "Matrix of fixed effects (e.g., "
                "environment, management), with "
                "individuals in rows and effects in "
                "columns.",
                "name": "fixed_effects",
                "type": "numpy.ndarray",
            },
            {
                "default": "additive",
                "description": 'Type of genetic model to fit: "additive" or "additive_dominance".',
                "name": "model_type",
                "type": "str",
            },
            {
                "default": "genomic_prediction_results.csv",
                "description": "File name to save the results.",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Matrix of genotype data, with "
                "individuals in rows and markers in "
                "columns. Values are typically coded "
                "as 0, 1, 2 for additive models or "
                "with specific encoding for "
                "dominance effects.",
                "name": "genotypes",
                "type": "numpy.ndarray",
            },
            {
                "default": None,
                "description": "Vector or matrix of phenotype data, with individuals in rows and traits in columns.",
                "name": "phenotypes",
                "type": "numpy.ndarray",
            },
        ],
    },
    {
        "description": "Performs PCR amplification of a target transgene and "
        "visualizes results using agarose gel electrophoresis.",
        "name": "perform_pcr_and_gel_electrophoresis",
        "optional_parameters": [
            {
                "default": None,
                "description": "Forward primer sequence. If not provided, will be designed based on target_region",
                "name": "forward_primer",
                "type": "str",
            },
            {
                "default": None,
                "description": "Reverse primer sequence. If not provided, will be designed based on target_region",
                "name": "reverse_primer",
                "type": "str",
            },
            {
                "default": None,
                "description": "Tuple of (start, end) positions for the target region in the genomic DNA",
                "name": "target_region",
                "type": "tuple",
            },
            {
                "default": 58,
                "description": "Annealing temperature for PCR in °C",
                "name": "annealing_temp",
                "type": "float",
            },
            {
                "default": 30,
                "description": "Extension time in seconds",
                "name": "extension_time",
                "type": "int",
            },
            {
                "default": 35,
                "description": "Number of PCR cycles",
                "name": "cycles",
                "type": "int",
            },
            {
                "default": 2.0,
                "description": "Percentage of agarose gel",
                "name": "gel_percentage",
                "type": "float",
            },
            {
                "default": "pcr_result",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to file containing genomic DNA sequence in FASTA format or the sequence itself",
                "name": "genomic_dna",
                "type": "str",
            }
        ],
    },
    {
        "description": "Perform phylogenetic analysis on a set of protein sequences. "
        "This function aligns sequences, constructs a phylogenetic "
        "tree, and visualizes evolutionary relationships.",
        "name": "analyze_protein_phylogeny",
        "optional_parameters": [
            {
                "default": "./",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": "clustalw",
                "description": 'Method for sequence alignment: "clustalw", "muscle", or "pre-aligned"',
                "name": "alignment_method",
                "type": "str",
            },
            {
                "default": "fasttree",
                "description": 'Method for tree construction: "iqtree" or fallback to neighbor-joining',
                "name": "tree_method",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to a FASTA file containing "
                "protein sequences or a string with "
                "FASTA-formatted sequences",
                "name": "fasta_sequences",
                "type": "str",
            }
        ],
    },
]
