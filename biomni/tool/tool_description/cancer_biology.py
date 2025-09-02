description = [
    {
        "description": "Analyze DNA Damage Response (DDR) network alterations and dependencies in cancer samples.",
        "name": "analyze_ddr_network_in_cancer",
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
                "description": "Path to gene expression data file (CSV format with genes as rows, samples as columns)",
                "name": "expression_data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to mutation data file (CSV "
                "format with genes as rows, samples "
                "as columns, values indicating "
                "mutation status)",
                "name": "mutation_data_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze flow cytometry data to quantify senescent and apoptotic cell populations.",
        "name": "analyze_cell_senescence_and_apoptosis",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the FCS file containing "
                "flow cytometry data with "
                "measurements for "
                "senescence-associated "
                "β-galactosidase (SA-β-Gal) and "
                "Annexin V/7-AAD staining",
                "name": "fcs_file_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Detects and annotates somatic mutations in tumor samples "
        "compared to matched normal samples using GATK Mutect2 for "
        "variant calling, GATK FilterMutectCalls for filtering, and "
        "SnpEff for functional annotation.",
        "name": "detect_and_annotate_somatic_mutations",
        "optional_parameters": [
            {
                "default": "GRCh38.105",
                "description": "SnpEff database to use for annotation",
                "name": "snpeff_database",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the tumor sample BAM file",
                "name": "tumor_bam",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the matched normal sample BAM file",
                "name": "normal_bam",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the reference genome FASTA file",
                "name": "reference_genome",
                "type": "str",
            },
            {
                "default": None,
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            },
        ],
    },
    {
        "description": "Detects and characterizes structural variations (SVs) in "
        "genomic sequencing data using LUMPY for SV detection "
        "followed by annotation with COSMIC and/or ClinVar databases.",
        "name": "detect_and_characterize_structural_variations",
        "optional_parameters": [
            {
                "default": None,
                "description": "Path to the COSMIC database for cancer annotation",
                "name": "cosmic_db_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the ClinVar database for clinical annotation",
                "name": "clinvar_db_path",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the aligned sequencing data in BAM format",
                "name": "bam_file_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the reference genome in FASTA format",
                "name": "reference_genome_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory where results will be saved",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs Non-negative Matrix Factorization (NMF) on gene "
        "expression data to extract metagenes and their associated "
        "sample weights for tumor subtype identification.",
        "name": "perform_gene_expression_nmf_analysis",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Number of metagenes (components) to extract.",
                "name": "n_components",
                "type": "int",
            },
            {
                "default": True,
                "description": "Whether to normalize the expression data before applying NMF.",
                "name": "normalize",
                "type": "bool",
            },
            {
                "default": "nmf_results",
                "description": "Directory to save the output files.",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 42,
                "description": "Random seed for reproducibility.",
                "name": "random_state",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to a CSV or TSV file "
                "containing gene expression data "
                "with genes as rows and samples as "
                "columns. Values should be "
                "non-negative.",
                "name": "expression_data_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "CNVkit-based copy number workflow performing CNV segmentation, purity & ploidy approximation, simplified HRD-style metrics, and focal amplification/deletion detection in selected genes.",
        "name": "analyze_copy_number_purity_ploidy_and_focal_events",
        "optional_parameters": [
            {
                "name": "normal_bam",
                "type": "str",
                "default": None,
                "description": "Path to matched normal BAM (recommended for CNVkit)",
            },
            {
                "name": "output_dir",
                "type": "str",
                "default": "cn_analysis_results",
                "description": "Directory to store CNVkit workflow outputs",
            },
            {
                "name": "targets_bed",
                "type": "str",
                "default": None,
                "description": "BED file of target regions for CNVkit (panel/exome)",
            },
            {
                "name": "antitargets_bed",
                "type": "str",
                "default": None,
                "description": "BED file of antitarget regions for CNVkit",
            },
            {
                "name": "gene_bed",
                "type": "str",
                "default": None,
                "description": "Gene BED (chrom, start, end, gene) for focal event annotation",
            },
            {
                "name": "focal_genes",
                "type": "list[str]",
                "default": None,
                "description": "Genes to highlight for focal events (defaults: MYC, ERBB2, CDKN2A)",
            },
            {
                "name": "log2_amp_threshold",
                "type": "float",
                "default": 1.0,
                "description": "Log2 ratio threshold for focal amplification calls",
            },
            {
                "name": "log2_del_threshold",
                "type": "float",
                "default": -1.0,
                "description": "Log2 ratio threshold for focal deep deletion calls",
            },
        ],
        "required_parameters": [
            {
                "name": "tumor_bam",
                "type": "str",
                "default": None,
                "description": "Path to tumor BAM (indexed)",
            },
            {
                "name": "reference_genome",
                "type": "str",
                "default": None,
                "description": "Path to reference genome FASTA",
            },
        ],
    },
]
