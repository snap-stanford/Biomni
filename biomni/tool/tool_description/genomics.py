description = [
    {
        "description": "Annotate cell types based on gene markers and transferred "
        "labels using LLM. After leiden clustering, annotate clusters "
        "using differentially expressed genes and optionally "
        "incorporate transferred labels from reference datasets.",
        "name": "annotate_celltype_scRNA",
        "optional_parameters": [
            {
                "default": "leiden",
                "description": "Clustering method to use for cell type annotation",
                "name": "cluster",
                "type": "str",
            },
            {
                "default": "claude-3-5-sonnet-20241022",
                "description": "Language model instance for cell type prediction",
                "name": "llm",
                "type": "str",
            },
            {
                "default": None,
                "description": "Transferred cell type composition for each cluster",
                "name": "composition",
                "type": "pd.DataFrame",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the AnnData file containing scRNA-seq data",
                "name": "adata_filename",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory containing the data files",
                "name": "data_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": 'Information about the scRNA-seq data (e.g., "homo sapiens, brain tissue, normal")',
                "name": "data_info",
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
        "description": "Perform cell type annotation of single-cell RNA-seq data using Panhuman Azimuth Neural Network. "
        "This function implements the Panhuman Azimuth workflow for cell type annotation using the "
        "panhumanpy package, providing hierarchical cell type labels for tissues across the human body. ",
        "name": "annotate_celltype_with_panhumanpy",
        "optional_parameters": [
            {
                "default": None,
                "description": "Column name in adata.var containing gene symbols (default: None, uses index)",
                "name": "feature_names_col",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to perform additional label refinement for consistent granularity",
                "name": "refine",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to generate ANN embeddings and UMAP",
                "name": "umap",
                "type": "bool",
            },
            {
                "default": "./output",
                "description": "Directory to save results",
                "name": "output_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the AnnData file containing scRNA-seq data",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Create scVI and scANVI embeddings for single-cell RNA-seq "
        "data, saving the results to an AnnData object.",
        "name": "create_scvi_embeddings_scRNA",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Filename of the AnnData object to load",
                "name": "adata_filename",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column name in adata.obs for batch information",
                "name": "batch_key",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column name in adata.obs for cell type labels",
                "name": "label_key",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path where the AnnData file is located and where output will be saved",
                "name": "data_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs batch integration on single-cell RNA-seq data using "
        "Harmony and saves the integrated embeddings.",
        "name": "create_harmony_embeddings_scRNA",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Filename of the AnnData object to load",
                "name": "adata_filename",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column name in adata.obs that defines the batch variable for integration",
                "name": "batch_key",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path where the input file is located and output will be saved",
                "name": "data_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Generate UCE embeddings for single-cell RNA-seq data and map "
        "them to a reference dataset for cell type annotation.",
        "name": "get_uce_embeddings_scRNA",
        "optional_parameters": [
            {
                "default": "/dfs/project/bioagentos/data/singlecell/",
                "description": "Root directory for single-cell data storage",
                "name": "DATA_ROOT",
                "type": "str",
            },
            {
                "default": None,
                "description": "Custom command line arguments to pass to the UCE script",
                "name": "custom_args",
                "type": "List[str]",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Filename of the AnnData object to process",
                "name": "adata_filename",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory where the input data is stored and output will be saved",
                "name": "data_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Map cell embeddings from the input dataset to the Integrated "
        "Megascale Atlas reference dataset using UCE embeddings.",
        "name": "map_to_ima_interpret_scRNA",
        "optional_parameters": [
            {
                "default": None,
                "description": "Dictionary of custom arguments "
                "including 'n_neighbors' and "
                "'metric' for nearest neighbor "
                "search",
                "name": "custom_args",
                "type": "dict",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Filename of the AnnData object to be mapped",
                "name": "adata_filename",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory containing the AnnData file",
                "name": "data_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Given a gene name, fetch RNA-seq expression data showing the "
        "top K tissues with highest transcripts-per-million (TPM) "
        "values.",
        "name": "get_rna_seq_archs4",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The number of tissues to return",
                "name": "K",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The gene name for which RNA-seq data is being fetched",
                "name": "gene_name",
                "type": "str",
            }
        ],
    },
    {
        "description": "Returns a list of supported databases for gene set enrichment analysis.",
        "name": "get_gene_set_enrichment_analysis_supported_database_list",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": "Perform enrichment analysis for a list of genes, with "
        "optional background gene set and plotting functionality.",
        "name": "gene_set_enrichment_analysis",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Number of top pathways to return",
                "name": "top_k",
                "type": "int",
            },
            {
                "default": "ontology",
                "description": "Database to use for enrichment analysis (e.g., pathway, transcription, ontology)",
                "name": "database",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of background genes to use for enrichment analysis",
                "name": "background_list",
                "type": "list",
            },
            {
                "default": False,
                "description": "Generate a bar plot of the top K enrichment results",
                "name": "plot",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of gene symbols to analyze",
                "name": "genes",
                "type": "list",
            }
        ],
    },
    {
        "description": "Analyze chromatin interactions from Hi-C data to identify "
        "enhancer-promoter interactions and TADs.",
        "name": "analyze_chromatin_interactions",
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
                "description": "Path to the Hi-C data file (.cool or .hic format)",
                "name": "hic_file_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to BED file containing genomic "
                "coordinates of regulatory elements "
                "(enhancers, promoters, CTCF sites, "
                "etc.)",
                "name": "regulatory_elements_bed",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform comparative genomics and haplotype analysis on "
        "multiple genome samples. Aligns genome samples to a "
        "reference, identifies variants, analyzes shared and unique "
        "genomic regions, and determines haplotype structure.",
        "name": "analyze_comparative_genomics_and_haplotypes",
        "optional_parameters": [
            {
                "default": "./output",
                "description": "Directory to store output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Paths to FASTA files containing whole-genome sequences to be analyzed",
                "name": "sample_fasta_files",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Path to the reference genome FASTA file",
                "name": "reference_genome_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform ChIP-seq peak calling using MACS2 to identify "
        "genomic regions with significant binding.",
        "name": "perform_chipseq_peak_calling_with_macs2",
        "optional_parameters": [
            {
                "default": "macs2_output",
                "description": "Prefix for output files",
                "name": "output_name",
                "type": "str",
            },
            {
                "default": "hs",
                "description": "Effective genome size shorthand: 'hs' for human, 'mm' for mouse, etc.",
                "name": "genome_size",
                "type": "str",
            },
            {
                "default": 0.05,
                "description": "q-value (minimum FDR) cutoff for peak calling",
                "name": "q_value",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the ChIP-seq read data file (BAM, BED, or other supported format)",
                "name": "chip_seq_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the control/input data file (BAM, BED, or other supported format)",
                "name": "control_file",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find DNA sequence motifs enriched in genomic regions using the HOMER motif discovery software.",
        "name": "find_enriched_motifs_with_homer",
        "optional_parameters": [
            {
                "default": "hg38",
                "description": "Reference genome for sequence extraction",
                "name": "genome",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to BED file with background "
                "regions for comparison. If None, "
                "HOMER will generate random "
                "background sequences automatically",
                "name": "background_file",
                "type": "str",
            },
            {
                "default": "8,10,12",
                "description": "Comma-separated list of motif lengths to discover",
                "name": "motif_length",
                "type": "str",
            },
            {
                "default": "./homer_motifs",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 10,
                "description": "Number of motifs to find",
                "name": "num_motifs",
                "type": "int",
            },
            {
                "default": 4,
                "description": "Number of CPU threads to use",
                "name": "threads",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to peak file in BED format "
                "containing genomic regions to "
                "analyze for motif enrichment",
                "name": "peak_file",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze overlaps between two or more sets of genomic regions.",
        "name": "analyze_genomic_region_overlap",
        "optional_parameters": [
            {
                "default": "overlap_analysis",
                "description": "Prefix for output files",
                "name": "output_prefix",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of genomic region sets. Each "
                "item can be either a string path to "
                "a BED file or a list of "
                "tuples/lists with format (chrom, "
                "start, end) or (chrom, start, end, "
                "name)",
                "name": "region_sets",
                "type": "list",
            }
        ],
    },
    {
        "description": "Convert ENSEMBL gene IDs between different species using BioMart homology mapping. "
        "This function converts a list of ENSEMBL gene IDs from one species to their "
        "homologous counterparts in another species using the Ensembl BioMart database. "
        "The conversion is based on one-to-one ortholog mappings between species.",
        "name": "interspecies_gene_conversion",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "List of ENSEMBL gene IDs to convert (e.g., ['ENSG00000007372', 'ENSG00000181449'])",
                "name": "gene_list",
                "type": "list[str]",
            },
            {
                "default": None,
                "description": "Source species name. Supported species: human, mouse, rat, zebrafish, fly, "
                "drosophila, worm, yeast, chicken, pig, cow, dog, macaque",
                "name": "source_species",
                "type": "str",
            },
            {
                "default": None,
                "description": "Target species name. Same supported species as source_species",
                "name": "target_species",
                "type": "str",
            },
        ],
    },
]
