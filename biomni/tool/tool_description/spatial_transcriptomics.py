description = [
    {
        "description": "Load 10x Visium data into a standardized AnnData object. Supports .h5ad files, 10x matrix directories, and wide counts CSV (rows=barcodes, cols=genes). Output has X=raw counts, obsm['spatial']=coordinates, obs['barcode'], var['gene_symbols'], uns['spatial_meta'].",
        "name": "load_visium_data",
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional path to coordinate CSV (barcode + x/y or Space Ranger "
                "tissue_positions format). If the h5ad already has obsm['spatial'], not needed.",
                "name": "coordinates_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Optional sample identifier stored in adata.obs['sample_id']",
                "name": "sample_id",
                "type": "str",
            },
            {
                "default": "./visium_loaded.h5ad",
                "description": "Path where the loaded AnnData will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input data: .h5ad file, 10x matrix directory, "
                "or wide counts CSV. Identify the format first (see description).",
                "name": "counts_file",
                "type": "str",
            },
        ],
    },
    {
        "description": "Filter low-quality spots in 10x Visium data. Removes spots with too few counts, too few detected genes, or high mitochondrial fraction. Adds QC metrics (n_genes_by_counts, total_counts, pct_counts_mt) to adata.obs.",
        "name": "filter_visium_spots",
        "optional_parameters": [
            {
                "default": "./visium_filtered.h5ad",
                "description": "Path where the filtered AnnData will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 200,
                "description": "Minimum number of total counts per spot",
                "name": "min_counts",
                "type": "int",
            },
            {
                "default": 20,
                "description": "Minimum number of detected genes per spot",
                "name": "min_genes",
                "type": "int",
            },
            {
                "default": 20.0,
                "description": "Maximum allowed percentage of mitochondrial counts",
                "name": "pct_mt",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input AnnData (h5ad) file",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Normalize and log-transform 10x Visium data. Applies the standard scanpy pipeline: normalize_total + log1p, optionally with highly variable gene selection. Raw counts are stored in adata.raw for downstream deconvolution.",
        "name": "normalize_visium",
        "optional_parameters": [
            {
                "default": "./visium_normalized.h5ad",
                "description": "Path where the normalized AnnData will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 10000.0,
                "description": "Target library size for normalize_total",
                "name": "target_sum",
                "type": "float",
            },
            {
                "default": 2000,
                "description": "Number of highly variable genes to select (0 to skip)",
                "name": "n_top_genes",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input AnnData (h5ad) file",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Cluster 10x Visium spots using the standard PCA + UMAP + Leiden pipeline. Saves cluster assignments in obs['leiden'] and the UMAP embedding in obsm['X_umap'].",
        "name": "cluster_spatial_data",
        "optional_parameters": [
            {
                "default": "./visium_clustered.h5ad",
                "description": "Path where the clustered AnnData will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 30,
                "description": "Number of principal components to use",
                "name": "n_pcs",
                "type": "int",
            },
            {
                "default": 1.0,
                "description": "Leiden clustering resolution",
                "name": "resolution",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the normalized AnnData (h5ad) file",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Identify spatial domains in 10x Visium data. Builds a spatial neighbor graph and applies Leiden clustering to detect spatially contiguous domains.",
        "name": "identify_spatial_domains",
        "optional_parameters": [
            {
                "default": "./visium_domains.h5ad",
                "description": "Path where the domain-annotated AnnData will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 1.0,
                "description": "Leiden clustering resolution for domain detection",
                "name": "resolution",
                "type": "float",
            },
            {
                "default": 6,
                "description": "Number of spatial neighbors per spot",
                "name": "n_neighs",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the normalized AnnData (h5ad) file with spatial coordinates",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Detect spatially variable genes (SVGs) in 10x Visium data using Moran's I autocorrelation and returns a ranked gene list.",
        "name": "find_spatially_variable_genes",
        "optional_parameters": [
            {
                "default": "./svg_results.csv",
                "description": "Path where the SVG ranking CSV will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 100,
                "description": "Number of top spatially variable genes to report",
                "name": "n_top_genes",
                "type": "int",
            },
            {
                "default": 1,
                "description": "Number of parallel jobs for autocorrelation computation",
                "name": "n_jobs",
                "type": "int",
            },
            {
                "default": None,
                "description": "Optional list of gene names to plot on spatial coordinates",
                "name": "genes_to_plot",
                "type": "list",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the normalized AnnData (h5ad) file with spatial coordinates",
                "name": "adata_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Deconvolve 10x Visium spots into cell-type proportions using SPOTlight with a reference scRNA-seq dataset.",
        "name": "deconvolve_spatial_spotlight",
        "optional_parameters": [
            {
                "default": "./spotlight_results",
                "description": "Directory where SPOTlight results (proportions CSV) are saved",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": "Rscript",
                "description": "Path to the Rscript executable",
                "name": "r_script_path",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference scRNA-seq AnnData (h5ad) with cell-type labels",
                "name": "ref_h5ad",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the Visium AnnData (h5ad) to deconvolve",
                "name": "st_h5ad",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column in ref_h5ad.obs containing cell-type labels",
                "name": "cell_type_key",
                "type": "str",
            },
        ],
    },
    {
        "description": "Deconvolve 10x Visium spots into cell-type proportions using DestVI (CondSCVI + DestVI) with a reference scRNA-seq dataset.",
        "name": "deconvolve_spatial_destvi",
        "optional_parameters": [
            {
                "default": "./destvi_results",
                "description": "Directory where DestVI results (proportions CSV) are saved",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 30,
                "description": "Latent dimensionality for CondSCVI",
                "name": "n_latent",
                "type": "int",
            },
            {
                "default": 400,
                "description": "Number of training epochs for the CondSCVI reference model",
                "name": "max_epochs",
                "type": "int",
            },
            {
                "default": 500,
                "description": "Number of training epochs for the DestVI spatial model",
                "name": "destvi_max_epochs",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference scRNA-seq AnnData (h5ad) with cell-type labels",
                "name": "ref_h5ad",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the Visium AnnData (h5ad) to deconvolve",
                "name": "st_h5ad",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column in ref_h5ad.obs containing cell-type labels",
                "name": "cell_type_key",
                "type": "str",
            },
        ],
    },
    {
        "description": "Infer spatially proximal cell-cell communication using CellChat v2 with the CellChatDB ligand-receptor database.",
        "name": "infer_spatial_cell_communication",
        "optional_parameters": [
            {
                "default": "./cellchat_results",
                "description": "Directory where CellChat results are saved",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": "Rscript",
                "description": "Path to the Rscript executable",
                "name": "r_script_path",
                "type": "str",
            },
            {
                "default": "human",
                "description": "Species for CellChatDB ('human' or 'mouse')",
                "name": "species",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the Visium AnnData (h5ad) with cell-type labels "
                "(e.g. from deconvolution) and spatial coordinates",
                "name": "st_h5ad",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column in st_h5ad.obs containing cell-type labels",
                "name": "cell_type_key",
                "type": "str",
            },
        ],
    },
    {
        "description": "Compute cell-type neighborhood enrichment in 10x Visium data using Squidpy, returning z-scores for co-localization and exclusion between cell types.",
        "name": "spatial_neighborhood_enrichment",
        "optional_parameters": [
            {
                "default": "./neighborhood_enrichment.csv",
                "description": "Path where the enrichment CSV will be saved",
                "name": "output_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path where the plot (PNG) will be saved",
                "name": "plot_path",
                "type": "str",
            },
            {
                "default": 6,
                "description": "Number of spatial neighbors per spot",
                "name": "n_neighs",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the AnnData (h5ad) with spatial coordinates and cell-type labels",
                "name": "adata_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Column in adata.obs containing cell-type labels",
                "name": "cell_type_key",
                "type": "str",
            },
        ],
    },
    {
        "description": "Create a standardized spatial transcriptomics project directory layout with results subdirectories for each analysis stage.",
        "name": "init_spatial_project",
        "optional_parameters": [
            {
                "default": False,
                "description": "If True, recreate the directory even if it already exists",
                "name": "overwrite",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Root path of the project directory to scaffold",
                "name": "project_dir",
                "type": "str",
            },
        ],
    },
]
