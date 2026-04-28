# Data lake dictionary with detailed descriptions (Commercial Mode - Drug Discovery focused)
data_lake_dict = {
    # === 药物发现核心数据 ===
    # "BindingDB_All_202409.tsv": "Measured binding affinities between proteins and small molecules for drug discovery.",  # Requires commercial license
    "broad_repurposing_hub_molecule_with_smiles.parquet": "Molecules from Broad Institute's Drug Repurposing Hub with SMILES annotations.",
    "broad_repurposing_hub_phase_moa_target_info.parquet": "Drug phases, mechanisms of action, and target information from Broad Institute.",
    # "enamine_cloud_library_smiles.pkl": "Compounds from Enamine REAL library with SMILES annotations.",  # Proprietary - Requires license
    # === 药物相互作用（CC BY-NC-SA 4.0，商业模式下不可用）===
    # "ddinter_alimentary_tract_metabolism.csv": "Drug-drug interactions for alimentary tract and metabolism drugs from DDInter 2.0 database.",
    # "ddinter_antineoplastic.csv": "Drug-drug interactions for antineoplastic and immunomodulating agents from DDInter 2.0 database.",
    # "ddinter_antiparasitic.csv": "Drug-drug interactions for antiparasitic products from DDInter 2.0 database.",
    # "ddinter_blood_organs.csv": "Drug-drug interactions for blood and blood forming organs drugs from DDInter 2.0 database.",
    # "ddinter_dermatological.csv": "Drug-drug interactions for dermatological drugs from DDInter 2.0 database.",
    # "ddinter_hormonal.csv": "Drug-drug interactions for systemic hormonal preparations from DDInter 2.0 database.",
    # "ddinter_respiratory.csv": "Drug-drug interactions for respiratory system drugs from DDInter 2.0 database.",
    # "ddinter_various.csv": "Drug-drug interactions for various drugs from DDInter 2.0 database.",
    # === EveBio 筛选数据（Proprietary，商业模式下不可用）===
    # "evebio_assay_table.csv": "Assay metadata with one row per assay from EveBio pharmome mapping.",
    # "evebio_bundle_table.csv": "Target subfamily bundles used for screening-to-profiling progression.",
    # "evebio_compound_table.csv": "Compound metadata with common identifiers from EveBio screening.",
    # "evebio_control_table.csv": "Control datapoints for all screening and profiling plates.",
    # "evebio_detailed_result_table.csv": "Expanded results on evebio_summary_result_table with curve fit parameters and phase categories.",
    # "evebio_observed_points_table.csv": "Raw observed datapoints from all screening and profiling experiments.",
    # "evebio_summary_result_table.csv": "Succinct summary of results for each assay-compound combination.",
    # "evebio_target_table.csv": "Target metadata with common identifiers from EveBio screening.",
    # === 蛋白质与靶点 ===
    "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
    "gene_info.parquet": "Comprehensive gene information.",
    # "omim.parquet": "Genetic disorders and associated genes from OMIM.",  # Requires commercial license
    # "DisGeNET.parquet": "Gene-disease associations from multiple sources.",  # CC BY-NC-SA 4.0
    # === 蛋白质相互作用（靶点网络分析）===
    "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
    "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
    "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
    "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
    # === 疾病与表型 ===
    "hp.obo": "Official HPO release in obographs format.",
    "kg.csv": "Precision medicine knowledge graph with 17,080 diseases and 4+ million relationships across biological scales.",
    "txgnn_name_mapping.pkl": "Name mapping for TXGNN.",
    "txgnn_prediction.pkl": "Prediction data for TXGNN.",
    # === 肿瘤药物敏感性（抗肿瘤药物研发）===
    "DepMap_CRISPRGeneDependency.csv": "Gene dependency probability estimates for cancer cell lines, including all DepMap models.",
    "DepMap_CRISPRGeneEffect.csv": "Genome-wide CRISPR gene effect estimates for cancer cell lines, including all DepMap models.",
    "DepMap_Model.csv": "Metadata describing all cancer models/cell lines which are referenced by a dataset contained within the DepMap portal.",
    "DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv": "Gene expression in TPMs for cancer cell lines, including all DepMap models.",
}

# Library content dictionary (Commercial Mode - Drug Discovery focused)
library_content_dict = {
    # === 化学信息学与药物发现 ===
    "rdkit": "[Python Package] A collection of cheminformatics and machine learning tools for working with chemical structures and drug discovery.",
    "openbabel": "[Python Package] A chemical toolbox designed to speak the many languages of chemical data, supporting file format conversion and molecular modeling.",
    "descriptastorus": "[Python Package] A library for computing molecular descriptors for machine learning applications in drug discovery.",
    "deeppurpose": "[Python Package] A deep learning library for drug-target interaction prediction and virtual screening.",
    "pyscreener": "[Python Package] A Python package for virtual screening of chemical compounds.",
    "pytdc": "[Python Package] A Python package for Therapeutics Data Commons, providing access to machine learning datasets for drug discovery.",
    "openmm": "[Python Package] A toolkit for molecular simulation using high-performance GPU computing.",
    "biopandas": "[Python Package] A package that provides pandas DataFrames for working with molecular structures and biological data.",
    "pdbfixer": "[Python Package] A Python package for fixing problems in PDB files in preparation for molecular simulations.",
    "biotite": "[Python Package] A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
    # === 分子对接 CLI 工具 ===
    "vina": "[CLI Tool] An open-source program for molecular docking and virtual screening, known for its speed and accuracy improvements over AutoDock 4.",
    "autosite": "[CLI Tool] A binding site detection tool used to identify potential ligand binding pockets on protein structures for molecular docking.",
    "ADFR": "AutoDock for Receptors suite for molecular docking and virtual screening.",
    # === 数据科学基础 ===
    "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
    "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
    "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
    "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
    "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
    "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
    "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
    "umap-learn": "[Python Package] Uniform Manifold Approximation and Projection, a dimension reduction technique.",
    # === 文献与数据库检索 ===
    "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
    "pymed": "[Python Package] A Python library for accessing PubMed articles.",
    "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
    "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
    "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
    "requests": "[Python Package] A simple HTTP library for Python, useful for accessing web APIs and databases.",
    # === 通用工具 ===
    "tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
    "joblib": "[Python Package] A set of tools to provide lightweight pipelining in Python, including transparent disk-caching and parallel computing.",
    "h5py": "[Python Package] A Python interface to the HDF5 binary data format, allowing storage of large amounts of numerical data.",
    "reportlab": "[Python Package] Creation of PDF documents.",
}
