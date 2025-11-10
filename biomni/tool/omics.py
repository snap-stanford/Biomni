"""
Universal Omics Data Analysis Module

This module provides universal functions applicable to all types of omics data
(transcriptomics, proteomics, metabolomics, etc.) including normalization,
imputation, filtering, statistical analysis, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Dict
from scipy import stats

def convert_gene_ids(
    gene_ids: list,
    from_type: str,
    to_type: str,
    organism: str = "human",
) -> dict:
    """
    Convert between different gene ID formats.
    
    Essential for integrating data from different sources that use different
    gene identifier systems (Ensembl, Entrez, Gene Symbol, RefSeq, etc.).
    
    Parameters
    ----------
    gene_ids : list
        List of gene identifiers to convert
    from_type : str
        Source identifier type. Options:
        - "ensembl": Ensembl gene IDs (e.g., "ENSG00000141510")
        - "entrez": NCBI Entrez gene IDs (e.g., "7157")
        - "symbol": Gene symbols (e.g., "TP53")
        - "refseq": RefSeq IDs (e.g., "NM_000546")
    to_type : str
        Target identifier type (same options as from_type)
    organism : str, optional (default: "human")
        Organism name. Options:
        - "human": Homo sapiens
        - "mouse": Mus musculus
        - "rat": Rattus norvegicus
    
    Returns
    -------
    dict
        Dictionary mapping input IDs to converted IDs.
        Format: {input_id: converted_id or None if not found}
        
    Examples
    --------
    >>> # Convert Ensembl IDs to gene symbols
    >>> ensembl_ids = ["ENSG00000141510", "ENSG00000157764"]
    >>> result = convert_gene_ids(ensembl_ids, "ensembl", "symbol")
    >>> print(result)
    {'ENSG00000141510': 'TP53', 'ENSG00000157764': 'BRAF'}
    
    >>> # Convert gene symbols to Entrez IDs
    >>> symbols = ["TP53", "BRCA1", "EGFR"]
    >>> result = convert_gene_ids(symbols, "symbol", "entrez")
    >>> print(result)
    {'TP53': '7157', 'BRCA1': '672', 'EGFR': '1956'}
    
    >>> # Convert for mouse genes
    >>> mouse_symbols = ["Tp53", "Brca1"]
    >>> result = convert_gene_ids(mouse_symbols, "symbol", "ensembl", organism="mouse")
    
    Notes
    -----
    - Uses gget library for gene ID conversion via Ensembl database
    - Some IDs may not have direct mappings and will return None
    - Case-sensitive for gene symbols in most databases
    - For large lists (>1000 genes), consider batching
    - Requires internet connection for database queries
    """
    import gget
    
    # Validate inputs
    valid_types = ["ensembl", "entrez", "symbol", "refseq"]
    if from_type not in valid_types:
        raise ValueError(f"from_type must be one of {valid_types}, got '{from_type}'")
    if to_type not in valid_types:
        raise ValueError(f"to_type must be one of {valid_types}, got '{to_type}'")
    
    # Map organism names to gget species codes
    organism_map = {
        "human": "homo_sapiens",
        "mouse": "mus_musculus",
        "rat": "rattus_norvegicus",
    }
    
    if organism.lower() not in organism_map:
        raise ValueError(f"organism must be one of {list(organism_map.keys())}, got '{organism}'")
    
    species = organism_map[organism.lower()]
    
    # Initialize result dictionary
    result = {}
    
    # Convert each gene ID
    for gene_id in gene_ids:
        try:
            # Use gget info to get gene information
            gene_info = gget.info(gene_id, species=species, verbose=False)
            
            if gene_info is None or len(gene_info) == 0:
                result[gene_id] = None
                continue
            
            # Extract the target ID type from the result
            if to_type == "ensembl":
                result[gene_id] = gene_info.iloc[0]["ensembl_id"]
            elif to_type == "entrez":
                entrez_id = gene_info.iloc[0].get("ncbi_gene_id", None)
                result[gene_id] = str(int(entrez_id)) if pd.notna(entrez_id) else None
            elif to_type == "symbol":
                result[gene_id] = gene_info.iloc[0].get("gene_name", None)
            elif to_type == "refseq":
                # Get RefSeq from transcript information
                refseq_ids = gene_info.iloc[0].get("refseq_id", None)
                if pd.notna(refseq_ids):
                    # Take first RefSeq ID if multiple exist
                    result[gene_id] = refseq_ids.split(";")[0] if isinstance(refseq_ids, str) else refseq_ids
                else:
                    result[gene_id] = None
            
        except Exception as e:
            # If conversion fails, return None for that ID
            result[gene_id] = None
    
    return result

def normalize_data(
    data: pd.DataFrame,
    sample_columns: list,
    method: str = "median",
    log_transform: bool = True,
    log_base: float = 2.0,
    **kwargs
) -> pd.DataFrame:
    """
    Universal normalization function for all omics data types.
    
    Applicable to proteomics, metabolomics, RNA-seq (non-count), and other
    continuous omics data. Removes systematic technical variations.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with features as rows and samples as columns
    sample_columns : list
        List of column names to normalize
    method : str, optional (default: "median")
        Normalization method:
        - "median": Median normalization
        - "quantile": Quantile normalization
        - "zscore": Z-score normalization
        - "total": Total intensity normalization
        - "pqn": Probabilistic Quotient Normalization
    log_transform : bool, optional (default: True)
        Whether to apply log transformation before normalization
    log_base : float, optional (default: 2.0)
        Base for log transformation
    **kwargs : dict
        Additional parameters:
        - reference_sample: For PQN (str)
        - gene_lengths: For TPM/FPKM (pd.Series)
    
    Returns
    -------
    pd.DataFrame
        Normalized data
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     's1': [100, 200, 50],
    ...     's2': [150, 250, 60]
    ... })
    >>> normalized = normalize_data(data, ['s1', 's2'], method='median')
    
    Notes
    -----
    - Choose method based on data characteristics
    - Log transformation stabilizes variance
    - PQN recommended for metabolomics
    - Median/quantile for proteomics
    """
    result = data.copy()
    data_matrix = data[sample_columns].values.astype(float)
    
    # Log transformation
    if log_transform:
        data_matrix = np.log(data_matrix + 1) / np.log(log_base)
    
    # Normalization
    if method == "median":
        medians = np.nanmedian(data_matrix, axis=0)
        global_median = np.nanmean(medians)
        normalized = data_matrix * (global_median / medians)
        
    elif method == "quantile":
        from scipy.interpolate import interp1d
        
        sorted_data = np.sort(data_matrix, axis=0)
        row_means = np.nanmean(sorted_data, axis=1)
        ranks = data_matrix.argsort(axis=0).argsort(axis=0)
        
        normalized = np.zeros_like(data_matrix)
        for j in range(data_matrix.shape[1]):
            normalized[:, j] = row_means[ranks[:, j]]
    
    elif method == "zscore":
        means = np.nanmean(data_matrix, axis=0)
        stds = np.nanstd(data_matrix, axis=0)
        stds[stds == 0] = 1
        normalized = (data_matrix - means) / stds
        
    elif method == "total":
        totals = np.nansum(data_matrix, axis=0)
        global_total = np.nanmean(totals)
        normalized = data_matrix * (global_total / totals)
        
    elif method == "pqn":
        # Probabilistic Quotient Normalization
        totals = np.nansum(data_matrix, axis=0)
        normalized_integral = data_matrix / totals * np.median(totals)
        
        reference_sample = kwargs.get('reference_sample', None)
        if reference_sample and reference_sample in sample_columns:
            ref_idx = sample_columns.index(reference_sample)
            ref_spectrum = normalized_integral[:, ref_idx]
        else:
            ref_spectrum = np.nanmedian(normalized_integral, axis=1)
        
        quotients = normalized_integral / ref_spectrum[:, np.newaxis]
        median_quotients = np.nanmedian(quotients, axis=0)
        normalized = data_matrix / median_quotients
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result[sample_columns] = normalized
    return result


def impute_missing_values(
    data: pd.DataFrame,
    sample_columns: list,
    method: str = "knn",
    **kwargs
) -> pd.DataFrame:
    """
    Universal missing value imputation for omics data.
    
    Handles missing values in any type of omics data. Essential preprocessing
    step for many downstream analyses that require complete data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with features as rows and samples as columns
    sample_columns : list
        Column names to impute
    method : str, optional (default: "knn")
        Imputation method:
        - "knn": K-nearest neighbors
        - "mean": Mean imputation
        - "median": Median imputation
        - "zero": Replace with zero
        - "min": Replace with minimum value
        - "minprob": Minimum probability (for MNAR data)
    **kwargs : dict
        Method-specific parameters:
        - n_neighbors: For KNN (default: 5)
        - downshift: For minprob (default: 1.8)
        - width: For minprob (default: 0.3)
    
    Returns
    -------
    pd.DataFrame
        Data with imputed values
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     's1': [1.0, np.nan, 3.0],
    ...     's2': [2.0, 2.5, np.nan]
    ... })
    >>> imputed = impute_missing_values(data, ['s1', 's2'], method='knn')
    
    Notes
    -----
    - KNN works well for MCAR (missing completely at random)
    - MinProb designed for MNAR (missing not at random) in proteomics
    - Choose method based on missingness mechanism
    """
    result = data.copy()
    data_matrix = data[sample_columns].values.astype(float)
    
    if method == "knn":
        from sklearn.impute import KNNImputer
        n_neighbors = kwargs.get('n_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed = imputer.fit_transform(data_matrix)
        
    elif method == "mean":
        means = np.nanmean(data_matrix, axis=1, keepdims=True)
        imputed = data_matrix.copy()
        mask = np.isnan(imputed)
        imputed[mask] = np.repeat(means, data_matrix.shape[1], axis=1)[mask]
        
    elif method == "median":
        medians = np.nanmedian(data_matrix, axis=1, keepdims=True)
        imputed = data_matrix.copy()
        mask = np.isnan(imputed)
        imputed[mask] = np.repeat(medians, data_matrix.shape[1], axis=1)[mask]
        
    elif method == "zero":
        imputed = np.nan_to_num(data_matrix, nan=0.0)
        
    elif method == "min":
        min_val = np.nanmin(data_matrix)
        imputed = np.nan_to_num(data_matrix, nan=min_val)
        
    elif method == "minprob":
        # Minimum probability imputation (MNAR-specific)
        downshift = kwargs.get('downshift', 1.8)
        width = kwargs.get('width', 0.3)
        
        imputed = data_matrix.copy()
        for i in range(data_matrix.shape[0]):
            row = data_matrix[i, :]
            if np.isnan(row).any():
                valid_data = row[~np.isnan(row)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    
                    min_val = mean_val - downshift * std_val
                    
                    nan_mask = np.isnan(row)
                    n_missing = nan_mask.sum()
                    
                    random_values = np.random.normal(
                        min_val, width * std_val, n_missing
                    )
                    imputed[i, nan_mask] = random_values
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result[sample_columns] = imputed
    return result


def filter_low_values(
    data: pd.DataFrame,
    sample_columns: list,
    min_value: float = 10,
    min_samples: int = None,
    method: str = "threshold"
) -> pd.DataFrame:
    """
    Universal low-value filtering for omics data.
    
    Removes features with low detection/expression to reduce noise and
    improve statistical power in downstream analyses.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows, samples as columns
    sample_columns : list
        Sample column names
    min_value : float, optional (default: 10)
        Minimum value threshold
    min_samples : int, optional
        Minimum number of samples meeting threshold.
        If None, uses half of samples.
    method : str, optional (default: "threshold")
        Filtering method:
        - "threshold": Count-based threshold
        - "cv": Coefficient of variation filter
        - "iqr": Interquartile range filter
    
    Returns
    -------
    pd.DataFrame
        Filtered data
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     's1': [100, 5, 200],
    ...     's2': [120, 3, 180]
    ... })
    >>> filtered = filter_low_values(data, ['s1', 's2'], min_value=10)
    
    Notes
    -----
    - Filtering should be done before normalization
    - Threshold depends on data type and scale
    """
    if min_samples is None:
        min_samples = len(sample_columns) // 2
    
    data_matrix = data[sample_columns].values
    
    if method == "threshold":
        samples_above = (data_matrix >= min_value).sum(axis=1)
        keep = samples_above >= min_samples
        
    elif method == "cv":
        means = np.nanmean(data_matrix, axis=1)
        stds = np.nanstd(data_matrix, axis=1)
        cv = stds / (means + 1e-10)
        keep = cv < min_value  # min_value acts as CV threshold
        
    elif method == "iqr":
        q75 = np.nanpercentile(data_matrix, 75, axis=1)
        q25 = np.nanpercentile(data_matrix, 25, axis=1)
        iqr = q75 - q25
        keep = iqr > min_value  # min_value acts as IQR threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    filtered = data.loc[keep]
    
    n_removed = len(data) - len(filtered)
    print(f"Filtered {n_removed} features. Kept {len(filtered)} features.")
    
    return filtered


def calculate_fold_changes(
    data: pd.DataFrame,
    group1_samples: list,
    group2_samples: list,
    log_transform: bool = True,
    adjust_pvalues: bool = True,
    test_method: str = "ttest"
) -> pd.DataFrame:
    """
    Universal fold change calculation for omics data.
    
    Calculates fold changes and statistical significance between two groups.
    Applicable to any omics data type.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized data (features × samples)
    group1_samples : list
        Sample names for group 1
    group2_samples : list
        Sample names for group 2
    log_transform : bool, optional (default: True)
        Return log2 fold changes
    adjust_pvalues : bool, optional (default: True)
        Apply FDR correction
    test_method : str, optional (default: "ttest")
        Statistical test: "ttest", "mannwhitney"
    
    Returns
    -------
    pd.DataFrame
        Results with fold changes, p-values, and significance
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     't1': [10, 20], 't2': [12, 22],
    ...     'c1': [5, 19], 'c2': [6, 21]
    ... })
    >>> fc_results = calculate_fold_changes(
    ...     data, ['t1', 't2'], ['c1', 'c2']
    ... )
    
    Notes
    -----
    - Data should be normalized before FC calculation
    - Log2FC > 0: higher in group1
    - Log2FC < 0: higher in group2
    """
    group1_data = data[group1_samples].values
    group2_data = data[group2_samples].values
    
    # Calculate means
    mean1 = np.nanmean(group1_data, axis=1)
    mean2 = np.nanmean(group2_data, axis=1)
    
    # Calculate fold change
    fc = mean1 / (mean2 + 1e-10)
    if log_transform:
        fc = np.log2(fc + 1e-10)
    
    # Perform statistical test
    if test_method == "ttest":
        t_stats, p_vals = stats.ttest_ind(group1_data, group2_data, axis=1)
    elif test_method == "mannwhitney":
        p_vals = []
        t_stats = []
        for i in range(group1_data.shape[0]):
            stat, p = stats.mannwhitneyu(
                group1_data[i, :], group2_data[i, :],
                alternative='two-sided'
            )
            t_stats.append(stat)
            p_vals.append(p)
        p_vals = np.array(p_vals)
        t_stats = np.array(t_stats)
    else:
        raise ValueError(f"Unknown test method: {test_method}")
    
    # Create results
    results = pd.DataFrame({
        'mean_group1': mean1,
        'mean_group2': mean2,
        'fold_change': fc,
        'p_value': p_vals,
        'statistic': t_stats
    }, index=data.index)
    
    # FDR correction
    if adjust_pvalues:
        valid_pvals = results['p_value'].dropna()
        if len(valid_pvals) > 0:
            n = len(valid_pvals)
            sorted_idx = np.argsort(valid_pvals.values)
            sorted_pvals = valid_pvals.values[sorted_idx]
            
            padj = np.zeros(n)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    padj[sorted_idx[i]] = min(sorted_pvals[i], 1.0)
                else:
                    padj[sorted_idx[i]] = min(
                        sorted_pvals[i] * n / (i+1),
                        padj[sorted_idx[i+1]],
                        1.0
                    )
            
            results['p_adj'] = np.nan
            results.loc[valid_pvals.index, 'p_adj'] = padj
        else:
            results['p_adj'] = np.nan
    
    results['significant'] = results.get('p_adj', results['p_value']) < 0.05
    
    return results.sort_values('p_value')


def draw_pca(
    data: pd.DataFrame,
    sample_columns: list,
    metadata: pd.DataFrame = None,
    color_by: str = None,
    n_components: int = 2,
    show_loadings: bool = False,
    n_loadings: int = 10,
    output_file: str = "pca_plot.png",
) -> str:
    """
    Universal PCA visualization for omics data.
    
    Creates PCA plot to visualize sample relationships and identify patterns
    or batch effects. Applicable to all omics data types.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows, samples as columns
    sample_columns : list
        Sample names to include
    metadata : pd.DataFrame, optional
        Sample metadata for coloring
    color_by : str, optional
        Metadata column for coloring
    n_components : int, optional (default: 2)
        Number of PCs to compute
    show_loadings : bool, optional (default: False)
        Show feature loading vectors
    n_loadings : int, optional (default: 10)
        Number of loadings to show
    output_file : str, optional
        Output file path
    
    Returns
    -------
    str
        Path to saved plot
    
    Examples
    --------
    >>> data = pd.DataFrame(np.random.randn(100, 6))
    >>> plot_path = draw_pca(data, data.columns.tolist())
    
    Notes
    -----
    - Normalize and clean data before PCA
    - Missing values not allowed
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Extract data
    data_matrix = data[sample_columns].values.T  # Transpose: samples x features
    
    # Remove missing values
    valid_features = ~np.any(np.isnan(data_matrix), axis=0)
    data_clean = data_matrix[:, valid_features]
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by metadata if provided
    if metadata is not None and color_by is not None:
        groups = metadata.loc[sample_columns, color_by]
        unique_groups = groups.unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[colors[i]],
                label=group,
                s=100,
                alpha=0.7,
                edgecolors='black'
            )
        ax.legend()
    else:
        ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
    
    # Add sample labels
    for i, sample in enumerate(sample_columns):
        ax.annotate(
            sample,
            (pca_result[i, 0], pca_result[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Labels
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=12)
    ax.set_title('PCA - Sample Overview', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def draw_volcano_plot(
    fc_results: pd.DataFrame,
    fc_column: str = "fold_change",
    pval_column: str = "p_adj",
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    label_top: int = 10,
    output_file: str = "volcano_plot.png"
) -> str:
    """
    Universal volcano plot for omics data.
    
    Visualizes fold changes vs statistical significance. Shows which features
    are significantly changed between conditions.
    
    Parameters
    ----------
    fc_results : pd.DataFrame
        Results from calculate_fold_changes()
    fc_column : str, optional
        Column name for fold change values
    pval_column : str, optional
        Column name for p-values
    fc_threshold : float, optional (default: 1.0)
        Log2 fold change threshold for significance
    pval_threshold : float, optional (default: 0.05)
        P-value threshold
    label_top : int, optional (default: 10)
        Number of top features to label
    output_file : str, optional
        Output file path
    
    Returns
    -------
    str
        Path to saved plot
    
    Examples
    --------
    >>> fc_results = calculate_fold_changes(data, group1, group2)
    >>> plot_path = draw_volcano_plot(fc_results)
    
    Notes
    -----
    - Points in upper corners are significant
    - Red: upregulated, Blue: downregulated
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fc = fc_results[fc_column].values
    pval = fc_results[pval_column].values
    
    # -log10 transform p-values
    neg_log_pval = -np.log10(pval + 1e-300)
    
    # Classify points
    sig_up = (fc > fc_threshold) & (pval < pval_threshold)
    sig_down = (fc < -fc_threshold) & (pval < pval_threshold)
    not_sig = ~(sig_up | sig_down)
    
    # Plot
    ax.scatter(fc[not_sig], neg_log_pval[not_sig],
               c='gray', alpha=0.5, s=20, label='Not significant')
    ax.scatter(fc[sig_up], neg_log_pval[sig_up],
               c='red', alpha=0.7, s=30, label=f'Up (n={sig_up.sum()})')
    ax.scatter(fc[sig_down], neg_log_pval[sig_down],
               c='blue', alpha=0.7, s=30, label=f'Down (n={sig_down.sum()})')
    
    # Threshold lines
    ax.axhline(-np.log10(pval_threshold), color='black',
               linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(fc_threshold, color='black',
               linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-fc_threshold, color='black',
               linestyle='--', linewidth=1, alpha=0.5)
    
    # Label top features
    if label_top > 0:
        sig_features = fc_results[sig_up | sig_down].copy()
        sig_features = sig_features.nsmallest(label_top, pval_column)
        
        for idx in sig_features.index:
            x = fc_results.loc[idx, fc_column]
            y = -np.log10(fc_results.loc[idx, pval_column] + 1e-300)
            ax.annotate(idx, (x, y), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-Log10 P-value', fontsize=12)
    ax.set_title('Volcano Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def draw_heatmap(
    data: pd.DataFrame,
    sample_columns: list,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    scale: str = "row",
    cmap: str = "RdBu_r",
    figsize: tuple = (10, 12),
    output_file: str = "heatmap.png"
) -> str:
    """
    Universal heatmap for omics data.
    
    Visualizes expression patterns across samples and features with
    hierarchical clustering.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot (features × samples)
    sample_columns : list
        Sample columns to include
    cluster_rows : bool, optional (default: True)
        Cluster features
    cluster_cols : bool, optional (default: True)
        Cluster samples
    scale : str, optional (default: "row")
        Scaling: "row", "column", or "none"
    cmap : str, optional (default: "RdBu_r")
        Color map
    figsize : tuple, optional
        Figure size
    output_file : str, optional
        Output file path
    
    Returns
    -------
    str
        Path to saved plot
    
    Examples
    --------
    >>> data = pd.DataFrame(np.random.randn(50, 6))
    >>> plot_path = draw_heatmap(data, data.columns.tolist())
    
    Notes
    -----
    - Row scaling recommended for gene expression
    - Clustering reveals sample/feature relationships
    """
    import seaborn as sns
    
    plot_data = data[sample_columns].copy()
    
    # Scaling
    if scale == "row":
        plot_data = plot_data.sub(plot_data.mean(axis=1), axis=0).div(
            plot_data.std(axis=1) + 1e-10, axis=0
        )
    elif scale == "column":
        plot_data = (plot_data - plot_data.mean(axis=0)) / (plot_data.std(axis=0) + 1e-10)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.clustermap(
        plot_data,
        cmap=cmap,
        center=0,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        figsize=figsize,
        cbar_kws={'label': 'Z-score' if scale != "none" else 'Value'}
    )
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def draw_umap(
    data: pd.DataFrame,
    sample_columns: list,
    group_labels: Optional[Union[pd.Series, Dict[str, List[str]]]] = None,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple = (10, 8),
    label_samples: bool = True,
    output_file: str = "umap_plot.png"
) -> str:
    """
    Universal UMAP (Uniform Manifold Approximation and Projection) visualization.
    
    Performs dimensionality reduction using UMAP to visualize sample clustering
    and group separation in 2D or 3D space. Useful for quality control and
    exploratory data analysis across all omics types.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_columns : list
        Sample column names to include in analysis
    group_labels : pd.Series or dict, optional
        Group labels for each sample. Can be:
        - pd.Series: Index matches sample_columns, values are group names
        - dict: {group_name: [sample1, sample2, ...]}
        If None, all samples are treated as one group
    n_components : int, optional (default: 2)
        Number of dimensions for UMAP (2 or 3)
    n_neighbors : int, optional (default: 15)
        Number of neighbors for UMAP (larger = more global structure)
    min_dist : float, optional (default: 0.1)
        Minimum distance between points in embedding (0.0-1.0)
    random_state : int, optional (default: 42)
        Random seed for reproducibility
    figsize : tuple, optional (default: (10, 8))
        Figure size
    label_samples : bool, optional (default: True)
        Whether to label individual samples
    output_file : str, optional (default: "umap_plot.png")
        Output file path
    
    Returns
    -------
    str
        Path to saved plot
    
    Examples
    --------
    >>> # With group labels as Series
    >>> groups = pd.Series(
    ...     ['Control']*3 + ['Treatment']*3,
    ...     index=['C1', 'C2', 'C3', 'T1', 'T2', 'T3']
    ... )
    >>> plot_path = draw_umap(data, ['C1', 'C2', 'C3', 'T1', 'T2', 'T3'], groups)
    >>> 
    >>> # With group labels as dict
    >>> groups = {
    ...     'Control': ['C1', 'C2', 'C3'],
    ...     'Treatment': ['T1', 'T2', 'T3']
    ... }
    >>> plot_path = draw_umap(data, sample_cols, groups)
    >>> 
    >>> # 3D UMAP
    >>> plot_path = draw_umap(data, sample_cols, groups, n_components=3)
    
    Notes
    -----
    - Data is standardized (z-score) before UMAP
    - UMAP preserves both local and global structure
    - Good for detecting batch effects and outliers
    - Requires umap-learn package
    
    References
    ----------
    [1] McInnes et al. "UMAP: Uniform Manifold Approximation and Projection",
        Journal of Open Source Software, 2018.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn package is required. Install with: pip install umap-learn"
        )
    
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Extract expression data
    expression_matrix = data[sample_columns].values.T  # Samples as rows
    
    # Handle missing values
    expression_matrix = pd.DataFrame(expression_matrix).fillna(0).values
    
    # Standardize the data
    scaler = StandardScaler()
    expression_matrix_scaled = scaler.fit_transform(expression_matrix)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    embedding = reducer.fit_transform(expression_matrix_scaled)
    
    # Prepare group labels
    if group_labels is None:
        groups = ['All'] * len(sample_columns)
    elif isinstance(group_labels, dict):
        groups = []
        for sample in sample_columns:
            for group_name, sample_list in group_labels.items():
                if sample in sample_list:
                    groups.append(group_name)
                    break
            else:
                groups.append('Unknown')
    elif isinstance(group_labels, pd.Series):
        groups = [group_labels.get(sample, 'Unknown') for sample in sample_columns]
    else:
        groups = ['All'] * len(sample_columns)
    
    # Create dataframe for plotting
    umap_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Group': groups,
        'Sample': sample_columns
    })
    
    if n_components == 3:
        umap_df['UMAP3'] = embedding[:, 2]
    
    # Create plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='Group', data=umap_df, s=100, ax=ax
        )
        
        if label_samples:
            for i, row in umap_df.iterrows():
                ax.annotate(
                    row['Sample'],
                    (row['UMAP1'], row['UMAP2']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('UMAP Projection of Samples', fontsize=14, fontweight='bold')
        ax.legend(title='Group')
        ax.grid(True, alpha=0.3)
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        unique_groups = umap_df['Group'].unique()
        colors = plt.cm.tab10(range(len(unique_groups)))
        
        for i, group in enumerate(unique_groups):
            group_data = umap_df[umap_df['Group'] == group]
            ax.scatter(
                group_data['UMAP1'],
                group_data['UMAP2'],
                group_data['UMAP3'],
                label=group,
                s=100,
                alpha=0.7,
                c=[colors[i]]
            )
        
        if label_samples:
            for i, row in umap_df.iterrows():
                ax.text(
                    row['UMAP1'],
                    row['UMAP2'],
                    row['UMAP3'],
                    row['Sample'],
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_zlabel('UMAP 3', fontsize=10)
        ax.set_title('UMAP Projection (3D)', fontsize=14, fontweight='bold')
        ax.legend()
    
    else:
        raise ValueError("n_components must be 2 or 3")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

