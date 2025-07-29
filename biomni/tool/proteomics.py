import pandas as pd


def t_test_FDR(df, columns1, columns2):
    """
    Performs a t-test analysis between two groups of data with FDR correction.

    This function conducts Welch's t-test (unequal variances t-test) for each row
    in the DataFrame, comparing values between two specified groups of columns.
    The resulting p-values, adjusted p-values (FDR corrected), and t-statistics
    are added as new columns to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze, where each row represents
        a different protein/feature and columns represent samples
    columns1 : list of str, pandas.Index, or numpy.ndarray
        Column names, pandas Index, or numpy array identifying the first group of samples
    columns2 : list of str, pandas.Index, or numpy.ndarray
        Column names, pandas Index, or numpy array identifying the second group of samples

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - 'p_value': raw p-values from the t-test analysis for each row
        - 'FDR': FDR-adjusted p-values using Benjamini-Hochberg method
        - 't_statistic': t-statistics from the t-test analysis

    Notes
    -----
    - Uses Welch's t-test (equal_var=False) which does not assume equal variances
    - Applies Benjamini-Hochberg FDR correction for multiple testing
    - Missing values (NaN) in the data may affect the results
    - Supports both column-based input (list of column names) and array-based input
    - Raises ValueError if NaN values are found in p-values or FDR-adjusted values

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'protein_id': ['P1', 'P2', 'P3'],
    ...     'AD_sample1': [1.2, 2.1, 0.8],
    ...     'AD_sample2': [1.5, 2.3, 0.9],
    ...     'WT_sample1': [0.8, 1.9, 1.2],
    ...     'WT_sample2': [0.9, 2.0, 1.1]
    ... })
    >>> ad_columns = ['AD_sample1', 'AD_sample2']
    >>> wt_columns = ['WT_sample1', 'WT_sample2']
    >>> result = t_test_FDR(df, ad_columns, wt_columns)
    >>> print(result[['protein_id', 'p_value', 'FDR', 't_statistic']])
    """
    import scipy.stats as stats
    from statsmodels.stats.multitest import multipletests

    # Perform t-test for each protein
    pvalues = []
    t_statistics = []
    # Vectorized approach for better performance
    if isinstance(columns1, list) and isinstance(columns2, list):
        # When group1 and group2 are column names
        group1_data = df[columns1].values
        group2_data = df[columns2].values

        # Perform vectorized t-test across all rows
        t_stats, p_vals = stats.ttest_ind(
            group1_data, group2_data, axis=1, equal_var=False
        )
        t_statistics = t_stats.tolist()
        pvalues = p_vals.tolist()

    elif isinstance(columns1, pd.core.indexes.base.Index) and isinstance(
        columns2, pd.core.indexes.base.Index
    ):
        group1_data = df[columns1].values
        group2_data = df[columns2].values

        # Perform vectorized t-test across all rows
        t_stats, p_vals = stats.ttest_ind(
            group1_data, group2_data, axis=1, equal_var=False
        )
        t_statistics = t_stats.tolist()
        pvalues = p_vals.tolist()
    else:
        # When group1 and group2 are already numpy arrays
        t_stats, p_vals = stats.ttest_ind(columns1, columns2, axis=1, equal_var=False)
        t_statistics = t_stats.tolist()
        pvalues = p_vals.tolist()

    # Check for NaN values in p-values
    nan_count = sum(1 for p in pvalues if pd.isna(p))
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaN values in p-values.")
    # Add p_value and t_statistic to the dataframe
    reject, padj, _, _ = multipletests(pvalues, method="fdr_bh")

    # Check for NaN values in FDR
    nan_count = sum(1 for p in padj if pd.isna(p))
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaN values in FDR (padj).")

    df["FDR"] = padj
    df["p_value"] = pvalues
    df["t_statistic"] = t_statistics
    return df


def filter_missing_values(df, columns, criteria=0.5):
    """
    Filter proteins based on detection rate to remove proteins with too many missing values.

    This function calculates the detection rate (non-missing values) for each protein across
    the specified columns and filters out proteins that have detection rates below the
    specified criteria threshold.

    Args:
        df (pd.DataFrame): Input dataframe containing protein expression data
        columns (list or pd.Index): Column names to calculate detection rate from
        criteria (float, optional): Detection rate threshold for filtering. Default is 0.5 (50%)

    Returns:
        pd.DataFrame: Filtered dataframe containing only proteins with detection rate above the criteria

    Example:
        >>> filtered_data = filter_missing_values(df, ['sample1', 'sample2', 'sample3'])
        >>> print(f"Original: {len(df)}, Filtered: {len(filtered_data)}")

        >>> # Set detection rate criteria to 70%
        >>> filtered_data = filter_missing_values(df, columns, criteria=0.7)

    Note:
        - Uses 50% detection rate as the default threshold for filtering
        - Missing values (NaN) are considered as not detected
        - Returns a copy of the filtered dataframe
        - Detection rate threshold can be adjusted via the criteria parameter
    """
    # Calculate detection rate in each group
    rate = ((df[columns].notna()) & (df[columns] != 0.0)).sum(axis=1) / len(columns)

    # Filter proteins detected in at least 50% of samples in both groups
    filtered_df = df[rate >= criteria].copy()
    return filtered_df


def draw_volcano_plot(output_filepath, df, columns1, columns2, fdr_column_name="FDR"):
    """
    Generate a volcano plot for differential expression analysis.

    This function creates a volcano plot to visualize the relationship between fold change
    and statistical significance in differential expression data. Points are colored based
    on significance (FDR < 0.05) and fold change magnitude (|log2FC| > 1).

    Args:
        output_filepath (str): Path where the volcano plot image will be saved
        df (pd.DataFrame): Input dataframe containing expression data or pre-calculated statistics
        columns1 (list): Column names for group 1 (treatment/condition of interest)
        columns2 (list): Column names for group 2 (control/reference condition)
        fdr_column_name (str, optional): Name of the adjusted p-value column. If None,
                                        t-test will be performed. Default is "FDR"

    Returns:
        None: Saves the volcano plot to the specified filepath

    Example:
        >>> # Using pre-calculated statistics
        >>> draw_volcano_plot('volcano.png', results_df, [], [], fdr_column_name='FDR')

        >>> # Calculating statistics from raw data
        >>> draw_volcano_plot('volcano.png', raw_df, ['treat1', 'treat2'], ['ctrl1', 'ctrl2'])

        >>> # Force t-test calculation
        >>> draw_volcano_plot('volcano.png', df, group1_cols, group2_cols, fdr_column_name=None)

    Note:
        - Red points: Significantly upregulated (FDR < 0.05, log2FC > 1)
        - Blue points: Significantly downregulated (FDR < 0.05, log2FC < -1)
        - Orange points: Significant but modest fold change (FDR < 0.05, |log2FC| ≤ 1)
        - Grey points: Not significant (FDR ≥ 0.05)
        - Horizontal dashed line indicates significance threshold
        - Vertical dashed lines indicate fold change thresholds (±1)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate t-test results
    if fdr_column_name is None:
        df2 = t_test_FDR(df, columns1, columns2)
        fdr_column_name = "FDR"
    else:
        df2 = df

    df2["columns1_mean"] = df2[columns1].mean(axis=1)
    df2["columns2_mean"] = df2[columns2].mean(axis=1)
    df2["columns1_std"] = df2[columns1].std(axis=1)
    df2["columns2_std"] = df2[columns2].std(axis=1)
    df2["log2_fold_change"] = np.log2(df2["columns1_mean"] / df2["columns2_mean"])
    df2["-log10_fdr"] = -np.log10(df2[fdr_column_name])
    df2["significant"] = df2[fdr_column_name] < 0.05
    # Create volcano plot
    plt.figure(figsize=(8, 8))
    # FDR < 0.05 & |log2FC| > 1 기준으로 색상 구분
    colors = np.where(
        df2["significant"],
        np.where(
            df2["log2_fold_change"] > 1,
            "red",
            np.where(df2["log2_fold_change"] < -1, "blue", "orange"),
        ),
        "grey",
    )

    plt.scatter(df2["log2_fold_change"], df2["-log10_fdr"], c=colors, alpha=0.6, s=10)

    plt.title("Volcano Plot", fontsize=16)
    plt.xlabel("Mean log2 Fold Change")
    plt.ylabel("-log10(p-value)")
    plt.axhline(
        -np.log10(df2[df2["significant"]]["-log10_fdr"].max()),
        color="gray",
        linestyle="--",
    )
    plt.axvline(1, color="gray", linestyle="--")
    plt.axvline(-1, color="gray", linestyle="--")

    # Add legend manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Upregulated (|log2FC|>1)",
            markerfacecolor="red",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Downregulated (|log2FC|>1)",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Significant (other)",
            markerfacecolor="orange",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Not Significant",
            markerfacecolor="grey",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.savefig(output_filepath)


def draw_heatmap_with_dendrogram(
    output_filepath,
    column_name_of_gene_or_protein,
    df_top,
    columns1,
    columns2,
):
    """
    Create a clustered heatmap with dendrogram for proteomics data visualization.

    This function generates a hierarchically clustered heatmap showing expression patterns
    across different conditions or samples. The data is z-score normalized for better
    visualization of relative expression changes.

    Parameters
    ----------
    output_filepath : str
        Path where the heatmap image will be saved (e.g., 'heatmap.png')
    column_name_of_gene_or_protein : str
        Name of the column containing gene/protein identifiers for row labels
    df_top : pandas.DataFrame
        DataFrame containing the top proteins/genes to visualize, with expression
        data in columns and proteins/genes in rows
    columns1 : list or array-like
        List of column names representing the first condition/group (e.g., treatment)
    columns2 : list or array-like
        List of column names representing the second condition/group (e.g., control)

    Returns
    -------
    None
        Saves the heatmap plot to the specified output filepath

    Notes
    -----
    - Data is z-score normalized row-wise to highlight relative expression patterns
    - Uses hierarchical clustering for both rows (genes/proteins) and columns (samples)
    - Colormap is set to 'viridis' for better accessibility
    - Dendrogram ratios are optimized for typical proteomics datasets

    Examples
    --------
    >>> draw_heatmap_with_dendrogram(
    ...     'top_proteins_heatmap.png',
    ...     'Gene_Name',
    ...     df_significant_proteins,
    ...     ['AD_sample1', 'AD_sample2'],
    ...     ['WT_sample1', 'WT_sample2']
    ... )
    """

    # Create a heatmap for these proteins
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Extract expression data for these proteins
    # Extract only the expression data columns (excluding gene name column)

    heatmap_data = df_top[list(columns1) + list(columns2)]

    # Get gene names for better labels
    gene_names = df_top[column_name_of_gene_or_protein].values

    # Normalize the data for better visualization (z-score)
    from scipy.stats import zscore
    import numpy as np

    # Convert to numpy array for easier manipulation
    heatmap_array = heatmap_data.values

    # Z-score normalization along rows
    heatmap_array_z = np.array([zscore(row) for row in heatmap_array])
    # Rename columns for better visualization
    col_labels = [col for col in heatmap_data.columns]
    # Create a heatmap with dendrogram
    plt.figure(figsize=(14, 10))
    sns.clustermap(
        heatmap_array_z,
        cmap="viridis",
        row_cluster=True,
        col_cluster=True,
        yticklabels=gene_names,
        xticklabels=col_labels,
        figsize=(14, 10),
        dendrogram_ratio=(0.1, 0.2),
        cbar_pos=(0.02, 0.32, 0.03, 0.2),
        cbar_kws={"label": "Z-score"},
    )
    plt.savefig(output_filepath)


def draw_umap(
    output_filepath, df, columns1, columns2, group_names=["Group1", "Group2"]
):
    """
    Generate a UMAP (Uniform Manifold Approximation and Projection) visualization
    of proteomics samples to visualize sample clustering and group separation.

    This function performs dimensionality reduction using UMAP on proteomics expression
    data and creates a scatter plot showing how samples cluster in 2D space. Each sample
    is colored by group (e.g., AD vs WT) and labeled for identification.

    Parameters
    ----------
    output_filepath : str
        Path where the UMAP plot image will be saved (e.g., 'umap_plot.png')
    df : pandas.DataFrame
        DataFrame containing proteomics data with proteins as rows and samples as columns
    columns1 : list or pandas.Index
        Column names for the first group (e.g., AD samples)
    columns2 : list or pandas.Index
        Column names for the second group (e.g., WT samples)
    group_names : list of str, optional
        Names for the two groups to be displayed in the legend, by default ["Group1", "Group2"]

    Returns
    -------
    None
        Saves the UMAP plot to the specified output filepath

    Notes
    -----
    - The function standardizes the expression data before applying UMAP
    - Samples are transposed to be rows (required for UMAP input)
    - The plot includes sample labels extracted from column names
    - Group colors are automatically assigned by seaborn
    - Sample labels are extracted by splitting column names on ": " and taking the second part

    Examples
    --------
    >>> draw_umap(
    ...     'umap_samples.png',
    ...     df_proteomics,
    ...     ['AD_sample1', 'AD_sample2', 'AD_sample3'],
    ...     ['WT_sample1', 'WT_sample2', 'WT_sample3']
    ... )

    >>> draw_umap(
    ...     'umap_custom.png',
    ...     df_data,
    ...     treatment_columns,
    ...     control_columns,
    ...     group_names=['Treatment', 'Control']
    ... )
    """
    import umap
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract expression data for all proteins
    expression_matrix = df[
        list(columns1) + list(columns2)
    ].values.T  # Transpose to get samples as rows

    # Standardize the data
    scaler = StandardScaler()
    expression_matrix_scaled = scaler.fit_transform(expression_matrix)

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(expression_matrix_scaled)

    # Create a dataframe for plotting
    umap_df = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Group": [group_names[0]] * len(columns1)
            + [group_names[1]] * len(columns2),
            "Sample": [col.split(": ")[1] for col in list(columns1) + list(columns2)],
        }
    )

    # Plot UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="Group", data=umap_df, s=100)

    # Add labels for each point
    for i, row in umap_df.iterrows():
        plt.annotate(
            row["Sample"].split(", ")[0],  # Just use the sample ID, not the group
            (row["UMAP1"], row["UMAP2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title("UMAP Projection of Samples")
    plt.tight_layout()
    plt.savefig(output_filepath)


def draw_boxplot(
    output_filepath,
    df,
    column_name_of_gene_or_protein,
    columns1,
    columns2,
    top_n=10,
    group_names=["Group1", "Group2"],
):
    """
    Create boxplots for the top N proteins/genes showing expression differences between groups.

    This function generates a multi-panel boxplot visualization showing the distribution
    of expression values for the most significant proteins/genes across two experimental
    conditions. Each subplot shows both boxplots and individual data points (stripplot)
    for better visualization of the data distribution.

    Parameters
    ----------
    output_filepath : str
        Path where the boxplot image will be saved (e.g., 'boxplots_top10.png')
    df : pandas.DataFrame
        DataFrame containing protein/gene expression data, typically sorted by
        statistical significance (e.g., by FDR or p-value)
    column_name_of_gene_or_protein : str
        Name of the column containing gene/protein identifiers for subplot titles
    columns1 : list or array-like
        List of column names representing the first condition/group (e.g., treatment)
    columns2 : list or array-like
        List of column names representing the second condition/group (e.g., control)
    top_n : int, optional
        Number of top proteins/genes to visualize (default: 10)
    group_names : list of str, optional
        Names for the two groups to display in the plot legend and labels
        (default: ["Group1", "Group2"])

    Returns
    -------
    None
        Saves the boxplot visualization to the specified output filepath

    Notes
    -----
    - The function creates a grid layout with 2 columns and top_n//2 rows
    - Each subplot shows both boxplots and individual data points (stripplot)
    - The input DataFrame should be pre-sorted by significance for meaningful results
    - Missing values in expression data may cause visualization issues

    Examples
    --------
    >>> draw_boxplot(
    ...     'top_proteins_boxplots.png',
    ...     significant_proteins_df,
    ...     'Gene_Name',
    ...     ['AD_sample1', 'AD_sample2'],
    ...     ['WT_sample1', 'WT_sample2'],
    ...     top_n=20,
    ...     group_names=['AD', 'WT']
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Create boxplots for the top 10 significant proteins
    plt.figure(figsize=(15, 20))
    for i, (idx, row) in enumerate(df.head(top_n).iterrows()):
        plt.subplot(top_n // 2, 2, i + 1)

        # Get the data for this protein
        vals1 = row[columns1].values
        vals2 = row[columns2].values

        # Create a dataframe for seaborn
        box_data = pd.DataFrame(
            {
                "Group": [group_names[0]] * len(vals1) + [group_names[1]] * len(vals2),
                "Expression": np.concatenate([vals1, vals2]),
            }
        )

        # Create the boxplot
        sns.boxplot(x="Group", y="Expression", data=box_data)

        # Add individual points
        sns.stripplot(
            x="Group", y="Expression", data=box_data, color="black", size=4, jitter=True
        )

        # Add title and labels
        gene_name = row[column_name_of_gene_or_protein]
        plt.title(f"{gene_name}")
        plt.ylabel("Expression (Scaled)")

    plt.tight_layout()
    plt.savefig(output_filepath)


def enrich_pathway_analysis(genes_of_interest, organism="human", cutoff=0.05):
    """
    Perform pathway enrichment analysis for a list of genes of interest.

    This function uses the GSEApy library to perform enrichment analysis against
    various gene set databases including Gene Ontology (GO) terms and KEGG pathways.

    Parameters
    ----------
    genes_of_interest : list
        List of gene symbols to analyze for pathway enrichment.
    organism : str, optional
        Target organism for the analysis. Default is "human".
        Other options include "mouse", "rat", etc.
    cutoff : float, optional
        P-value cutoff for significant enrichment results. Default is 0.05.

    Returns
    -------
    str
        A formatted string containing the enrichment results including gene set names,
        p-values, and associated genes.

    Notes
    -----
    The function creates an output directory 'enrichment_results_up' containing
    detailed results files.

    Currently uses GO_Biological_Process_2023 gene set library. Additional
    libraries like GO_Molecular_Function_2023, GO_Cellular_Component_2023,
    and KEGG_2019_Mouse can be uncommented as needed.

    Examples
    --------
    >>> genes = ['APOE', 'TREM2', 'CD33']
    >>> results = enrich_pathway_analysis(genes)
    >>> print(results)
    """
    import gseapy as gp

    # Available gene sets in Enrichr
    # all_libs = gp.get_library_name()
    # print(all_libs)  # Print first 10 libraries

    libraries = [
        "GO_Biological_Process_2023",
        # "GO_Molecular_Function_2023",
        # "GO_Cellular_Component_2023",
        # "KEGG_2019_Mouse",
    ]

    # For up-regulated genes
    enr_up = gp.enrichr(
        gene_list=genes_of_interest,
        gene_sets=libraries,
        organism=organism,
        outdir="enrichment_results_up",
        cutoff=cutoff,
    )

    # Process up-regulated gene enrichment results
    result_string = ""
    result_string += "ENRICHMENT RESULTS ===\n\n"
    result_string += enr_up.results.to_string() + "\n\n"

    return result_string
