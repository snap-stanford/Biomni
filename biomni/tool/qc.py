"""
Quality Control Module for Omics Data

This module provides comprehensive quality control functions for multi-omics
data analysis including RNA-seq, proteomics, and metabolomics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from typing import Optional, List, Dict, Tuple, Union


def calculate_qc_metrics(
    data: pd.DataFrame,
    sample_columns: list,
    data_type: str = "proteomics",
    metadata: pd.DataFrame = None,
) -> dict:
    """
    Calculate comprehensive quality control metrics for omics data.
    
    Essential for assessing data quality before downstream analysis.
    Identifies potential issues like outlier samples, batch effects,
    and technical artifacts.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with features as rows and samples as columns
    sample_columns : list
        List of column names containing sample data to analyze
    data_type : str, optional (default: "proteomics")
        Type of omics data. Options:
        - "rnaseq": RNA-seq count data
        - "proteomics": Protein abundance data
        - "metabolomics": Metabolite abundance data
    metadata : pd.DataFrame, optional
        Sample metadata with sample names as index.
        Used for batch effect assessment if available.
    
    Returns
    -------
    dict
        Dictionary containing QC metrics:
        - sample_metrics: Per-sample QC metrics (DataFrame)
        - overall_metrics: Overall data quality metrics (dict)
        - correlation_matrix: Sample correlation matrix (DataFrame)
        - outlier_samples: List of potential outlier samples
        - pca_variance: Variance explained by first 5 PCs (dict)
        - pca_coordinates: PCA coordinates for each sample (DataFrame)
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> data = pd.DataFrame(
    ...     np.random.randn(1000, 6) + 10,
    ...     columns=['s1', 's2', 's3', 's4', 's5', 's6']
    ... )
    >>> 
    >>> # Calculate QC metrics
    >>> qc_results = calculate_qc_metrics(
    ...     data, 
    ...     ['s1', 's2', 's3', 's4', 's5', 's6'],
    ...     data_type="proteomics"
    ... )
    >>> 
    >>> print(qc_results['sample_metrics'])
    >>> print(f"Outliers: {qc_results['outlier_samples']}")
    
    Notes
    -----
    - Missing values should be handled before or after QC assessment
    - PCA is performed on standardized data
    - Outliers are identified using isolation forest method
    - Correlation matrix uses Pearson correlation
    - For RNA-seq, data should be log-transformed counts
    
    References
    ----------
    [1] Conesa et al. "A survey of best practices for RNA-seq data 
        analysis", Genome Biol, 2016.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    
    # Validate inputs
    missing_samples = set(sample_columns) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All sample_columns must exist in data.columns. "
            f"Available columns: {list(data.columns)[:10]}..."
        )
    
    if metadata is not None:
        missing_metadata_samples = set(sample_columns) - set(metadata.index)
        if missing_metadata_samples:
            raise ValueError(
                f"Sample(s) not found in metadata.index: {missing_metadata_samples}. "
                f"metadata.index must contain all sample_columns. "
                f"Example: metadata = pd.DataFrame(..., index=sample_columns)"
            )
    
    # Extract data matrix
    data_matrix = data[sample_columns].values.T  # Transpose to samples x features
    n_samples, n_features = data_matrix.shape
    
    # Initialize results
    results = {}
    
    # === Per-Sample Metrics ===
    sample_metrics = pd.DataFrame(index=sample_columns)
    
    # Total intensity/counts per sample
    sample_metrics['total_intensity'] = np.nansum(data_matrix, axis=1)
    
    # Number of detected features
    if data_type == "rnaseq":
        sample_metrics['detected_features'] = np.sum(data_matrix > 0, axis=1)
    else:
        sample_metrics['detected_features'] = np.sum(~np.isnan(data_matrix), axis=1)
    
    # Missing value percentage
    sample_metrics['missing_percent'] = (np.isnan(data_matrix).sum(axis=1) / n_features) * 100
    
    # Median intensity
    sample_metrics['median_intensity'] = np.nanmedian(data_matrix, axis=1)
    
    # Coefficient of variation (if multiple samples)
    if n_samples > 1:
        sample_metrics['cv'] = np.nanstd(data_matrix, axis=1) / np.nanmean(data_matrix, axis=1)
    
    results['sample_metrics'] = sample_metrics
    
    # === Overall Data Quality Metrics ===
    overall_metrics = {}
    
    # Average detection rate
    overall_metrics['avg_detection_rate'] = sample_metrics['detected_features'].mean() / n_features
    
    # Average missing percentage
    overall_metrics['avg_missing_percent'] = sample_metrics['missing_percent'].mean()
    
    # Dynamic range
    overall_metrics['dynamic_range'] = np.log10(np.nanmax(data_matrix) / np.nanmin(data_matrix[data_matrix > 0]))
    
    # Feature detection consistency (% of features detected in all samples)
    if data_type == "rnaseq":
        detected_all = np.sum(np.all(data_matrix > 0, axis=0))
    else:
        detected_all = np.sum(np.all(~np.isnan(data_matrix), axis=0))
    overall_metrics['features_detected_all_samples'] = detected_all
    overall_metrics['features_detected_all_samples_percent'] = (detected_all / n_features) * 100
    
    results['overall_metrics'] = overall_metrics
    
    # === Sample Correlation Matrix ===
    # Remove NaN for correlation calculation
    clean_matrix = data_matrix.copy()
    # Replace NaN with column mean for correlation
    col_mean = np.nanmean(clean_matrix, axis=0)
    inds = np.where(np.isnan(clean_matrix))
    clean_matrix[inds] = np.take(col_mean, inds[1])
    
    corr_matrix = np.corrcoef(clean_matrix)
    results['correlation_matrix'] = pd.DataFrame(
        corr_matrix,
        index=sample_columns,
        columns=sample_columns
    )
    
    # === PCA for Outlier Detection ===
    # Standardize data
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(clean_matrix)
    
    # Perform PCA
    pca = PCA(n_components=min(5, n_samples))
    pca_result = pca.fit_transform(scaled_matrix)
    
    results['pca_variance'] = {
        f'PC{i+1}': var for i, var in enumerate(pca.explained_variance_ratio_)
    }
    results['pca_coordinates'] = pd.DataFrame(
        pca_result,
        index=sample_columns,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )
    
    # === Outlier Detection ===
    # Use Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_predictions = iso_forest.fit_predict(scaled_matrix)
    
    outlier_samples = [
        sample for sample, pred in zip(sample_columns, outlier_predictions)
        if pred == -1
    ]
    results['outlier_samples'] = outlier_samples
    
    # === Batch Effect Assessment (if metadata provided) ===
    if metadata is not None and 'batch' in metadata.columns:
        try:
            from sklearn.metrics import silhouette_score
            
            batch_labels = metadata.loc[sample_columns, 'batch'].values
            
            # Calculate silhouette score (measures batch separation)
            # Negative score means batches are NOT well separated (good!)
            # Positive score means batches ARE separated (potential batch effect)
            if len(np.unique(batch_labels)) > 1:
                sil_score = silhouette_score(scaled_matrix, batch_labels)
                results['batch_effect_score'] = sil_score
                
                if sil_score > 0.3:
                    results['batch_effect_warning'] = "Strong batch effect detected (silhouette > 0.3). Consider batch correction."
                elif sil_score > 0.1:
                    results['batch_effect_warning'] = "Moderate batch effect detected (silhouette > 0.1). Review data carefully."
                else:
                    results['batch_effect_warning'] = "No strong batch effect detected."
        except:
            pass
    
    return results


# def draw_pca(
#     data: pd.DataFrame,
#     sample_columns: list,
#     metadata: pd.DataFrame = None,
#     color_by: str = None,
#     n_components: int = 2,
#     show_loadings: bool = False,
#     n_loadings: int = 10,
#     output_file: str = "pca_plot.png",
# ) -> str:
#     """
#     Create PCA visualization for quality control and sample relationship assessment.
    
#     Essential QC visualization tool for identifying sample relationships, batch effects,
#     and outliers. Part of quality control workflow for omics data analysis.
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         Data with features as rows, samples as columns
#     sample_columns : list
#         Sample names to include. All must exist in data.columns.
#     metadata : pd.DataFrame, optional
#         Sample metadata for coloring. Index must match sample_columns.
#         Example: metadata = pd.DataFrame(..., index=sample_columns)
#     color_by : str, optional
#         Metadata column for coloring (e.g., 'batch', 'group', 'treatment')
#     n_components : int, optional (default: 2)
#         Number of PCs to compute (2 or 3 for visualization)
#     show_loadings : bool, optional (default: False)
#         Show feature loading vectors (not yet implemented)
#     n_loadings : int, optional (default: 10)
#         Number of loadings to show if show_loadings=True
#     output_file : str, optional (default: "pca_plot.png")
#         Output file path
    
#     Returns
#     -------
#     str
#         Path to saved plot
    
#     Examples
#     --------
#     >>> import pandas as pd
#     >>> import numpy as np
#     >>> 
#     >>> # Create sample data
#     >>> data = pd.DataFrame(np.random.randn(100, 6))
#     >>> 
#     >>> # Basic PCA plot
#     >>> plot_path = draw_pca(data, data.columns.tolist())
#     >>> 
#     >>> # With metadata coloring
#     >>> metadata = pd.DataFrame({
#     ...     'batch': ['Batch1', 'Batch1', 'Batch2', 'Batch2', 'Batch3', 'Batch3']
#     ... }, index=data.columns)
#     >>> plot_path = draw_pca(data, data.columns.tolist(), metadata, color_by='batch')
    
#     Notes
#     -----
#     - Normalize and clean data before PCA
#     - Missing values are automatically removed (features with any NaN are excluded)
#     - Data is standardized (z-score) before PCA
#     - Use this for QC to identify batch effects and outliers
#     - For QC metrics calculation, use calculate_qc_metrics() which also performs PCA
#     - This function focuses on visualization, while calculate_qc_metrics() focuses on metrics
    
#     See Also
#     --------
#     calculate_qc_metrics : Calculate comprehensive QC metrics including PCA coordinates
#     generate_qc_report : Generate HTML QC report with PCA plot included
#     """
#     from sklearn.decomposition import PCA
#     from sklearn.preprocessing import StandardScaler
    
#     # Validate inputs
#     missing_samples = set(sample_columns) - set(data.columns)
#     if missing_samples:
#         raise ValueError(
#             f"Sample(s) not found in data.columns: {missing_samples}. "
#             f"All sample_columns must exist in data.columns. "
#             f"Available columns: {list(data.columns)[:10]}..."
#         )
    
#     if metadata is not None and color_by is not None:
#         missing_metadata_samples = set(sample_columns) - set(metadata.index)
#         if missing_metadata_samples:
#             raise ValueError(
#                 f"Sample(s) not found in metadata.index: {missing_metadata_samples}. "
#                 f"metadata.index must contain all sample_columns. "
#                 f"Example: metadata = pd.DataFrame(..., index=sample_columns)"
#             )
    
#     # Extract data
#     data_matrix = data[sample_columns].values.T  # Transpose: samples x features
    
#     # Remove missing values
#     valid_features = ~np.any(np.isnan(data_matrix), axis=0)
#     data_clean = data_matrix[:, valid_features]
    
#     if data_clean.shape[1] == 0:
#         raise ValueError("No valid features after removing missing values. Check data quality.")
    
#     # Standardize
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data_clean)
    
#     # Perform PCA
#     pca = PCA(n_components=n_components)
#     pca_result = pca.fit_transform(data_scaled)
    
#     # Calculate variance explained
#     variance_explained = pca.explained_variance_ratio_ * 100
    
#     # Set style
#     plt.style.use('seaborn-v0_8-whitegrid')
#     sns.set_palette("husl")
    
#     # Create plot with enhanced styling
#     if n_components == 2:
#         fig, ax = plt.subplots(figsize=(12, 10))
#         fig.patch.set_facecolor('white')
        
#         # Color by metadata if provided
#         if metadata is not None and color_by is not None:
#             groups = metadata.loc[sample_columns, color_by]
#             unique_groups = groups.unique()
#             colors = sns.color_palette("husl", len(unique_groups))
            
#             for i, group in enumerate(unique_groups):
#                 mask = groups == group
#                 ax.scatter(
#                     pca_result[mask, 0],
#                     pca_result[mask, 1],
#                     c=[colors[i]],
#                     label=group,
#                     s=150,
#                     alpha=0.7,
#                     edgecolors='white',
#                     linewidths=2,
#                     zorder=3
#                 )
            
#             # Enhanced legend
#             legend = ax.legend(title=color_by, loc='best', frameon=True,
#                               fancybox=True, shadow=True,
#                               fontsize=11, title_fontsize=12,
#                               framealpha=0.95, edgecolor='gray', facecolor='white')
#             legend.get_frame().set_linewidth(1.5)
#         else:
#             ax.scatter(
#                 pca_result[:, 0],
#                 pca_result[:, 1],
#                 c='#3498DB',
#                 s=150,
#                 alpha=0.7,
#                 edgecolors='white',
#                 linewidths=2,
#                 zorder=3
#             )
        
#         # Add sample labels with better styling
#         for i, sample in enumerate(sample_columns):
#             ax.annotate(
#                 sample,
#                 (pca_result[i, 0], pca_result[i, 1]),
#                 xytext=(8, 8),
#                 textcoords='offset points',
#                 fontsize=9,
#                 fontweight='bold',
#                 alpha=0.8,
#                 bbox=dict(boxstyle='round,pad=0.3', 
#                         facecolor='white', 
#                         alpha=0.7,
#                         edgecolor='gray',
#                         linewidth=0.5),
#                 zorder=4
#             )
        
#         # Enhanced labels with variance explained
#         ax.set_xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)', 
#                      fontsize=14, fontweight='bold', color='#2C3E50')
#         ax.set_ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)', 
#                      fontsize=14, fontweight='bold', color='#2C3E50')
#         ax.set_title('Principal Component Analysis', fontsize=16, fontweight='bold', 
#                     color='#2C3E50', pad=15)
        
#         # Enhanced grid
#         ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
#         ax.set_axisbelow(True)
#         ax.set_facecolor('#FAFAFA')
        
#         # Style spines
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_color('#7F8C8D')
#         ax.spines['bottom'].set_color('#7F8C8D')
#         ax.spines['left'].set_linewidth(1.5)
#         ax.spines['bottom'].set_linewidth(1.5)
        
#     elif n_components == 3:
#         from mpl_toolkits.mplot3d import Axes3D
        
#         fig = plt.figure(figsize=(12, 10))
#         fig.patch.set_facecolor('white')
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Color by metadata if provided
#         if metadata is not None and color_by is not None:
#             groups = metadata.loc[sample_columns, color_by]
#             unique_groups = groups.unique()
#             colors = sns.color_palette("husl", len(unique_groups))
            
#             for i, group in enumerate(unique_groups):
#                 mask = groups == group
#                 ax.scatter(
#                     pca_result[mask, 0],
#                     pca_result[mask, 1],
#                     pca_result[mask, 2],
#                     c=[colors[i]],
#                     label=group,
#                     s=150,
#                     alpha=0.7,
#                     edgecolors='white',
#                     linewidths=1.5
#                 )
            
#             # Enhanced legend
#             legend = ax.legend(title=color_by, loc='upper left', frameon=True,
#                               fancybox=True, shadow=True,
#                               fontsize=11, title_fontsize=12, framealpha=0.95)
#             legend.get_frame().set_linewidth(1.5)
#         else:
#             ax.scatter(
#                 pca_result[:, 0],
#                 pca_result[:, 1],
#                 pca_result[:, 2],
#                 c='#3498DB',
#                 s=150,
#                 alpha=0.7,
#                 edgecolors='white',
#                 linewidths=1.5
#             )
        
#         # Add sample labels
#         for i, sample in enumerate(sample_columns):
#             ax.text(
#                 pca_result[i, 0],
#                 pca_result[i, 1],
#                 pca_result[i, 2],
#                 sample,
#                 fontsize=9,
#                 fontweight='bold',
#                 alpha=0.8,
#                 bbox=dict(boxstyle='round,pad=0.3', 
#                         facecolor='white', 
#                         alpha=0.7,
#                         edgecolor='gray')
#             )
        
#         # Enhanced labels with variance explained
#         ax.set_xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)', 
#                      fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)', 
#                      fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_zlabel(f'PC3 ({variance_explained[2]:.1f}% variance)', 
#                      fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_title('Principal Component Analysis (3D)', fontsize=16, fontweight='bold', pad=20)
        
#         # Style 3D axes
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False
#         ax.xaxis.pane.set_edgecolor('gray')
#         ax.yaxis.pane.set_edgecolor('gray')
#         ax.zaxis.pane.set_edgecolor('gray')
#         ax.xaxis.pane.set_alpha(0.1)
#         ax.yaxis.pane.set_alpha(0.1)
#         ax.zaxis.pane.set_alpha(0.1)
    
#     else:
#         raise ValueError("n_components must be 2 or 3 for visualization")
    
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight', 
#                facecolor='white', edgecolor='none')
#     plt.close()
    
#     return output_file


def generate_qc_report(
    qc_results: dict,
    output_file: str = "qc_report.html",
) -> str:
    """
    Generate an HTML QC report with plots and metrics.
    
    Parameters
    ----------
    qc_results : dict
        Results from calculate_qc_metrics()
    output_file : str
        Path to save HTML report
    
    Returns
    -------
    str
        Path to saved report
    """
    # Create QC plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sample_metrics = qc_results['sample_metrics']
    
    # 1. Total intensity distribution
    axes[0, 0].bar(range(len(sample_metrics)), sample_metrics['total_intensity'])
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Total Intensity')
    axes[0, 0].set_title('Total Intensity per Sample')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Detection rate
    axes[0, 1].bar(range(len(sample_metrics)), sample_metrics['detected_features'])
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Number of Detected Features')
    axes[0, 1].set_title('Feature Detection per Sample')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Correlation heatmap
    corr_matrix = qc_results['correlation_matrix']
    im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 0].set_xticks(range(len(corr_matrix)))
    axes[1, 0].set_yticks(range(len(corr_matrix)))
    axes[1, 0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(corr_matrix.index)
    axes[1, 0].set_title('Sample Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. PCA plot
    pca_coords = qc_results['pca_coordinates']
    axes[1, 1].scatter(pca_coords['PC1'], pca_coords['PC2'], s=100, alpha=0.6)
    
    # Mark outliers
    for sample in qc_results['outlier_samples']:
        idx = pca_coords.index.get_loc(sample)
        axes[1, 1].scatter(
            pca_coords.loc[sample, 'PC1'],
            pca_coords.loc[sample, 'PC2'],
            s=150, c='red', marker='x', linewidths=3,
            label='Outlier' if sample == qc_results['outlier_samples'][0] else ''
        )
    
    # Add sample labels
    for idx, sample in enumerate(pca_coords.index):
        axes[1, 1].annotate(
            sample,
            (pca_coords.loc[sample, 'PC1'], pca_coords.loc[sample, 'PC2']),
            xytext=(5, 5), textcoords='offset points', fontsize=8
        )
    
    pca_var = qc_results['pca_variance']
    axes[1, 1].set_xlabel(f"PC1 ({pca_var['PC1']*100:.1f}%)")
    axes[1, 1].set_ylabel(f"PC2 ({pca_var['PC2']*100:.1f}%)")
    axes[1, 1].set_title('PCA - Sample Overview')
    if qc_results['outlier_samples']:
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_file.replace('.html', '_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QC Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .warning {{ color: #ff9800; font-weight: bold; }}
            .good {{ color: #4CAF50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Quality Control Report</h1>
        
        <h2>Overall Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """
    
    for key, value in qc_results['overall_metrics'].items():
        if isinstance(value, float):
            html_content += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
        else:
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html_content += "</table>\n"
    
    # Outlier warnings
    if qc_results['outlier_samples']:
        html_content += f"""
        <h2 class="warning">⚠️ Outlier Samples Detected</h2>
        <p>The following samples were identified as potential outliers:</p>
        <ul>
        """
        for sample in qc_results['outlier_samples']:
            html_content += f"<li>{sample}</li>\n"
        html_content += "</ul>\n"
    else:
        html_content += '<h2 class="good">✓ No Outlier Samples Detected</h2>\n'
    
    # Batch effect warning
    if 'batch_effect_warning' in qc_results:
        html_content += f"""
        <h2>Batch Effect Assessment</h2>
        <p>{qc_results['batch_effect_warning']}</p>
        """
    
    # Add plots
    html_content += f"""
        <h2>Quality Control Plots</h2>
        <img src="{plot_file}" style="width:100%; max-width:1200px;">
        
        <h2>Sample Metrics</h2>
    """
    
    # Add sample metrics table
    html_content += sample_metrics.to_html()
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file


def correct_batch_effects(
    data: pd.DataFrame,
    sample_columns: List[str],
    batch: Union[pd.Series, List],
    covariates: Optional[pd.DataFrame] = None,
    method: str = "combat",
    preserve_design: bool = True
) -> pd.DataFrame:
    """
    Correct for batch effects in omics data.
    
    Essential for removing systematic technical variation due to different
    processing batches, instruments, or experimental runs. Critical for
    multi-batch studies and meta-analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_columns : list
        List of sample column names to correct
    batch : pd.Series or list
        Batch labels for each sample. Must have same length as sample_columns.
    covariates : pd.DataFrame, optional
        Biological covariates to preserve (e.g., treatment, disease status).
        Index must match sample_columns exactly (same length and same elements).
        Example: covariates = pd.DataFrame(..., index=sample_columns)
    method : str, optional (default: "combat")
        Batch correction method:
        - "combat": ComBat (parametric, recommended for >2 batches)
        - "combat_nonparametric": Non-parametric ComBat (small batches)
        - "mean_center": Simple mean centering per batch
        - "quantile": Quantile normalization across batches
    preserve_design : bool, optional (default: True)
        Whether to preserve biological variation in covariates
    
    Returns
    -------
    pd.DataFrame
        Batch-corrected data with same structure as input
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create data with batch effects
    >>> data = pd.DataFrame(np.random.randn(100, 12))
    >>> data.iloc[:, 0:4] += 2  # Batch 1 higher
    >>> data.iloc[:, 4:8] -= 1  # Batch 2 lower
    >>> 
    >>> # Define batches
    >>> batch = ['Batch1']*4 + ['Batch2']*4 + ['Batch3']*4
    >>> 
    >>> # Correct batch effects
    >>> corrected = correct_batch_effects(
    ...     data, data.columns.tolist(), batch, method='combat'
    ... )
    
    Notes
    -----
    - ComBat uses empirical Bayes to shrink batch effect estimates
    - Most effective when batch sizes are balanced (>3 samples per batch)
    - Covariates should include biological factors to preserve
    - Mean centering is simplest but less robust than ComBat
    - Run QC before and after to verify correction
    
    References
    ----------
    [1] Johnson et al. "Adjusting batch effects in microarray expression 
        data using empirical Bayes methods", Biostatistics, 2007.
    [2] Leek et al. "Tackling the widespread and critical impact of batch 
        effects in high-throughput data", Nat Rev Genet, 2010.
    """
    result = data.copy()
    
    # Convert batch to array
    if isinstance(batch, pd.Series):
        batch_array = batch.values
    else:
        batch_array = np.array(batch)
    
    if len(batch_array) != len(sample_columns):
        raise ValueError(f"Batch length ({len(batch_array)}) != sample_columns length ({len(sample_columns)})")
    
    # Validate sample_columns exist in data
    missing_samples = set(sample_columns) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All sample_columns must exist in data.columns."
        )
    
    # Validate covariates if provided
    if covariates is not None:
        missing_cov_samples = set(sample_columns) - set(covariates.index)
        if missing_cov_samples:
            raise ValueError(
                f"Sample(s) not found in covariates.index: {missing_cov_samples}. "
                f"covariates.index must match sample_columns exactly. "
                f"Example: covariates = pd.DataFrame(..., index=sample_columns)"
            )
    
    # Get data matrix
    data_matrix = data[sample_columns].values.astype(float)
    
    if method == "combat":
        # ComBat batch correction (parametric)
        corrected = _combat_correction(
            data_matrix, batch_array, covariates, parametric=True
        )
    
    elif method == "combat_nonparametric":
        # ComBat non-parametric
        corrected = _combat_correction(
            data_matrix, batch_array, covariates, parametric=False
        )
    
    elif method == "mean_center":
        # Simple mean centering per batch
        corrected = data_matrix.copy()
        unique_batches = np.unique(batch_array)
        
        # Calculate overall mean per feature
        overall_mean = np.nanmean(data_matrix, axis=1, keepdims=True)
        
        for batch_label in unique_batches:
            batch_mask = batch_array == batch_label
            batch_data = data_matrix[:, batch_mask]
            
            # Calculate batch mean
            batch_mean = np.nanmean(batch_data, axis=1, keepdims=True)
            
            # Center to overall mean
            corrected[:, batch_mask] = batch_data - batch_mean + overall_mean
    
    elif method == "quantile":
        # Quantile normalization across batches
        from scipy.interpolate import interp1d
        
        # Normalize within each batch
        corrected = data_matrix.copy()
        unique_batches = np.unique(batch_array)
        
        for batch_label in unique_batches:
            batch_mask = batch_array == batch_label
            batch_data = data_matrix[:, batch_mask]
            
            # Quantile normalize batch
            sorted_data = np.sort(batch_data, axis=0)
            row_means = np.nanmean(sorted_data, axis=1)
            ranks = batch_data.argsort(axis=0).argsort(axis=0)
            
            for j in range(batch_data.shape[1]):
                corrected[:, batch_mask][:, j] = row_means[ranks[:, j]]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Update result
    result[sample_columns] = corrected
    
    return result


def _combat_correction(
    data_matrix: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[pd.DataFrame],
    parametric: bool = True
) -> np.ndarray:
    """
    ComBat batch correction implementation.
    
    Simplified version of ComBat algorithm for batch effect removal.
    """
    n_features, n_samples = data_matrix.shape
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)
    
    # Design matrix for batch
    batch_design = np.zeros((n_samples, n_batches))
    for i, batch_label in enumerate(unique_batches):
        batch_design[:, i] = (batch == batch_label).astype(int)
    
    # Add covariates if provided
    if covariates is not None:
        # Standardize covariates
        cov_matrix = covariates.values
        cov_std = (cov_matrix - cov_matrix.mean(axis=0)) / (cov_matrix.std(axis=0) + 1e-8)
        design_matrix = np.hstack([batch_design, cov_std])
    else:
        design_matrix = batch_design
    
    # Standardize data
    grand_mean = np.nanmean(data_matrix, axis=1, keepdims=True)
    data_std = data_matrix - grand_mean
    
    # Estimate batch effects using least squares
    gamma_hat = np.zeros((n_features, n_batches))
    delta_hat = np.zeros((n_features, n_batches))
    
    for i in range(n_batches):
        batch_mask = batch == unique_batches[i]
        if batch_mask.sum() > 0:
            batch_data = data_std[:, batch_mask]
            
            # Additive batch effect (location)
            gamma_hat[:, i] = np.nanmean(batch_data, axis=1)
            
            # Multiplicative batch effect (scale)
            delta_hat[:, i] = np.nanvar(batch_data, axis=1)
    
    # Empirical Bayes shrinkage (if parametric)
    if parametric and n_batches > 1:
        # Shrink batch effects toward overall distribution
        for i in range(n_batches):
            # Shrink location parameters
            gamma_bar = np.nanmean(gamma_hat, axis=1)
            gamma_var = np.nanvar(gamma_hat, axis=1)
            
            # Simple shrinkage
            shrinkage = gamma_var / (gamma_var + np.nanvar(data_std, axis=1))
            gamma_hat[:, i] = shrinkage * gamma_hat[:, i] + (1 - shrinkage) * gamma_bar
    
    # Apply correction
    corrected = data_matrix.copy()
    
    for i, batch_label in enumerate(unique_batches):
        batch_mask = batch == batch_label
        if batch_mask.sum() > 0:
            # Remove additive batch effect
            corrected[:, batch_mask] = (
                corrected[:, batch_mask] - gamma_hat[:, i:i+1]
            )
    
    return corrected


def assess_technical_replicates(
    data: pd.DataFrame,
    replicate_groups: Dict[str, List[str]],
    metrics: List[str] = ["cv", "icc", "correlation"]
) -> pd.DataFrame:
    """
    Assess quality of technical replicates.
    
    Essential for validating experimental reproducibility and identifying
    technical issues. High replicate variability indicates problems with
    sample preparation or measurement.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    replicate_groups : dict
        Dictionary mapping group names to lists of replicate sample names.
        Format: {group_name: [sample1, sample2, sample3]}
        All samples in replicate_groups must exist in data.columns.
    metrics : list, optional
        Metrics to calculate. Options:
        - "cv": Coefficient of variation
        - "icc": Intraclass correlation coefficient
        - "correlation": Pearson correlation
        - "mad": Median absolute deviation
    
    Returns
    -------
    pd.DataFrame
        Summary statistics for each replicate group
    
    Examples
    --------
    >>> # Define technical replicates
    >>> replicates = {
    ...     'Sample1': ['S1_rep1', 'S1_rep2', 'S1_rep3'],
    ...     'Sample2': ['S2_rep1', 'S2_rep2', 'S2_rep3']
    ... }
    >>> 
    >>> # Assess quality
    >>> rep_qc = assess_technical_replicates(data, replicates)
    >>> print(f"Median CV: {rep_qc['median_cv'].mean():.2f}%")
    
    Notes
    -----
    - CV < 20% generally considered good for technical replicates
    - ICC > 0.75 indicates high reproducibility
    - High replicate variability may indicate technical problems
    - Compare biological vs technical variation
    
    References
    ----------
    [1] Koo & Li. "A Guideline of Selecting and Reporting Intraclass 
        Correlation Coefficients for Reliability Research", J Chiropr Med, 2016.
    """
    # Validate that all samples exist in data
    all_replicate_samples = []
    for sample_list in replicate_groups.values():
        all_replicate_samples.extend(sample_list)
    
    missing_samples = set(all_replicate_samples) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All samples in replicate_groups must exist in data.columns. "
            f"Available columns: {list(data.columns)[:10]}..."
        )
    
    results = []
    
    for group_name, sample_list in replicate_groups.items():
        if len(sample_list) < 2:
            continue
        
        # Extract replicate data
        rep_data = data[sample_list].values
        
        group_metrics = {'group': group_name, 'n_replicates': len(sample_list)}
        
        # Calculate requested metrics
        if "cv" in metrics:
            # Coefficient of variation per feature
            means = np.nanmean(rep_data, axis=1)
            stds = np.nanstd(rep_data, axis=1)
            cvs = (stds / (means + 1e-10)) * 100
            
            group_metrics['median_cv'] = np.nanmedian(cvs)
            group_metrics['mean_cv'] = np.nanmean(cvs)
            group_metrics['cv_75th'] = np.nanpercentile(cvs, 75)
        
        if "correlation" in metrics:
            # Average pairwise correlation
            correlations = []
            for i in range(len(sample_list)):
                for j in range(i+1, len(sample_list)):
                    x = rep_data[:, i]
                    y = rep_data[:, j]
                    
                    # Remove NaN pairs
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() > 0:
                        corr, _ = stats.pearsonr(x[mask], y[mask])
                        correlations.append(corr)
            
            if correlations:
                group_metrics['mean_correlation'] = np.mean(correlations)
                group_metrics['min_correlation'] = np.min(correlations)
        
        if "icc" in metrics:
            # Intraclass correlation coefficient
            # ICC(2,1) - two-way random effects model
            n_features = rep_data.shape[0]
            n_reps = len(sample_list)
            
            # Calculate variance components
            grand_mean = np.nanmean(rep_data)
            
            # Between-features variance
            feature_means = np.nanmean(rep_data, axis=1)
            bms = n_reps * np.nanvar(feature_means)
            
            # Within-features variance
            residuals = rep_data - feature_means[:, np.newaxis]
            wms = np.nanmean(np.nanvar(residuals, axis=1))
            
            # ICC(2,1)
            icc = (bms - wms) / (bms + (n_reps - 1) * wms)
            group_metrics['icc'] = max(0, min(1, icc))  # Bound between 0 and 1
        
        if "mad" in metrics:
            # Median absolute deviation
            medians = np.nanmedian(rep_data, axis=1, keepdims=True)
            mads = np.nanmedian(np.abs(rep_data - medians), axis=1)
            
            group_metrics['median_mad'] = np.nanmedian(mads)
            group_metrics['mean_mad'] = np.nanmean(mads)
        
        results.append(group_metrics)
    
    return pd.DataFrame(results)


def detect_outlier_features(
    data: pd.DataFrame,
    sample_columns: List[str],
    method: str = "iqr",
    threshold: float = 3.0
) -> Dict[str, List[str]]:
    """
    Detect outlier features (not samples).
    
    Identifies features with unusual patterns that may indicate technical
    artifacts, contamination, or measurement errors.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_columns : list
        Sample columns to analyze
    method : str, optional (default: "iqr")
        Detection method:
        - "iqr": Interquartile range method
        - "zscore": Z-score method
        - "mad": Median absolute deviation
    threshold : float, optional (default: 3.0)
        Threshold for outlier detection (interpretation depends on method)
    
    Returns
    -------
    dict
        Dictionary with:
        - outlier_features: List of outlier feature names
        - outlier_scores: Dictionary mapping features to outlier scores
    
    Examples
    --------
    >>> outliers = detect_outlier_features(data, sample_cols, method='iqr')
    >>> print(f"Found {len(outliers['outlier_features'])} outlier features")
    
    Notes
    -----
    - IQR method: Values beyond Q1-1.5*IQR or Q3+1.5*IQR
    - Z-score: |z| > threshold (typically 3)
    - MAD: More robust to extreme outliers than z-score
    - Review outliers manually before removal
    - This function DETECTS outlier features (returns list), does not remove them
    - For filtering LOW expression/detection features (preprocessing), use filter_low_values() from omics module
    - This is for QC purposes to identify problematic features, not for routine filtering
    """
    # Validate that all samples exist in data
    missing_samples = set(sample_columns) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All sample_columns must exist in data.columns. "
            f"Available columns: {list(data.columns)[:10]}..."
        )
    
    data_matrix = data[sample_columns].values
    
    outlier_scores = {}
    
    if method == "iqr":
        # IQR method
        for i, feature in enumerate(data.index):
            values = data_matrix[i, :]
            values_clean = values[~np.isnan(values)]
            
            if len(values_clean) > 0:
                q1 = np.percentile(values_clean, 25)
                q3 = np.percentile(values_clean, 75)
                iqr = q3 - q1
                
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                
                n_outliers = ((values_clean < lower) | (values_clean > upper)).sum()
                outlier_scores[feature] = n_outliers / len(values_clean)
    
    elif method == "zscore":
        # Z-score method
        for i, feature in enumerate(data.index):
            values = data_matrix[i, :]
            values_clean = values[~np.isnan(values)]
            
            if len(values_clean) > 0:
                z_scores = np.abs(stats.zscore(values_clean))
                n_outliers = (z_scores > threshold).sum()
                outlier_scores[feature] = n_outliers / len(values_clean)
    
    elif method == "mad":
        # Median absolute deviation (more robust)
        for i, feature in enumerate(data.index):
            values = data_matrix[i, :]
            values_clean = values[~np.isnan(values)]
            
            if len(values_clean) > 0:
                median = np.median(values_clean)
                mad = np.median(np.abs(values_clean - median))
                
                if mad > 0:
                    modified_z = 0.6745 * (values_clean - median) / mad
                    n_outliers = (np.abs(modified_z) > threshold).sum()
                    outlier_scores[feature] = n_outliers / len(values_clean)
                else:
                    outlier_scores[feature] = 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Identify outliers (features with >10% outlier values)
    outlier_features = [
        feat for feat, score in outlier_scores.items()
        if score > 0.1
    ]
    
    return {
        'outlier_features': outlier_features,
        'outlier_scores': outlier_scores
    }


def assess_missing_value_patterns(
    data: pd.DataFrame,
    sample_columns: List[str],
    groups: Optional[pd.Series] = None
) -> Dict[str, any]:
    """
    Analyze missing value patterns to identify mechanism.
    
    Critical for choosing appropriate imputation strategy. Different missing
    value mechanisms (MCAR, MAR, MNAR) require different handling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_columns : list
        Sample columns to analyze
    groups : pd.Series, optional
        Group labels for each sample (to test if missingness is group-related).
        Index must match sample_columns exactly.
        Example: groups = pd.Series(['Group1', 'Group1', 'Group2'], index=sample_columns)
    
    Returns
    -------
    dict
        Dictionary with:
        - total_missing: Total percentage of missing values
        - missing_per_feature: Missing percentage per feature
        - missing_per_sample: Missing percentage per sample
        - likely_mechanism: Predicted mechanism (MCAR, MAR, MNAR)
        - group_bias: Whether missingness is biased by group (if groups provided)
    
    Examples
    --------
    >>> missing_info = assess_missing_value_patterns(data, sample_cols, groups)
    >>> print(f"Likely mechanism: {missing_info['likely_mechanism']}")
    >>> print(f"Total missing: {missing_info['total_missing']:.1f}%")
    
    Notes
    -----
    - MCAR: Missing Completely At Random (safe to ignore/impute)
    - MAR: Missing At Random (conditional on observed data)
    - MNAR: Missing Not At Random (related to unobserved values)
    - Proteomics often has MNAR (low abundance proteins not detected)
    - Use Little's MCAR test for formal testing
    
    References
    ----------
    [1] Lazar et al. "Accounting for the Multiple Natures of Missing Values",
        J Proteome Res, 2016.
    """
    # Validate inputs
    missing_samples = set(sample_columns) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All sample_columns must exist in data.columns. "
            f"Available columns: {list(data.columns)[:10]}..."
        )
    
    if groups is not None:
        missing_group_samples = set(sample_columns) - set(groups.index)
        if missing_group_samples:
            raise ValueError(
                f"Sample(s) not found in groups.index: {missing_group_samples}. "
                f"groups.index must match sample_columns exactly. "
                f"Example: groups = pd.Series(['Group1', 'Group1', 'Group2'], index=sample_columns)"
            )
    
    data_matrix = data[sample_columns].values
    
    # Calculate missing percentages
    missing_mask = np.isnan(data_matrix)
    total_missing = (missing_mask.sum() / missing_mask.size) * 100
    
    missing_per_feature = (missing_mask.sum(axis=1) / len(sample_columns)) * 100
    missing_per_sample = (missing_mask.sum(axis=0) / len(data)) * 100
    
    results = {
        'total_missing': total_missing,
        'missing_per_feature': pd.Series(missing_per_feature, index=data.index),
        'missing_per_sample': pd.Series(missing_per_sample, index=sample_columns)
    }
    
    # Test if missingness related to intensity (MNAR indicator)
    # Features with low mean intensity should have more missingness
    mean_intensity = np.nanmean(data_matrix, axis=1)
    
    # Calculate correlation between mean intensity and missingness
    valid_mask = ~np.isnan(mean_intensity)
    if valid_mask.sum() > 10:
        corr, pval = stats.spearmanr(
            mean_intensity[valid_mask],
            missing_per_feature[valid_mask]
        )
        
        if corr < -0.3 and pval < 0.05:
            likely_mechanism = "MNAR"  # Low intensity -> more missing
        elif abs(corr) < 0.1:
            likely_mechanism = "MCAR"  # No relationship
        else:
            likely_mechanism = "MAR"   # Some relationship
        
        results['intensity_missing_correlation'] = corr
        results['intensity_missing_pvalue'] = pval
    else:
        likely_mechanism = "Unknown"
    
    results['likely_mechanism'] = likely_mechanism
    
    # Test for group bias in missingness
    if groups is not None:
        group_missing_rates = []
        unique_groups = groups.unique()
        
        for group in unique_groups:
            group_samples = groups[groups == group].index.tolist()
            group_cols = [col for col in group_samples if col in sample_columns]
            
            if group_cols:
                group_data = data[group_cols].values
                group_missing = (np.isnan(group_data).sum() / group_data.size) * 100
                group_missing_rates.append(group_missing)
        
        # Test if missing rates differ significantly between groups
        if len(group_missing_rates) > 1:
            # Chi-square test for independence
            # Simplified: check if variance in missing rates is high
            missing_variance = np.var(group_missing_rates)
            
            if missing_variance > 5:  # >5% variance between groups
                results['group_bias'] = "Yes - missingness varies by group"
            else:
                results['group_bias'] = "No - missingness similar across groups"
            
            results['group_missing_rates'] = dict(zip(unique_groups, group_missing_rates))
    
    return results


def test_normality(
    data: pd.DataFrame,
    sample_columns: List[str],
    test: str = "shapiro",
    sample_size: int = 100
) -> pd.DataFrame:
    """
    Test normality of feature distributions.
    
    Important for choosing appropriate statistical tests and transformations.
    Many parametric tests assume normality.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_columns : list
        Sample columns to test. All must exist in data.columns.
    test : str, optional (default: "shapiro")
        Normality test:
        - "shapiro": Shapiro-Wilk test (recommended for n<50)
        - "anderson": Anderson-Darling test
        - "ks": Kolmogorov-Smirnov test
    sample_size : int, optional (default: 100)
        Maximum number of features to test (for speed)
    
    Returns
    -------
    pd.DataFrame
        Test results with columns:
        - feature: Feature name
        - statistic: Test statistic
        - p_value: P-value
        - is_normal: Boolean (p > 0.05)
    
    Examples
    --------
    >>> normality = test_normality(data, sample_cols, test='shapiro')
    >>> normal_pct = (normality['is_normal'].sum() / len(normality)) * 100
    >>> print(f"{normal_pct:.1f}% of features are normally distributed")
    
    Notes
    -----
    - Shapiro-Wilk most powerful for small samples
    - Anderson-Darling gives more weight to tails
    - Many omics features are NOT normally distributed
    - Consider log transformation if non-normal
    """
    # Validate inputs
    missing_samples = set(sample_columns) - set(data.columns)
    if missing_samples:
        raise ValueError(
            f"Sample(s) not found in data.columns: {missing_samples}. "
            f"All sample_columns must exist in data.columns. "
            f"Available columns: {list(data.columns)[:10]}..."
        )
    
    # Sample features if too many
    if len(data) > sample_size:
        sampled_features = np.random.choice(data.index, sample_size, replace=False)
        test_data = data.loc[sampled_features]
    else:
        test_data = data
    
    results = []
    
    for feature in test_data.index:
        values = data.loc[feature, sample_columns].values
        values_clean = values[~np.isnan(values)]
        
        if len(values_clean) < 3:
            continue
        
        try:
            if test == "shapiro":
                stat, pval = stats.shapiro(values_clean)
            elif test == "anderson":
                result = stats.anderson(values_clean)
                stat = result.statistic
                # Use 5% significance level
                pval = 0.05 if stat > result.critical_values[2] else 0.1
            elif test == "ks":
                stat, pval = stats.kstest(values_clean, 'norm')
            else:
                raise ValueError(f"Unknown test: {test}")
            
            results.append({
                'feature': feature,
                'statistic': stat,
                'p_value': pval,
                'is_normal': pval > 0.05
            })
        except:
            continue
    
    return pd.DataFrame(results)

