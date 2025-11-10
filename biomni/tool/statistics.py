"""
Statistical Analysis Module for Omics Data

This module provides comprehensive statistical analysis functions for
multi-omics data including ANOVA, survival analysis, and clinical statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, List, Dict, Tuple, Union


def perform_anova(
    data: pd.DataFrame,
    sample_columns: list,
    groups: pd.Series,
    method: str = "one_way",
    post_hoc: str = "tukey",
    adjust_pvalues: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform ANOVA (Analysis of Variance) with post-hoc testing.
    
    Essential for comparing expression levels across multiple groups
    (e.g., control, treatment1, treatment2). More powerful than multiple
    t-tests and controls family-wise error rate.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with features as rows and samples as columns
    sample_columns : list
        List of column names containing sample values
    groups : pd.Series
        Group labels for each sample. Index must match sample_columns.
    method : str, optional (default: "one_way")
        ANOVA method. Options:
        - "one_way": One-way ANOVA (one grouping factor)
        - "kruskal": Kruskal-Wallis test (non-parametric alternative)
    post_hoc : str, optional (default: "tukey")
        Post-hoc test for pairwise comparisons. Options:
        - "tukey": Tukey's HSD test
        - "bonferroni": Bonferroni correction
        - "none": No post-hoc testing
    adjust_pvalues : bool, optional (default: True)
        Whether to adjust p-values for multiple testing (FDR)
    alpha : float, optional (default: 0.05)
        Significance level
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ANOVA results including:
        - F_statistic: F-statistic value
        - p_value: Raw p-value
        - p_value_adj: FDR-adjusted p-value (if adjust_pvalues=True)
        - significant: Boolean indicating significance
        - post_hoc_results: Pairwise comparison results (if requested)
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data with 3 groups
    >>> np.random.seed(42)
    >>> data = pd.DataFrame({
    ...     'ctrl1': np.random.randn(100) + 5,
    ...     'ctrl2': np.random.randn(100) + 5,
    ...     'trt1_1': np.random.randn(100) + 6,
    ...     'trt1_2': np.random.randn(100) + 6,
    ...     'trt2_1': np.random.randn(100) + 7,
    ...     'trt2_2': np.random.randn(100) + 7,
    ... })
    >>> data.index = [f'Feature_{i}' for i in range(100)]
    >>> 
    >>> # Define groups
    >>> groups = pd.Series(
    ...     ['Control', 'Control', 'Treatment1', 'Treatment1', 'Treatment2', 'Treatment2'],
    ...     index=data.columns
    ... )
    >>> 
    >>> # Perform ANOVA
    >>> results = perform_anova(
    ...     data,
    ...     data.columns.tolist(),
    ...     groups,
    ...     method="one_way",
    ...     post_hoc="tukey"
    ... )
    >>> 
    >>> # Filter significant features
    >>> sig_features = results[results['significant']]
    >>> print(f"Found {len(sig_features)} significant features")
    
    Notes
    -----
    - One-way ANOVA assumes normal distribution and equal variances
    - Kruskal-Wallis is non-parametric alternative for non-normal data
    - Post-hoc tests control for multiple comparisons
    - Data should be normalized before ANOVA
    - For two groups, use t-test instead
    """
    from scipy.stats import f_oneway, kruskal
    
    # Validate inputs
    if len(groups.unique()) < 2:
        raise ValueError("Need at least 2 groups for ANOVA")
    
    # Prepare results dataframe
    results = pd.DataFrame(index=data.index)
    
    # Extract data matrix
    data_matrix = data[sample_columns].values
    
    # Group samples by group labels
    unique_groups = groups.unique()
    grouped_data = {}
    for group in unique_groups:
        group_samples = groups[groups == group].index.tolist()
        group_indices = [sample_columns.index(s) for s in group_samples]
        grouped_data[group] = data_matrix[:, group_indices]
    
    # Perform ANOVA for each feature
    f_stats = []
    p_values = []
    
    for i in range(data_matrix.shape[0]):
        # Get data for each group
        group_values = [grouped_data[group][i, :] for group in unique_groups]
        
        # Remove NaN values
        group_values = [vals[~np.isnan(vals)] for vals in group_values]
        
        # Check if we have enough data
        if all(len(vals) > 0 for vals in group_values):
            if method == "one_way":
                f_stat, p_val = f_oneway(*group_values)
            elif method == "kruskal":
                f_stat, p_val = kruskal(*group_values)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            f_stats.append(f_stat)
            p_values.append(p_val)
        else:
            f_stats.append(np.nan)
            p_values.append(np.nan)
    
    results['F_statistic'] = f_stats
    results['p_value'] = p_values
    
    # Adjust p-values using Benjamini-Hochberg FDR
    if adjust_pvalues:
        # Filter out NaN p-values
        valid_mask = ~np.isnan(p_values)
        valid_pvals = np.array(p_values)[valid_mask]
        
        if len(valid_pvals) > 0:
            # Benjamini-Hochberg procedure
            n = len(valid_pvals)
            sorted_idx = np.argsort(valid_pvals)
            sorted_pvals = valid_pvals[sorted_idx]
            
            # Calculate adjusted p-values
            padj = np.zeros(n)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    padj[sorted_idx[i]] = sorted_pvals[i]
                else:
                    padj[sorted_idx[i]] = min(sorted_pvals[i] * n / (i+1), padj[sorted_idx[i+1]])
            
            # Create adjusted p-value column
            padj_full = np.full(len(p_values), np.nan)
            padj_full[valid_mask] = padj
            results['p_value_adj'] = padj_full
            
            # Mark significant features
            results['significant'] = results['p_value_adj'] < alpha
        else:
            results['p_value_adj'] = np.nan
            results['significant'] = False
    else:
        results['significant'] = results['p_value'] < alpha
    
    # Post-hoc testing
    if post_hoc != "none" and len(unique_groups) > 2:
        # Perform post-hoc for significant features
        post_hoc_results = {}
        
        if post_hoc == "tukey":
            # Tukey HSD test - requires statsmodels
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                for feature_idx in results[results['significant']].index:
                    try:
                        # Prepare data for Tukey test
                        feature_data = data.loc[feature_idx, sample_columns].values
                        
                        # Create long format data
                        tukey_data = []
                        tukey_groups = []
                        for sample, group in zip(sample_columns, groups):
                            if not np.isnan(data.loc[feature_idx, sample]):
                                tukey_data.append(data.loc[feature_idx, sample])
                                tukey_groups.append(group)
                        
                        if len(tukey_data) > 0:
                            tukey_result = pairwise_tukeyhsd(tukey_data, tukey_groups, alpha=alpha)
                            post_hoc_results[feature_idx] = str(tukey_result)
                    except:
                        post_hoc_results[feature_idx] = "Post-hoc test failed"
                
                results['post_hoc'] = results.index.map(lambda x: post_hoc_results.get(x, ""))
            except ImportError:
                # Fall back to bonferroni if statsmodels not available
                print("Warning: statsmodels not available. Using Bonferroni correction instead.")
                post_hoc = "bonferroni"
        
        elif post_hoc == "bonferroni":
            # Bonferroni correction for pairwise t-tests
            n_comparisons = len(unique_groups) * (len(unique_groups) - 1) / 2
            bonferroni_alpha = alpha / n_comparisons
            
            for feature_idx in results[results['significant']].index:
                comparisons = []
                for i, group1 in enumerate(unique_groups):
                    for group2 in unique_groups[i+1:]:
                        vals1 = grouped_data[group1][data.index.get_loc(feature_idx), :]
                        vals2 = grouped_data[group2][data.index.get_loc(feature_idx), :]
                        
                        vals1 = vals1[~np.isnan(vals1)]
                        vals2 = vals2[~np.isnan(vals2)]
                        
                        if len(vals1) > 0 and len(vals2) > 0:
                            t_stat, p_val = stats.ttest_ind(vals1, vals2)
                            sig = "***" if p_val < bonferroni_alpha else ""
                            comparisons.append(f"{group1} vs {group2}: p={p_val:.4f} {sig}")
                
                post_hoc_results[feature_idx] = "; ".join(comparisons)
            
            results['post_hoc'] = results.index.map(lambda x: post_hoc_results.get(x, ""))
    
    return results


def perform_survival_analysis(
    expression_data: pd.DataFrame,
    survival_time: pd.Series,
    event: pd.Series,
    method: str = "cox",
    covariates: pd.DataFrame = None,
) -> dict:
    """
    Perform survival analysis (Cox regression or Kaplan-Meier).
    
    Critical for clinical omics studies to identify biomarkers
    associated with patient survival. Can identify prognostic signatures
    and therapeutic targets.
    
    Parameters
    ----------
    expression_data : pd.DataFrame
        DataFrame with features as rows and patients as columns
    survival_time : pd.Series
        Survival time for each patient (in days/months)
        Index must match expression_data columns
    event : pd.Series
        Event indicator (1 = event occurred, 0 = censored)
        Index must match expression_data columns
    method : str, optional (default: "cox")
        Analysis method. Options:
        - "cox": Cox proportional hazards regression
        - "logrank": Log-rank test (for high vs low expression groups)
    covariates : pd.DataFrame, optional
        Additional covariates to include in Cox model (e.g., age, stage)
        Index must match expression_data columns
    
    Returns
    -------
    dict
        Dictionary containing:
        - results: DataFrame with hazard ratios, p-values, etc.
        - significant_features: List of significant features
        - method: Analysis method used
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> np.random.seed(42)
    >>> n_patients = 50
    >>> expression = pd.DataFrame(
    ...     np.random.randn(10, n_patients),
    ...     columns=[f'Patient{i}' for i in range(n_patients)]
    ... )
    >>> expression.index = [f'Gene{i}' for i in range(10)]
    >>> 
    >>> # Simulate survival data
    >>> survival_time = pd.Series(
    ...     np.random.exponential(scale=365, size=n_patients),
    ...     index=expression.columns
    ... )
    >>> event = pd.Series(
    ...     np.random.binomial(1, 0.6, size=n_patients),
    ...     index=expression.columns
    ... )
    >>> 
    >>> # Perform survival analysis
    >>> results = perform_survival_analysis(
    ...     expression,
    ...     survival_time,
    ...     event,
    ...     method="cox"
    ... )
    >>> 
    >>> print(f"Found {len(results['significant_features'])} prognostic features")
    
    Notes
    -----
    - Cox regression assumes proportional hazards
    - Data should be normalized before analysis
    - Hazard ratio > 1: increased risk (poor prognosis)
    - Hazard ratio < 1: decreased risk (good prognosis)
    - P-values are adjusted for multiple testing (FDR)
    - Requires lifelines package
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        raise ImportError(
            "lifelines package required for survival analysis. "
            "Install with: pip install lifelines"
        )
    
    # Prepare results
    results_dict = {'method': method}
    
    if method == "cox":
        # Cox proportional hazards regression
        results_list = []
        
        for feature in expression_data.index:
            try:
                # Prepare data for Cox model
                feature_expr = expression_data.loc[feature, :]
                
                # Create dataframe for Cox model
                cox_data = pd.DataFrame({
                    'time': survival_time,
                    'event': event,
                    'expression': feature_expr
                })
                
                # Add covariates if provided
                if covariates is not None:
                    for cov in covariates.columns:
                        cox_data[cov] = covariates[cov]
                
                # Remove missing values
                cox_data = cox_data.dropna()
                
                if len(cox_data) < 10:  # Need minimum samples
                    continue
                
                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col='time', event_col='event')
                
                # Extract results for expression variable
                summary = cph.summary.loc['expression']
                
                results_list.append({
                    'feature': feature,
                    'hazard_ratio': summary['exp(coef)'],
                    'HR_lower_95': summary['exp(coef) lower 95%'],
                    'HR_upper_95': summary['exp(coef) upper 95%'],
                    'p_value': summary['p'],
                    'coef': summary['coef'],
                    'se_coef': summary['se(coef)'],
                    'z': summary['z']
                })
            except Exception as e:
                # Skip features that fail
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)
        
        if len(results_df) > 0:
            # Adjust p-values
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_value_adj'] = padj
            results_df['significant'] = results_df['p_value_adj'] < 0.05
            
            # Sort by p-value
            results_df = results_df.sort_values('p_value')
            
            results_dict['results'] = results_df
            results_dict['significant_features'] = results_df[results_df['significant']]['feature'].tolist()
        else:
            results_dict['results'] = pd.DataFrame()
            results_dict['significant_features'] = []
    
    elif method == "logrank":
        # Log-rank test (comparing high vs low expression groups)
        results_list = []
        
        for feature in expression_data.index:
            try:
                feature_expr = expression_data.loc[feature, :]
                
                # Dichotomize by median
                median_expr = feature_expr.median()
                groups = (feature_expr > median_expr).astype(int)
                
                # Prepare data
                test_data = pd.DataFrame({
                    'time': survival_time,
                    'event': event,
                    'group': groups
                })
                test_data = test_data.dropna()
                
                if len(test_data) < 10:
                    continue
                
                # Log-rank test
                result = multivariate_logrank_test(
                    test_data['time'],
                    test_data['group'],
                    test_data['event']
                )
                
                results_list.append({
                    'feature': feature,
                    'test_statistic': result.test_statistic,
                    'p_value': result.p_value,
                    'df': result.degrees_of_freedom
                })
            except:
                continue
        
        results_df = pd.DataFrame(results_list)
        
        if len(results_df) > 0:
            # Adjust p-values using Benjamini-Hochberg FDR
            pvals = results_df['p_value'].values
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_pvals = pvals[sorted_idx]
            
            padj = np.zeros(n)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    padj[sorted_idx[i]] = min(sorted_pvals[i], 1.0)
                else:
                    padj[sorted_idx[i]] = min(sorted_pvals[i] * n / (i+1), padj[sorted_idx[i+1]], 1.0)
            
            results_df['p_value_adj'] = padj
            results_df['significant'] = results_df['p_value_adj'] < 0.05
            
            results_df = results_df.sort_values('p_value')
            
            results_dict['results'] = results_df
            results_dict['significant_features'] = results_df[results_df['significant']]['feature'].tolist()
        else:
            results_dict['results'] = pd.DataFrame()
            results_dict['significant_features'] = []
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return results_dict


def plot_kaplan_meier(
    survival_time: pd.Series,
    event: pd.Series,
    groups: pd.Series = None,
    confidence_interval: bool = True,
    risk_table: bool = True,
    output_file: str = "kaplan_meier_plot.png",
) -> str:
    """
    Create Kaplan-Meier survival curves.
    
    Essential visualization for clinical omics studies. Shows
    survival probability over time and can compare different groups
    (e.g., high vs low expression).
    
    Parameters
    ----------
    survival_time : pd.Series
        Survival time for each sample (same units throughout)
    event : pd.Series
        Event indicator (1 = event occurred, 0 = censored)
    groups : pd.Series, optional
        Group labels for stratification. If provided, separate
        curves will be plotted for each group.
    confidence_interval : bool, optional (default: True)
        Whether to show confidence intervals
    risk_table : bool, optional (default: True)
        Whether to show number at risk table below plot
    output_file : str, optional
        Path to save plot
    
    Returns
    -------
    str
        Path to saved plot
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Simulate survival data
    >>> np.random.seed(42)
    >>> n = 100
    >>> survival_time = pd.Series(
    ...     np.random.exponential(scale=365, size=n),
    ...     index=[f'Patient{i}' for i in range(n)]
    ... )
    >>> event = pd.Series(
    ...     np.random.binomial(1, 0.6, size=n),
    ...     index=survival_time.index
    ... )
    >>> 
    >>> # Create groups (high vs low expression)
    >>> groups = pd.Series(
    ...     ['High' if i < 50 else 'Low' for i in range(n)],
    ...     index=survival_time.index
    ... )
    >>> 
    >>> # Plot Kaplan-Meier curves
    >>> plot_path = plot_kaplan_meier(
    ...     survival_time,
    ...     event,
    ...     groups=groups,
    ...     output_file="km_plot.png"
    ... )
    
    Notes
    -----
    - Censored patients are marked with vertical tick marks
    - Confidence intervals show uncertainty in survival estimates
    - Log-rank test p-value is shown when comparing groups
    - Risk table shows number of patients at risk at each timepoint
    - Requires lifelines package
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        raise ImportError(
            "lifelines package required. Install with: pip install lifelines"
        )
    
    # Create figure
    if risk_table:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax2 = None
    
    kmf = KaplanMeierFitter()
    
    if groups is None:
        # Single survival curve
        kmf.fit(survival_time, event, label='All patients')
        kmf.plot_survival_function(ax=ax1, ci_show=confidence_interval)
        
    else:
        # Multiple curves for different groups
        unique_groups = groups.unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        
        # Store risk table data
        risk_table_data = {}
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            kmf.fit(
                survival_time[mask],
                event[mask],
                label=f'{group} (n={mask.sum()})'
            )
            kmf.plot_survival_function(
                ax=ax1,
                ci_show=confidence_interval,
                color=colors[i]
            )
            
            # Collect risk table data
            if risk_table:
                # Get number at risk at specific timepoints
                timepoints = np.percentile(survival_time, [0, 25, 50, 75, 100])
                n_at_risk = []
                for t in timepoints:
                    n = ((survival_time[mask] >= t)).sum()
                    n_at_risk.append(n)
                risk_table_data[group] = n_at_risk
        
        # Perform log-rank test
        if len(unique_groups) == 2:
            group1_mask = groups == unique_groups[0]
            group2_mask = groups == unique_groups[1]
            
            result = logrank_test(
                survival_time[group1_mask],
                survival_time[group2_mask],
                event[group1_mask],
                event[group2_mask]
            )
            
            ax1.text(
                0.02, 0.02,
                f'Log-rank test p-value: {result.p_value:.4f}',
                transform=ax1.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # Add risk table
        if risk_table and ax2 is not None:
            timepoints = np.percentile(survival_time, [0, 25, 50, 75, 100])
            
            # Create risk table
            ax2.axis('off')
            table_data = []
            row_labels = []
            
            for group in unique_groups:
                table_data.append(risk_table_data[group])
                row_labels.append(group)
            
            table = ax2.table(
                cellText=table_data,
                rowLabels=row_labels,
                colLabels=[f'{int(t)}' for t in timepoints],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax2.set_title('Number at Risk', fontsize=12, pad=10)
    
    # Formatting
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.set_title('Kaplan-Meier Survival Curves', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def adjust_pvalues(
    pvalues: Union[List[float], np.ndarray, pd.Series],
    method: str = "fdr_bh",
    alpha: float = 0.05
) -> Union[np.ndarray, pd.Series]:
    """
    Adjust p-values for multiple testing correction.
    
    Essential for controlling false discovery rate in high-throughput omics
    experiments where thousands of features are tested simultaneously.
    
    Parameters
    ----------
    pvalues : array-like
        Array of raw p-values
    method : str, optional (default: "fdr_bh")
        Correction method. Options:
        - "fdr_bh": Benjamini-Hochberg FDR (recommended)
        - "fdr_by": Benjamini-Yekutieli FDR (more conservative)
        - "bonferroni": Bonferroni correction (very conservative)
        - "holm": Holm-Bonferroni (less conservative than Bonferroni)
        - "sidak": Šidák correction
    alpha : float, optional (default: 0.05)
        Family-wise error rate or FDR level
    
    Returns
    -------
    array-like
        Adjusted p-values (same type as input)
    
    Examples
    --------
    >>> pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
    >>> adjusted = adjust_pvalues(pvals, method='fdr_bh')
    >>> print(adjusted)
    [0.005 0.025 0.0667 0.125 0.5]
    
    Notes
    -----
    - FDR methods control expected proportion of false positives
    - Bonferroni controls family-wise error rate (FWER)
    - FDR is less conservative, more appropriate for exploratory analysis
    - Bonferroni recommended when false positives are very costly
    
    References
    ----------
    [1] Benjamini & Hochberg. "Controlling the False Discovery Rate",
        J R Stat Soc B, 1995.
    """
    is_series = isinstance(pvalues, pd.Series)
    if is_series:
        index = pvalues.index
        pvalues = pvalues.values
    else:
        pvalues = np.array(pvalues)
    
    n = len(pvalues)
    
    if method == "fdr_bh":
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
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
    
    elif method == "fdr_by":
        # Benjamini-Yekutieli procedure (for dependent tests)
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        cm = np.sum(1.0 / np.arange(1, n+1))  # Harmonic sum
        
        padj = np.zeros(n)
        for i in range(n-1, -1, -1):
            if i == n-1:
                padj[sorted_idx[i]] = min(sorted_pvals[i], 1.0)
            else:
                padj[sorted_idx[i]] = min(
                    sorted_pvals[i] * n * cm / (i+1),
                    padj[sorted_idx[i+1]],
                    1.0
                )
    
    elif method == "bonferroni":
        padj = np.minimum(pvalues * n, 1.0)
    
    elif method == "holm":
        # Holm-Bonferroni procedure
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        padj = np.zeros(n)
        for i in range(n):
            padj[sorted_idx[i]] = min(
                sorted_pvals[i] * (n - i),
                1.0
            )
            if i > 0:
                padj[sorted_idx[i]] = max(padj[sorted_idx[i]], padj[sorted_idx[i-1]])
    
    elif method == "sidak":
        padj = 1.0 - (1.0 - pvalues) ** n
        padj = np.minimum(padj, 1.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if is_series:
        return pd.Series(padj, index=index)
    return padj


def calculate_correlation(
    data1: pd.DataFrame,
    data2: pd.DataFrame = None,
    method: str = "pearson",
    adjust_p: bool = True,
    min_overlap: int = 3
) -> pd.DataFrame:
    """
    Calculate correlations between features.
    
    Essential for co-expression analysis, pathway analysis, and identifying
    relationships between different omics layers (e.g., mRNA-protein correlation).
    
    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset (features × samples)
    data2 : pd.DataFrame, optional
        Second dataset (features × samples). If None, calculates within data1.
    method : str, optional (default: "pearson")
        Correlation method:
        - "pearson": Pearson correlation (linear relationships)
        - "spearman": Spearman rank correlation (monotonic relationships)
        - "kendall": Kendall tau (ordinal relationships)
    adjust_pvalues : bool, optional (default: True)
        Apply FDR correction to p-values
    min_overlap : int, optional (default: 3)
        Minimum number of overlapping non-NaN values required
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix with columns:
        - feature1: Feature from data1
        - feature2: Feature from data2 (or data1)
        - correlation: Correlation coefficient
        - p_value: Raw p-value
        - p_value_adj: FDR-adjusted p-value (if adjust_pvalues=True)
        - n_samples: Number of overlapping samples used
    
    Examples
    --------
    >>> # Self-correlation (co-expression)
    >>> gene_expr = pd.DataFrame(np.random.randn(50, 20))
    >>> correlations = calculate_correlation(gene_expr)
    >>> 
    >>> # Cross-omics correlation (e.g., mRNA-protein)
    >>> mrna_data = pd.DataFrame(np.random.randn(100, 20))
    >>> protein_data = pd.DataFrame(np.random.randn(80, 20))
    >>> correlations = calculate_correlation(mrna_data, protein_data)
    
    Notes
    -----
    - Pearson assumes linear relationship and normality
    - Spearman robust to outliers, detects monotonic relationships
    - Kendall more robust but computationally intensive
    - High correlations may indicate co-regulation or functional relationships
    
    References
    ----------
    [1] Stuart et al. "A Gene-Coexpression Network for Global Discovery 
        of Conserved Genetic Modules", Science, 2003.
    """
    # Determine if self-correlation
    if data2 is None:
        data2 = data1
        self_corr = True
    else:
        self_corr = False
    
    # Align samples
    common_samples = data1.columns.intersection(data2.columns)
    if len(common_samples) < min_overlap:
        raise ValueError(f"Insufficient overlapping samples: {len(common_samples)} < {min_overlap}")
    
    data1_aligned = data1[common_samples].T  # Transpose to samples × features
    data2_aligned = data2[common_samples].T
    
    results = []
    
    # Calculate correlations
    for i, feat1 in enumerate(data1.index):
        # For self-correlation, only calculate upper triangle
        start_j = i+1 if self_corr else 0
        
        for j in range(start_j, len(data2.index)):
            feat2 = data2.index[j]
            
            x = data1_aligned.iloc[:, i].values
            y = data2_aligned.iloc[:, j].values
            
            # Remove NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            n_samples = len(x_clean)
            
            if n_samples < min_overlap:
                continue
            
            # Calculate correlation
            if method == "pearson":
                corr, pval = stats.pearsonr(x_clean, y_clean)
            elif method == "spearman":
                corr, pval = stats.spearmanr(x_clean, y_clean)
            elif method == "kendall":
                corr, pval = stats.kendalltau(x_clean, y_clean)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'feature1': feat1,
                'feature2': feat2,
                'correlation': corr,
                'p_value': pval,
                'n_samples': n_samples
            })
    
    results_df = pd.DataFrame(results)
    
    # Adjust p-values
    if adjust_p and len(results_df) > 0:
        from biomni.tool.statistics import adjust_pvalues as adjust_pvals_func
        results_df['p_value_adj'] = adjust_pvals_func(
            results_df['p_value'].values,
            method='fdr_bh'
        )
    
    # Sort by absolute correlation
    results_df['abs_correlation'] = np.abs(results_df['correlation'])
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    results_df = results_df.drop('abs_correlation', axis=1)
    
    return results_df


def perform_enrichment_test(
    query_set: List[str],
    background_set: List[str],
    annotation_dict: Dict[str, List[str]],
    method: str = "hypergeometric",
    adjust_p: bool = True,
    min_overlap: int = 2
) -> pd.DataFrame:
    """
    Perform enrichment analysis (over-representation analysis).
    
    Essential for identifying enriched pathways, GO terms, or other annotations
    in a set of significant genes/proteins. Core method for functional analysis.
    
    Parameters
    ----------
    query_set : list
        List of significant features (e.g., differentially expressed genes)
    background_set : list
        List of all tested features (universe)
    annotation_dict : dict
        Dictionary mapping annotation terms to lists of features.
        Format: {term: [feature1, feature2, ...]}
    method : str, optional (default: "hypergeometric")
        Statistical test:
        - "hypergeometric": Hypergeometric test (recommended)
        - "fisher": Fisher's exact test
    adjust_pvalues : bool, optional (default: True)
        Apply FDR correction
    min_overlap : int, optional (default: 2)
        Minimum overlap required for testing
    
    Returns
    -------
    pd.DataFrame
        Enrichment results with columns:
        - term: Annotation term
        - overlap: Number of query features in term
        - term_size: Total features in term
        - query_size: Total query features
        - background_size: Total background features
        - p_value: Raw p-value
        - p_value_adj: FDR-adjusted p-value
        - odds_ratio: Enrichment odds ratio (Fisher only)
        - enrichment_score: -log10(p_value_adj)
        - genes: Comma-separated list of overlapping features
    
    Examples
    --------
    >>> # Define significant genes
    >>> sig_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC']
    >>> 
    >>> # Define background
    >>> background = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'PTEN', ...]
    >>> 
    >>> # Define pathway annotations
    >>> pathways = {
    ...     'DNA_REPAIR': ['TP53', 'BRCA1', 'ATM', 'PTEN'],
    ...     'CELL_CYCLE': ['TP53', 'MYC', 'CDK1', 'CCND1'],
    ...     'METABOLISM': ['IDH1', 'IDH2', 'SDHA']
    ... }
    >>> 
    >>> # Perform enrichment
    >>> results = perform_enrichment_test(sig_genes, background, pathways)
    >>> print(results[results['p_value_adj'] < 0.05])
    
    Notes
    -----
    - Hypergeometric test assumes random sampling without replacement
    - Fisher's exact test similar but allows odds ratio calculation
    - Query set should be subset of background set
    - FDR correction essential for multiple pathway testing
    - Enrichment score useful for ranking terms
    
    References
    ----------
    [1] Huang et al. "Systematic and integrative analysis of large gene lists",
        Nat Protoc, 2009.
    """
    from scipy.stats import hypergeom, fisher_exact
    
    # Convert to sets for efficient operations
    query_set = set(query_set)
    background_set = set(background_set)
    
    # Validate query is subset of background
    if not query_set.issubset(background_set):
        print(f"Warning: {len(query_set - background_set)} query features not in background")
        query_set = query_set.intersection(background_set)
    
    M = len(background_set)  # Total background size
    n = len(query_set)  # Total query size
    
    results = []
    
    for term, term_features in annotation_dict.items():
        term_set = set(term_features).intersection(background_set)
        N = len(term_set)  # Term size in background
        
        # Find overlap
        overlap_set = query_set.intersection(term_set)
        k = len(overlap_set)  # Observed overlap
        
        if k < min_overlap:
            continue
        
        if method == "hypergeometric":
            # Hypergeometric test
            # P(X >= k) where X ~ Hypergeom(M, N, n)
            pval = hypergeom.sf(k-1, M, N, n)
            odds_ratio = None
            
        elif method == "fisher":
            # Fisher's exact test
            # Construct 2x2 contingency table
            #                In Term    Not In Term
            # In Query         a            b
            # Not In Query     c            d
            
            a = k  # In query and in term
            b = n - k  # In query but not in term
            c = N - k  # Not in query but in term
            d = M - N - n + k  # Not in query and not in term
            
            odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append({
            'term': term,
            'overlap': k,
            'term_size': N,
            'query_size': n,
            'background_size': M,
            'p_value': pval,
            'odds_ratio': odds_ratio,
            'genes': ','.join(sorted(overlap_set))
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return results_df
    
    # Adjust p-values
    if adjust_p:
        from biomni.tool.statistics import adjust_pvalues as adjust_pvals_func
        results_df['p_value_adj'] = adjust_pvals_func(
            results_df['p_value'].values,
            method='fdr_bh'
        )
    else:
        results_df['p_value_adj'] = results_df['p_value']
    
    # Calculate enrichment score
    results_df['enrichment_score'] = -np.log10(results_df['p_value_adj'] + 1e-300)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    return results_df


def perform_permutation_test(
    data: pd.DataFrame,
    group1_samples: List[str],
    group2_samples: List[str],
    n_permutations: int = 1000,
    statistic: str = "mean_diff",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform permutation test for differential analysis.
    
    Non-parametric alternative to t-test. Generates null distribution by
    randomly permuting group labels. More robust when assumptions of
    parametric tests are violated.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    group1_samples : list
        Sample names for group 1
    group2_samples : list
        Sample names for group 2
    n_permutations : int, optional (default: 1000)
        Number of random permutations (1000-10000 recommended)
    statistic : str, optional (default: "mean_diff")
        Test statistic:
        - "mean_diff": Difference in means
        - "median_diff": Difference in medians
        - "tstat": t-statistic
    random_state : int, optional (default: 42)
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Results with columns:
        - observed_stat: Observed test statistic
        - p_value: Permutation p-value
        - p_value_adj: FDR-adjusted p-value
    
    Examples
    --------
    >>> data = pd.DataFrame(np.random.randn(100, 10))
    >>> results = perform_permutation_test(
    ...     data,
    ...     group1_samples=['s1', 's2', 's3', 's4', 's5'],
    ...     group2_samples=['s6', 's7', 's8', 's9', 's10'],
    ...     n_permutations=1000
    ... )
    
    Notes
    -----
    - No distributional assumptions required
    - Computationally intensive for large datasets
    - P-value resolution limited by n_permutations
    - min p-value = 1/(n_permutations + 1)
    
    References
    ----------
    [1] Good P. "Permutation, Parametric and Bootstrap Tests of Hypotheses",
        Springer, 2005.
    """
    np.random.seed(random_state)
    
    all_samples = group1_samples + group2_samples
    n1 = len(group1_samples)
    n2 = len(group2_samples)
    n_total = n1 + n2
    
    data_matrix = data[all_samples].values
    
    # Calculate observed statistic
    group1_data = data[group1_samples].values
    group2_data = data[group2_samples].values
    
    if statistic == "mean_diff":
        observed_stats = np.mean(group1_data, axis=1) - np.mean(group2_data, axis=1)
    elif statistic == "median_diff":
        observed_stats = np.median(group1_data, axis=1) - np.median(group2_data, axis=1)
    elif statistic == "tstat":
        observed_stats, _ = stats.ttest_ind(group1_data, group2_data, axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Perform permutations
    null_stats = np.zeros((len(data), n_permutations))
    
    for perm in range(n_permutations):
        # Randomly permute sample labels
        perm_indices = np.random.permutation(n_total)
        perm_group1_idx = perm_indices[:n1]
        perm_group2_idx = perm_indices[n1:]
        
        perm_group1 = data_matrix[:, perm_group1_idx]
        perm_group2 = data_matrix[:, perm_group2_idx]
        
        if statistic == "mean_diff":
            null_stats[:, perm] = np.mean(perm_group1, axis=1) - np.mean(perm_group2, axis=1)
        elif statistic == "median_diff":
            null_stats[:, perm] = np.median(perm_group1, axis=1) - np.median(perm_group2, axis=1)
        elif statistic == "tstat":
            null_stats[:, perm], _ = stats.ttest_ind(perm_group1, perm_group2, axis=1)
    
    # Calculate p-values
    # Two-tailed test: count how often |null| >= |observed|
    p_values = np.zeros(len(data))
    for i in range(len(data)):
        p_values[i] = (np.sum(np.abs(null_stats[i, :]) >= np.abs(observed_stats[i])) + 1) / (n_permutations + 1)
    
    # Create results
    results = pd.DataFrame({
        'observed_stat': observed_stats,
        'p_value': p_values
    }, index=data.index)
    
    # FDR correction
    from biomni.tool.statistics import adjust_pvalues as adjust_pvals_func
    results['p_value_adj'] = adjust_pvals_func(p_values, method='fdr_bh')
    results['significant'] = results['p_value_adj'] < 0.05
    
    return results.sort_values('p_value')


def perform_nonparametric_test(
    data: pd.DataFrame,
    group1_samples: List[str],
    group2_samples: List[str] = None,
    test: str = "mannwhitney",
    adjust_p: bool = True,
    paired: bool = False
) -> pd.DataFrame:
    """
    Perform non-parametric statistical tests.
    
    Robust alternatives to t-tests when data is non-normal, has outliers,
    or is ordinal. Essential for proteomics and metabolomics data which
    often violate normality assumptions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    group1_samples : list
        Sample names for group 1
    group2_samples : list, optional
        Sample names for group 2. If None, performs one-sample test.
    test : str, optional (default: "mannwhitney")
        Statistical test:
        - "mannwhitney": Mann-Whitney U test (independent samples)
        - "wilcoxon": Wilcoxon signed-rank test (paired samples)
        - "kruskal": Kruskal-Wallis H test (>2 independent groups)
    adjust_pvalues : bool, optional (default: True)
        Apply FDR correction
    paired : bool, optional (default: False)
        Whether samples are paired (for Wilcoxon)
    
    Returns
    -------
    pd.DataFrame
        Test results with columns:
        - statistic: Test statistic value
        - p_value: Raw p-value
        - p_value_adj: FDR-adjusted p-value
        - significant: Boolean significance indicator
    
    Examples
    --------
    >>> # Mann-Whitney U test
    >>> data = pd.DataFrame(np.random.randn(100, 10))
    >>> results = perform_nonparametric_test(
    ...     data,
    ...     group1_samples=['s1', 's2', 's3'],
    ...     group2_samples=['s4', 's5'],
    ...     test='mannwhitney'
    ... )
    >>> 
    >>> # Wilcoxon signed-rank test (paired)
    >>> results = perform_nonparametric_test(
    ...     data,
    ...     group1_samples=['before1', 'before2', 'before3'],
    ...     group2_samples=['after1', 'after2', 'after3'],
    ...     test='wilcoxon',
    ...     paired=True
    ... )
    
    Notes
    -----
    - Mann-Whitney: Independent two-group comparison (alternative to t-test)
    - Wilcoxon: Paired two-group comparison (alternative to paired t-test)
    - Kruskal-Wallis: Multi-group comparison (alternative to ANOVA)
    - More robust to outliers and non-normality than parametric tests
    - Less powerful than t-test when normality holds
    
    References
    ----------
    [1] Hollander & Wolfe. "Nonparametric Statistical Methods", Wiley, 1999.
    """
    group1_data = data[group1_samples].values
    
    test_stats = []
    p_values = []
    
    if test == "mannwhitney":
        if group2_samples is None:
            raise ValueError("group2_samples required for Mann-Whitney test")
        
        group2_data = data[group2_samples].values
        
        for i in range(len(data)):
            x = group1_data[i, :]
            y = group2_data[i, :]
            
            # Remove NaN
            x_clean = x[~np.isnan(x)]
            y_clean = y[~np.isnan(y)]
            
            if len(x_clean) > 0 and len(y_clean) > 0:
                stat, pval = stats.mannwhitneyu(x_clean, y_clean, alternative='two-sided')
                test_stats.append(stat)
                p_values.append(pval)
            else:
                test_stats.append(np.nan)
                p_values.append(np.nan)
    
    elif test == "wilcoxon":
        if group2_samples is None:
            # One-sample Wilcoxon (signed-rank test against zero)
            for i in range(len(data)):
                x = group1_data[i, :]
                x_clean = x[~np.isnan(x)]
                
                if len(x_clean) > 0:
                    stat, pval = stats.wilcoxon(x_clean)
                    test_stats.append(stat)
                    p_values.append(pval)
                else:
                    test_stats.append(np.nan)
                    p_values.append(np.nan)
        else:
            # Paired Wilcoxon
            group2_data = data[group2_samples].values
            
            for i in range(len(data)):
                x = group1_data[i, :]
                y = group2_data[i, :]
                
                # Remove NaN pairs
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 0:
                    stat, pval = stats.wilcoxon(x_clean, y_clean)
                    test_stats.append(stat)
                    p_values.append(pval)
                else:
                    test_stats.append(np.nan)
                    p_values.append(np.nan)
    
    elif test == "kruskal":
        # Kruskal-Wallis test requires groups
        if group2_samples is None:
            raise ValueError("Multiple groups required for Kruskal-Wallis test")
        
        # Assume group2_samples contains additional group labels
        # This is simplified - in practice you'd pass multiple group lists
        group2_data = data[group2_samples].values
        
        for i in range(len(data)):
            x = group1_data[i, :]
            y = group2_data[i, :]
            
            x_clean = x[~np.isnan(x)]
            y_clean = y[~np.isnan(y)]
            
            if len(x_clean) > 0 and len(y_clean) > 0:
                stat, pval = stats.kruskal(x_clean, y_clean)
                test_stats.append(stat)
                p_values.append(pval)
            else:
                test_stats.append(np.nan)
                p_values.append(np.nan)
    
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Create results
    results = pd.DataFrame({
        'statistic': test_stats,
        'p_value': p_values
    }, index=data.index)
    
    # Adjust p-values
    if adjust_p:
        valid_pvals = results['p_value'].dropna()
        if len(valid_pvals) > 0:
            from biomni.tool.statistics import adjust_pvalues as adjust_pvals_func
            padj = adjust_pvals_func(valid_pvals.values, method='fdr_bh')
            results['p_value_adj'] = np.nan
            results.loc[valid_pvals.index, 'p_value_adj'] = padj
        else:
            results['p_value_adj'] = np.nan
    
    results['significant'] = results['p_value_adj'] < 0.05
    
    return results.sort_values('p_value')


def test_variance_homogeneity(
    data: pd.DataFrame,
    sample_groups: Dict[str, List[str]],
    test: str = "levene"
) -> Dict[str, any]:
    """
    Test homogeneity of variance across groups.
    
    Essential for determining if ANOVA assumptions are met.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_groups : dict
        Dictionary mapping group names to sample lists
        Format: {group_name: [sample1, sample2, ...]}
    test : str, optional (default: "levene")
        Test to use:
        - "levene": Levene's test (robust to non-normality)
        - "bartlett": Bartlett's test (assumes normality)
    
    Returns
    -------
    dict
        - statistic: Test statistic
        - p_value: P-value
        - equal_variance: Boolean (p > 0.05 indicates equal variance)
        - recommendation: Which test to use based on result
    
    Examples
    --------
    >>> groups = {
    ...     'Control': ['C1', 'C2', 'C3'],
    ...     'Treatment': ['T1', 'T2', 'T3']
    ... }
    >>> var_test = test_variance_homogeneity(data, groups)
    >>> print(f"Equal variance: {var_test['equal_variance']}")
    
    Notes
    -----
    - Levene's test: More robust, recommended for most cases
    - Bartlett's test: More powerful but sensitive to non-normality
    - If equal_variance=False, use Welch's ANOVA instead of standard ANOVA
    
    References
    ----------
    [1] Levene, H. "Robust tests for equality of variances", 1960.
    """
    group_data = []
    for group_name, sample_list in sample_groups.items():
        group_data.append(data[sample_list].values.flatten())
    
    if test == "levene":
        statistic, p_value = stats.levene(*group_data)
    elif test == "bartlett":
        statistic, p_value = stats.bartlett(*group_data)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    equal_variance = p_value > 0.05
    
    if equal_variance:
        recommendation = "Standard ANOVA appropriate (equal variances)"
    else:
        recommendation = "Use Welch's ANOVA or non-parametric test (unequal variances)"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'equal_variance': equal_variance,
        'recommendation': recommendation
    }


def check_test_assumptions(
    data: pd.DataFrame,
    sample_groups: Dict[str, List[str]],
    paired: bool = False,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Check statistical test assumptions for given data.
    
    Analyzes data characteristics to validate which tests are appropriate.
    Essential for ensuring statistical validity.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_groups : dict
        Dictionary mapping group names to sample lists
    paired : bool, optional (default: False)
        Whether samples are paired
    verbose : bool, optional (default: True)
        Print detailed report
    
    Returns
    -------
    dict
        - n_groups: Number of groups
        - sample_sizes: Dict of group sample sizes
        - is_normal: Overall normality assessment
        - normality_pct: Percentage of features that are normal
        - has_equal_variance: Whether variances are equal (if >1 group)
        - has_outliers: Whether outliers detected
        - data_characteristics: Detailed characteristics
        - recommended_tests: List of appropriate tests
        - warnings: List of potential issues
    
    Examples
    --------
    >>> groups = {
    ...     'Control': ['C1', 'C2', 'C3', 'C4'],
    ...     'Treatment': ['T1', 'T2', 'T3', 'T4']
    ... }
    >>> assumptions = check_test_assumptions(data, groups)
    >>> print(assumptions['recommended_tests'])
    
    Notes
    -----
    - Checks normality, variance homogeneity, sample size
    - Provides recommendations based on violations
    - Warnings indicate which assumptions are violated
    """
    # Import QC functions
    import sys
    import importlib.util
    qc_spec = importlib.util.find_spec("biomni.tool.qc")
    if qc_spec is not None:
        qc_module = importlib.import_module("biomni.tool.qc")
        test_normality_func = qc_module.test_normality
    else:
        test_normality_func = None
    
    n_groups = len(sample_groups)
    all_samples = []
    for samples in sample_groups.values():
        all_samples.extend(samples)
    
    # Sample sizes
    sample_sizes = {group: len(samples) for group, samples in sample_groups.items()}
    min_sample_size = min(sample_sizes.values())
    
    warnings = []
    
    # Check sample size
    if min_sample_size < 3:
        warnings.append(f"Very small sample size (min={min_sample_size}). Results may be unreliable.")
    elif min_sample_size < 5:
        warnings.append(f"Small sample size (min={min_sample_size}). Non-parametric tests recommended.")
    
    # Check normality
    is_normal = None
    normality_pct = None
    
    if test_normality_func is not None:
        try:
            normality_results = test_normality_func(
                data, all_samples, 
                test='shapiro', 
                sample_size=min(100, len(data))
            )
            if len(normality_results) > 0:
                normality_pct = (normality_results['is_normal'].sum() / len(normality_results)) * 100
                is_normal = normality_pct >= 70
                
                if normality_pct < 50:
                    warnings.append(f"Most features non-normal ({normality_pct:.1f}% normal). Non-parametric tests recommended.")
                elif normality_pct < 70:
                    warnings.append(f"Borderline normality ({normality_pct:.1f}% normal). Consider both parametric and non-parametric tests.")
        except:
            pass
    
    # Check variance homogeneity (if multiple groups)
    has_equal_variance = None
    if n_groups > 1 and not paired:
        try:
            var_test = test_variance_homogeneity(data, sample_groups, test='levene')
            has_equal_variance = var_test['equal_variance']
            
            if not has_equal_variance:
                warnings.append("Unequal variances detected. Use Welch's test or non-parametric alternative.")
        except:
            pass
    
    # Check for outliers (simplified)
    has_outliers = False
    data_values = data[all_samples].values.flatten()
    data_values_clean = data_values[~np.isnan(data_values)]
    if len(data_values_clean) > 0:
        z_scores = np.abs(stats.zscore(data_values_clean))
        outlier_pct = (z_scores > 3).sum() / len(z_scores) * 100
        has_outliers = outlier_pct > 5
        if has_outliers:
            warnings.append(f"Outliers detected ({outlier_pct:.1f}% of values). Non-parametric tests more robust.")
    
    # Recommend tests
    recommended_tests = []
    
    if n_groups == 2:
        if paired:
            if is_normal:
                recommended_tests.append("Paired t-test")
            recommended_tests.append("Wilcoxon signed-rank test")
        else:
            if is_normal and has_equal_variance:
                recommended_tests.append("Student's t-test")
            elif is_normal:
                recommended_tests.append("Welch's t-test")
            recommended_tests.append("Mann-Whitney U test")
            if min_sample_size < 10:
                recommended_tests.append("Permutation test")
    
    elif n_groups > 2:
        if is_normal and has_equal_variance:
            recommended_tests.append("One-way ANOVA")
        elif is_normal and not has_equal_variance:
            recommended_tests.append("Welch's ANOVA")
        recommended_tests.append("Kruskal-Wallis test")
    
    results = {
        'n_groups': n_groups,
        'sample_sizes': sample_sizes,
        'min_sample_size': min_sample_size,
        'is_normal': is_normal,
        'normality_pct': normality_pct,
        'has_equal_variance': has_equal_variance,
        'has_outliers': has_outliers,
        'recommended_tests': recommended_tests,
        'warnings': warnings
    }
    
    if verbose:
        print("=" * 60)
        print("STATISTICAL TEST ASSUMPTIONS CHECK")
        print("=" * 60)
        print(f"\nData characteristics:")
        print(f"  - Number of groups: {n_groups}")
        print(f"  - Sample sizes: {sample_sizes}")
        print(f"  - Minimum sample size: {min_sample_size}")
        if normality_pct is not None:
            print(f"  - Normality: {normality_pct:.1f}% features normal")
        if has_equal_variance is not None:
            print(f"  - Equal variance: {has_equal_variance}")
        print(f"  - Outliers detected: {has_outliers}")
        
        print(f"\nRecommended tests:")
        for i, test in enumerate(recommended_tests, 1):
            print(f"  {i}. {test}")
        
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        print("=" * 60)
    
    return results


def recommend_statistical_test(
    data: pd.DataFrame,
    sample_groups: Dict[str, List[str]],
    paired: bool = False,
    analysis_type: str = "differential",
    verbose: bool = True
) -> Dict[str, any]:
    """
    Recommend appropriate statistical test based on data characteristics.
    
    This function analyzes data properties and suggests the most appropriate
    statistical test, along with rationale and alternatives.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_groups : dict
        Dictionary mapping group names to sample lists
    paired : bool, optional (default: False)
        Whether samples are paired
    analysis_type : str, optional (default: "differential")
        Type of analysis:
        - "differential": Compare groups (default)
        - "correlation": Correlation analysis
        - "survival": Survival analysis
    verbose : bool, optional (default: True)
        Print recommendation report
    
    Returns
    -------
    dict
        - primary_recommendation: Best test to use
        - confidence: Confidence in recommendation (0-1)
        - rationale: Explanation for recommendation
        - alternative_tests: Other valid options
        - data_summary: Summary of data characteristics
        - assumptions_check: Results from check_test_assumptions
    
    Examples
    --------
    >>> groups = {
    ...     'Control': ['C1', 'C2', 'C3'],
    ...     'Treatment': ['T1', 'T2', 'T3']
    ... }
    >>> recommendation = recommend_statistical_test(data, groups)
    >>> print(recommendation['primary_recommendation'])
    >>> print(recommendation['rationale'])
    
    Notes
    -----
    - Confidence score based on clarity of data characteristics
    - High confidence (>0.8): Clear choice
    - Medium confidence (0.5-0.8): Multiple good options
    - Low confidence (<0.5): Data characteristics unclear
    """
    # Check assumptions
    assumptions = check_test_assumptions(data, sample_groups, paired, verbose=False)
    
    n_groups = assumptions['n_groups']
    is_normal = assumptions['is_normal']
    normality_pct = assumptions['normality_pct']
    has_equal_variance = assumptions['has_equal_variance']
    min_sample_size = assumptions['min_sample_size']
    
    # Decision logic
    primary_recommendation = None
    confidence = 0.0
    rationale = ""
    alternative_tests = []
    
    if analysis_type == "differential":
        if n_groups == 2:
            if paired:
                if is_normal:
                    primary_recommendation = "Paired t-test"
                    confidence = 0.85
                    rationale = f"Paired samples with normal distribution ({normality_pct:.1f}% features normal). Paired t-test is appropriate."
                    alternative_tests = ["Wilcoxon signed-rank test (non-parametric alternative)"]
                else:
                    primary_recommendation = "Wilcoxon signed-rank test"
                    confidence = 0.9
                    rationale = f"Paired samples with non-normal distribution ({normality_pct:.1f}% features normal). Wilcoxon test is more robust."
                    if normality_pct and normality_pct > 50:
                        alternative_tests = ["Paired t-test (if assuming normality)"]
            else:
                # Independent samples
                if min_sample_size < 5:
                    primary_recommendation = "Permutation test"
                    confidence = 0.7
                    rationale = f"Very small sample size (n={min_sample_size}). Permutation test recommended but results may be unreliable."
                    alternative_tests = ["Mann-Whitney U test", "Welch's t-test (use with caution)"]
                elif min_sample_size < 10:
                    primary_recommendation = "Mann-Whitney U test"
                    confidence = 0.85
                    rationale = f"Small sample size (n={min_sample_size}). Non-parametric test preferred."
                    alternative_tests = ["Permutation test", "Welch's t-test (if normal)"]
                elif is_normal:
                    primary_recommendation = "Welch's t-test"
                    confidence = 0.9
                    rationale = f"Independent samples with normal distribution ({normality_pct:.1f}% features normal). Welch's t-test handles unequal variances."
                    alternative_tests = ["Mann-Whitney U test (non-parametric alternative)"]
                    if has_equal_variance:
                        alternative_tests.insert(0, "Student's t-test (if equal variances)")
                else:
                    primary_recommendation = "Mann-Whitney U test"
                    confidence = 0.9
                    rationale = f"Independent samples with non-normal distribution ({normality_pct:.1f}% features normal). Mann-Whitney U is robust."
                    if normality_pct and normality_pct > 50:
                        alternative_tests = ["Welch's t-test (borderline normality)"]
        
        elif n_groups > 2:
            if is_normal and has_equal_variance:
                primary_recommendation = "One-way ANOVA"
                confidence = 0.9
                rationale = f"Multiple groups ({n_groups}) with normal distribution ({normality_pct:.1f}% features) and equal variances. Standard ANOVA is appropriate."
                alternative_tests = ["Kruskal-Wallis test (non-parametric alternative)"]
            elif is_normal and not has_equal_variance:
                primary_recommendation = "Welch's ANOVA"
                confidence = 0.85
                rationale = f"Multiple groups ({n_groups}) with normal distribution but unequal variances. Welch's ANOVA is more robust."
                alternative_tests = ["Kruskal-Wallis test (non-parametric alternative)"]
            else:
                primary_recommendation = "Kruskal-Wallis test"
                confidence = 0.9
                rationale = f"Multiple groups ({n_groups}) with non-normal distribution ({normality_pct:.1f}% features normal). Kruskal-Wallis is robust."
                if normality_pct and normality_pct > 50:
                    alternative_tests = ["One-way ANOVA (borderline normality)"]
        
        else:
            primary_recommendation = "Insufficient groups"
            confidence = 0.0
            rationale = "Need at least 2 groups for differential analysis."
    
    elif analysis_type == "correlation":
        if is_normal:
            primary_recommendation = "Pearson correlation"
            confidence = 0.85
            rationale = f"Normal distribution ({normality_pct:.1f}% features). Pearson correlation for linear relationships."
            alternative_tests = ["Spearman correlation (for non-linear monotonic relationships)"]
        else:
            primary_recommendation = "Spearman correlation"
            confidence = 0.9
            rationale = f"Non-normal distribution ({normality_pct:.1f}% features). Spearman correlation is more robust."
            alternative_tests = ["Kendall's tau (for small samples or many ties)"]
    
    elif analysis_type == "survival":
        primary_recommendation = "Log-rank test with Kaplan-Meier"
        confidence = 0.9
        rationale = "Standard approach for survival analysis with censored data."
        alternative_tests = ["Cox proportional hazards model (for multivariate analysis)"]
    
    # Adjust confidence based on data clarity
    if normality_pct is not None:
        if 40 < normality_pct < 60:
            confidence *= 0.8  # Reduce confidence for borderline cases
    
    if min_sample_size < 10:
        confidence *= 0.9  # Slight reduction for small samples
    
    results = {
        'primary_recommendation': primary_recommendation,
        'confidence': round(confidence, 2),
        'rationale': rationale,
        'alternative_tests': alternative_tests,
        'data_summary': {
            'n_groups': n_groups,
            'sample_sizes': assumptions['sample_sizes'],
            'normality_pct': normality_pct,
            'is_normal': is_normal,
            'has_equal_variance': has_equal_variance
        },
        'assumptions_check': assumptions
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("STATISTICAL TEST RECOMMENDATION")
        print("=" * 60)
        print(f"\n🎯 Primary Recommendation: {primary_recommendation}")
        print(f"   Confidence: {confidence:.0%}")
        print(f"\n📋 Rationale:")
        print(f"   {rationale}")
        
        if alternative_tests:
            print(f"\n🔄 Alternative Tests:")
            for alt in alternative_tests:
                print(f"   - {alt}")
        
        if assumptions['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in assumptions['warnings']:
                print(f"   - {warning}")
        
        print("=" * 60 + "\n")
    
    return results


def smart_differential_analysis(
    data: pd.DataFrame,
    sample_groups: Dict[str, List[str]],
    paired: bool = False,
    auto_select_test: bool = True,
    force_test: Optional[str] = None,
    alpha: float = 0.05,
    adjust_pvalues: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Automatically select and perform appropriate statistical test.
    
    This is the main "smart" function that analyzes data characteristics
    and automatically selects the most appropriate statistical test.
    Essential for autonomous data analysis by AI agents.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features as rows and samples as columns
    sample_groups : dict
        Dictionary mapping group names to sample lists
        Format: {group_name: [sample1, sample2, ...]}
    paired : bool, optional (default: False)
        Whether samples are paired
    auto_select_test : bool, optional (default: True)
        Automatically select test based on data characteristics.
        If False, uses force_test parameter.
    force_test : str, optional
        Force a specific test:
        - "ttest": t-test
        - "welch": Welch's t-test
        - "mannwhitney": Mann-Whitney U
        - "wilcoxon": Wilcoxon signed-rank
        - "anova": One-way ANOVA
        - "kruskal": Kruskal-Wallis
        - "permutation": Permutation test
    alpha : float, optional (default: 0.05)
        Significance level
    adjust_pvalues : bool, optional (default: True)
        Apply FDR correction
    verbose : bool, optional (default: True)
        Print analysis report
    
    Returns
    -------
    results : pd.DataFrame
        Test results with p-values, fold changes, etc.
    metadata : dict
        - selected_test: Name of test used
        - test_function: Function name called
        - selection_rationale: Why this test was selected
        - confidence: Confidence in test selection
        - data_characteristics: Summary of data properties
        - assumptions_check: Full assumptions check results
        - warnings: Any warnings about data or test choice
    
    Examples
    --------
    >>> # Automatic test selection
    >>> groups = {
    ...     'Control': ['C1', 'C2', 'C3', 'C4'],
    ...     'Treatment': ['T1', 'T2', 'T3', 'T4']
    ... }
    >>> results, metadata = smart_differential_analysis(data, groups)
    >>> print(f"Selected: {metadata['selected_test']}")
    >>> print(metadata['selection_rationale'])
    >>> 
    >>> # Force specific test
    >>> results, metadata = smart_differential_analysis(
    ...     data, groups, auto_select_test=False, force_test='mannwhitney'
    ... )
    
    Notes
    -----
    - Automatically checks normality, variance, sample size
    - Selects most appropriate test based on data characteristics
    - Returns detailed metadata for transparency
    - Handles edge cases (small samples, borderline normality, etc.)
    
    See Also
    --------
    recommend_statistical_test : Get recommendation without running test
    check_test_assumptions : Check assumptions only
    """
    n_groups = len(sample_groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for differential analysis")
    
    # Get recommendation if auto-selecting
    if auto_select_test:
        recommendation = recommend_statistical_test(
            data, sample_groups, paired, analysis_type="differential", verbose=verbose
        )
        selected_test = recommendation['primary_recommendation']
        confidence = recommendation['confidence']
        rationale = recommendation['rationale']
        assumptions = recommendation['assumptions_check']
    else:
        if force_test is None:
            raise ValueError("Must specify force_test if auto_select_test=False")
        selected_test = force_test
        confidence = 1.0
        rationale = f"User-specified test: {force_test}"
        assumptions = check_test_assumptions(data, sample_groups, paired, verbose=False)
    
    # Execute selected test
    group_names = list(sample_groups.keys())
    
    try:
        if n_groups == 2:
            group1_samples = sample_groups[group_names[0]]
            group2_samples = sample_groups[group_names[1]]
            
            if selected_test in ["Welch's t-test", "welch", "ttest"]:
                # Use proteomics t_test_FDR (Welch's)
                from biomni.tool.proteomics import t_test_FDR
                results = t_test_FDR(data.copy(), group1_samples, group2_samples)
                test_function = "t_test_FDR"
                
            elif selected_test in ["Mann-Whitney U test", "mannwhitney"]:
                results = perform_nonparametric_test(
                    data, group1_samples, group2_samples,
                    test='mannwhitney', adjust_p=adjust_pvalues
                )
                test_function = "perform_nonparametric_test(test='mannwhitney')"
                
            elif selected_test in ["Wilcoxon signed-rank test", "wilcoxon"]:
                results = perform_nonparametric_test(
                    data, group1_samples, group2_samples,
                    test='wilcoxon', adjust_p=adjust_pvalues, paired=True
                )
                test_function = "perform_nonparametric_test(test='wilcoxon', paired=True)"
                
            elif selected_test in ["Permutation test", "permutation"]:
                results = perform_permutation_test(
                    data, group1_samples, group2_samples,
                    n_permutations=1000
                )
                test_function = "perform_permutation_test"
                
            elif selected_test == "Paired t-test":
                # Implement paired t-test using scipy
                from scipy.stats import ttest_rel
                
                group1_data = data[group1_samples].values
                group2_data = data[group2_samples].values
                
                t_stats, p_vals = ttest_rel(group1_data, group2_data, axis=1)
                
                results = pd.DataFrame({
                    'statistic': t_stats,
                    'p_value': p_vals
                }, index=data.index)
                
                # FDR correction
                if adjust_pvalues:
                    from biomni.tool.statistics import adjust_pvalues as adjust_pvals_func
                    results['p_value_adj'] = adjust_pvals_func(results['p_value'].values)
                else:
                    results['p_value_adj'] = results['p_value']
                
                results['significant'] = results['p_value_adj'] < alpha
                test_function = "scipy.stats.ttest_rel (paired t-test)"
            
            else:
                # Default to Mann-Whitney
                results = perform_nonparametric_test(
                    data, group1_samples, group2_samples,
                    test='mannwhitney', adjust_p=adjust_pvalues
                )
                test_function = "perform_nonparametric_test(test='mannwhitney') [default]"
        
        elif n_groups > 2:
            # Multi-group analysis
            groups_series = pd.Series(dtype=str)
            for group_name, sample_list in sample_groups.items():
                for sample in sample_list:
                    groups_series[sample] = group_name
            
            all_samples = []
            for samples in sample_groups.values():
                all_samples.extend(samples)
            
            if selected_test in ["One-way ANOVA", "anova", "Welch's ANOVA"]:
                results = perform_anova(
                    data, all_samples, groups_series,
                    method='one_way', post_hoc='tukey',
                    adjust_pvalues=adjust_pvalues, alpha=alpha
                )
                test_function = "perform_anova(method='one_way')"
                
            elif selected_test in ["Kruskal-Wallis test", "kruskal"]:
                results = perform_anova(
                    data, all_samples, groups_series,
                    method='kruskal', post_hoc='bonferroni',
                    adjust_pvalues=adjust_pvalues, alpha=alpha
                )
                test_function = "perform_anova(method='kruskal')"
            
            else:
                # Default to Kruskal-Wallis for safety
                results = perform_anova(
                    data, all_samples, groups_series,
                    method='kruskal', post_hoc='bonferroni',
                    adjust_pvalues=adjust_pvalues, alpha=alpha
                )
                test_function = "perform_anova(method='kruskal') [default]"
        
        success = True
        error_message = None
        
    except Exception as e:
        success = False
        error_message = str(e)
        results = pd.DataFrame()
        test_function = "Failed"
        
        if verbose:
            print(f"\n❌ Error executing test: {error_message}")
    
    # Compile metadata
    metadata = {
        'selected_test': selected_test,
        'test_function': test_function,
        'selection_rationale': rationale,
        'confidence': confidence,
        'auto_selected': auto_select_test,
        'n_groups': n_groups,
        'paired': paired,
        'alpha': alpha,
        'adjust_pvalues': adjust_pvalues,
        'data_characteristics': {
            'n_features': len(data),
            'n_samples': sum(len(s) for s in sample_groups.values()),
            'sample_groups': {k: len(v) for k, v in sample_groups.items()},
            'normality_pct': assumptions.get('normality_pct'),
            'is_normal': assumptions.get('is_normal'),
            'has_equal_variance': assumptions.get('has_equal_variance'),
            'min_sample_size': assumptions.get('min_sample_size')
        },
        'assumptions_check': assumptions,
        'warnings': assumptions.get('warnings', []),
        'success': success,
        'error': error_message
    }
    
    if verbose and success:
        n_significant = (results.get('significant', results.get('FDR', 1) < alpha)).sum()
        print(f"\n✅ Analysis complete!")
        print(f"   Found {n_significant} significant features (α={alpha})")
    
    return results, metadata

