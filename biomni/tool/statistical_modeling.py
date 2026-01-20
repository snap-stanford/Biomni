"""
Statistical Modeling and Analysis Tools.

This module provides functions for statistical modeling, hypothesis testing,
regression analysis, time series analysis, and probability distributions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f


def linear_regression_analysis(X, y, feature_names=None, alpha=0.05):
    """
    Perform multiple linear regression analysis with statistical tests.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Independent variables (predictors)
    y : array-like, shape (n_samples,)
        Dependent variable (response)
    feature_names : list of str, optional
        Names of the features
    alpha : float
        Significance level for hypothesis tests

    Returns:
    --------
    str
        Comprehensive regression analysis report
    """
    from scipy.stats import t as t_dist
    from sklearn.linear_model import LinearRegression

    log = "# Multiple Linear Regression Analysis\n\n"

    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape
    log += "## Dataset:\n"
    log += f"- Number of observations: {n_samples}\n"
    log += f"- Number of predictors: {n_features}\n\n"

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared and Adjusted R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)

    log += "## Model Summary:\n"
    log += f"- R² = {r_squared:.4f}\n"
    log += f"- Adjusted R² = {adj_r_squared:.4f}\n"
    log += f"- Residual Standard Error (RSE) = {np.sqrt(ss_residual / (n_samples - n_features - 1)):.4f}\n\n"

    # F-statistic
    ms_model = (ss_total - ss_residual) / n_features
    ms_residual = ss_residual / (n_samples - n_features - 1)
    f_statistic = ms_model / ms_residual
    f_p_value = 1 - f.cdf(f_statistic, n_features, n_samples - n_features - 1)

    log += "## Overall Model Test:\n"
    log += f"- F-statistic = {f_statistic:.4f}\n"
    log += f"- p-value = {f_p_value:.4e}\n"
    if f_p_value < alpha:
        log += f"✓ Model is statistically significant at α={alpha}\n\n"
    else:
        log += f"✗ Model is NOT statistically significant at α={alpha}\n\n"

    # Coefficient estimates with t-tests
    log += "## Coefficient Estimates:\n\n"
    log += "| Variable | Coefficient | Std Error | t-value | p-value | Significance |\n"
    log += "|----------|-------------|-----------|---------|---------|---------------|\n"

    # Intercept
    log += f"| Intercept | {model.intercept_:.4f} | - | - | - | - |\n"

    # Calculate standard errors for coefficients
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    var_residual = ss_residual / (n_samples - n_features - 1)
    cov_matrix = var_residual * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_errors = np.sqrt(np.diag(cov_matrix))[1:]  # Exclude intercept

    for i, coef in enumerate(model.coef_):
        se = std_errors[i]
        t_value = coef / se
        p_value = 2 * (1 - t_dist.cdf(abs(t_value), n_samples - n_features - 1))
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        var_name = feature_names[i] if feature_names else f"X{i + 1}"
        log += f"| {var_name} | {coef:.4f} | {se:.4f} | {t_value:.4f} | {p_value:.4e} | {sig} |\n"

    log += "\n"
    log += "Significance codes: *** p<0.001, ** p<0.01, * p<0.05\n\n"

    # Residual diagnostics
    log += "## Residual Diagnostics:\n"
    log += f"- Mean of residuals: {np.mean(residuals):.6f} (should be ≈0)\n"
    log += f"- Std of residuals: {np.std(residuals):.4f}\n"

    # Test for normality of residuals (Shapiro-Wilk)
    if n_samples < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        log += f"- Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.4e}\n"
        if shapiro_p > 0.05:
            log += "  ✓ Residuals appear normally distributed\n"
        else:
            log += "  ⚠ Residuals may not be normally distributed\n"

    return log


def time_series_analysis(data, timestamps=None, seasonal_period=None):
    """
    Perform time series analysis including trend, seasonality, and stationarity tests.

    Parameters:
    -----------
    data : array-like
        Time series data
    timestamps : array-like, optional
        Timestamps for the data
    seasonal_period : int, optional
        Expected seasonal period (e.g., 12 for monthly data with yearly seasonality)

    Returns:
    --------
    str
        Time series analysis report
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    log = "# Time Series Analysis\n\n"

    data = np.array(data)
    n = len(data)

    log += "## Dataset:\n"
    log += f"- Number of observations: {n}\n"
    log += f"- Mean: {np.mean(data):.4f}\n"
    log += f"- Std: {np.std(data):.4f}\n"
    log += f"- Min: {np.min(data):.4f}\n"
    log += f"- Max: {np.max(data):.4f}\n\n"

    # Augmented Dickey-Fuller test for stationarity
    log += "## Stationarity Test (Augmented Dickey-Fuller):\n"
    try:
        adf_result = adfuller(data, autolag="AIC")
        log += f"- ADF Statistic: {adf_result[0]:.4f}\n"
        log += f"- p-value: {adf_result[1]:.4f}\n"
        log += "- Critical values:\n"
        for key, value in adf_result[4].items():
            log += f"  - {key}: {value:.4f}\n"

        if adf_result[1] < 0.05:
            log += "✓ Time series is likely stationary (reject unit root hypothesis)\n\n"
        else:
            log += "⚠ Time series may be non-stationary (cannot reject unit root hypothesis)\n"
            log += "  Consider differencing or detrending the data.\n\n"
    except Exception as e:
        log += f"⚠ Could not perform ADF test: {str(e)}\n\n"

    # Seasonal decomposition
    if seasonal_period and n >= 2 * seasonal_period:
        log += f"## Seasonal Decomposition (period={seasonal_period}):\n"
        try:
            # Create a pandas Series for seasonal_decompose
            if timestamps is not None:
                ts = pd.Series(data, index=pd.to_datetime(timestamps))
            else:
                ts = pd.Series(data)

            decomposition = seasonal_decompose(ts, model="additive", period=seasonal_period)

            log += f"- Trend component variance: {np.var(decomposition.trend.dropna()):.4f}\n"
            log += f"- Seasonal component variance: {np.var(decomposition.seasonal):.4f}\n"
            log += f"- Residual component variance: {np.var(decomposition.resid.dropna()):.4f}\n\n"

            # Calculate strength of trend and seasonality
            var_resid = np.var(decomposition.resid.dropna())
            var_detrend = np.var((ts - decomposition.trend).dropna())
            var_deseason = np.var((ts - decomposition.seasonal).dropna())

            strength_trend = max(0, 1 - var_resid / var_detrend)
            strength_seasonal = max(0, 1 - var_resid / var_deseason)

            log += f"- Strength of trend: {strength_trend:.4f}\n"
            log += f"- Strength of seasonality: {strength_seasonal:.4f}\n\n"

        except Exception as e:
            log += f"⚠ Could not perform seasonal decomposition: {str(e)}\n\n"

    # Autocorrelation analysis
    log += "## Autocorrelation Analysis:\n"
    max_lag = min(20, n // 2)
    autocorr = [np.corrcoef(data[: -i or None], data[i:])[0, 1] for i in range(1, max_lag + 1)]

    log += "First 10 lags:\n"
    for i, ac in enumerate(autocorr[:10], 1):
        log += f"- Lag {i}: {ac:.4f}\n"

    log += "\n"

    return log


def hypothesis_test_two_samples(sample1, sample2, test_type="t-test", alpha=0.05, alternative="two-sided"):
    """
    Perform hypothesis tests comparing two samples.

    Parameters:
    -----------
    sample1, sample2 : array-like
        Two samples to compare
    test_type : str
        Type of test: 't-test', 'mann-whitney', 'ks-test'
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', 'greater'

    Returns:
    --------
    str
        Hypothesis test results
    """
    log = f"# Two-Sample Hypothesis Test: {test_type}\n\n"

    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    log += "## Sample Statistics:\n"
    log += "### Sample 1:\n"
    log += f"- n = {len(sample1)}\n"
    log += f"- Mean = {np.mean(sample1):.4f}\n"
    log += f"- Std = {np.std(sample1, ddof=1):.4f}\n"
    log += f"- Median = {np.median(sample1):.4f}\n\n"

    log += "### Sample 2:\n"
    log += f"- n = {len(sample2)}\n"
    log += f"- Mean = {np.mean(sample2):.4f}\n"
    log += f"- Std = {np.std(sample2, ddof=1):.4f}\n"
    log += f"- Median = {np.median(sample2):.4f}\n\n"

    log += "## Hypothesis Test:\n"
    log += f"- Test type: {test_type}\n"
    log += f"- Significance level: α = {alpha}\n"
    log += f"- Alternative hypothesis: {alternative}\n\n"

    if test_type == "t-test":
        # Independent two-sample t-test
        statistic, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
        log += "### Results:\n"
        log += f"- t-statistic = {statistic:.4f}\n"
        log += f"- p-value = {p_value:.4e}\n"
        log += f"- Mean difference = {np.mean(sample1) - np.mean(sample2):.4f}\n\n"

    elif test_type == "mann-whitney":
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
        log += "### Results:\n"
        log += f"- U-statistic = {statistic:.4f}\n"
        log += f"- p-value = {p_value:.4e}\n\n"

    elif test_type == "ks-test":
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(sample1, sample2, alternative=alternative)
        log += "### Results:\n"
        log += f"- KS-statistic = {statistic:.4f}\n"
        log += f"- p-value = {p_value:.4e}\n\n"

    else:
        return f"Error: Unknown test type '{test_type}'"

    # Interpretation
    log += "### Interpretation:\n"
    if p_value < alpha:
        log += f"✓ Reject the null hypothesis (p < {alpha})\n"
        log += "  There is significant evidence of a difference between the two samples.\n"
    else:
        log += f"✗ Fail to reject the null hypothesis (p ≥ {alpha})\n"
        log += "  There is insufficient evidence of a difference between the two samples.\n"

    return log


def fit_distribution(data, distribution="norm"):
    """
    Fit a probability distribution to data and perform goodness-of-fit tests.

    Parameters:
    -----------
    data : array-like
        Data to fit
    distribution : str
        Distribution name: 'norm', 'lognorm', 'exponential', 'gamma', 'beta', etc.

    Returns:
    --------
    str
        Distribution fitting results and goodness-of-fit statistics
    """
    log = f"# Distribution Fitting: {distribution}\n\n"

    data = np.array(data)

    log += "## Data Summary:\n"
    log += f"- n = {len(data)}\n"
    log += f"- Mean = {np.mean(data):.4f}\n"
    log += f"- Std = {np.std(data):.4f}\n"
    log += f"- Min = {np.min(data):.4f}\n"
    log += f"- Max = {np.max(data):.4f}\n\n"

    # Get the distribution from scipy.stats
    try:
        dist = getattr(stats, distribution)
    except AttributeError:
        return f"Error: Unknown distribution '{distribution}'"

    # Fit the distribution
    log += f"## Fitting {distribution} distribution:\n"
    params = dist.fit(data)

    log += f"- Fitted parameters: {params}\n\n"

    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))

    log += "## Goodness-of-Fit Test (Kolmogorov-Smirnov):\n"
    log += f"- KS statistic = {ks_statistic:.4f}\n"
    log += f"- p-value = {ks_p_value:.4e}\n\n"

    if ks_p_value > 0.05:
        log += f"✓ The {distribution} distribution fits the data reasonably well (p > 0.05)\n"
    else:
        log += f"✗ The {distribution} distribution may not fit the data well (p ≤ 0.05)\n"

    # Calculate percentiles
    log += "\n## Fitted Distribution Percentiles:\n"
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        theoretical = dist.ppf(p / 100, *params)
        empirical = np.percentile(data, p)
        log += f"- {p}th: Theoretical={theoretical:.4f}, Empirical={empirical:.4f}\n"

    return log


def anova_one_way(*groups, alpha=0.05):
    """
    Perform one-way ANOVA to test if means of multiple groups are equal.

    Parameters:
    -----------
    *groups : array-like
        Multiple groups to compare
    alpha : float
        Significance level

    Returns:
    --------
    str
        ANOVA results and post-hoc analysis
    """
    log = "# One-Way ANOVA\n\n"

    log += "## Setup:\n"
    log += f"- Number of groups: {len(groups)}\n"
    log += f"- Group sizes: {[len(g) for g in groups]}\n"
    log += f"- Significance level: α = {alpha}\n\n"

    # Perform ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)

    log += "## ANOVA Results:\n"
    log += f"- F-statistic = {f_statistic:.4f}\n"
    log += f"- p-value = {p_value:.4e}\n\n"

    if p_value < alpha:
        log += f"✓ Reject the null hypothesis (p < {alpha})\n"
        log += "  At least one group mean is significantly different.\n\n"
    else:
        log += f"✗ Fail to reject the null hypothesis (p ≥ {alpha})\n"
        log += "  No significant difference between group means detected.\n\n"

    # Group statistics
    log += "## Group Statistics:\n"
    for i, group in enumerate(groups, 1):
        log += f"### Group {i}:\n"
        log += f"- n = {len(group)}\n"
        log += f"- Mean = {np.mean(group):.4f}\n"
        log += f"- Std = {np.std(group, ddof=1):.4f}\n\n"

    return log


def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=10000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for a statistic.

    Parameters:
    -----------
    data : array-like
        Original data
    statistic_func : callable
        Function to compute the statistic (e.g., np.mean, np.median)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)

    Returns:
    --------
    str
        Bootstrap confidence interval results
    """
    log = "# Bootstrap Confidence Interval\n\n"

    data = np.array(data)
    n = len(data)

    log += "## Setup:\n"
    log += f"- Sample size: {n}\n"
    log += f"- Number of bootstrap samples: {n_bootstrap}\n"
    log += f"- Confidence level: {confidence_level * 100}%\n\n"

    # Compute observed statistic
    observed_stat = statistic_func(data)
    log += f"## Observed statistic: {observed_stat:.4f}\n\n"

    # Bootstrap resampling
    bootstrap_statistics = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics.append(statistic_func(resample))

    bootstrap_statistics = np.array(bootstrap_statistics)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
    ci_upper = np.percentile(bootstrap_statistics, upper_percentile)

    log += "## Bootstrap Results:\n"
    log += f"- Mean of bootstrap statistics: {np.mean(bootstrap_statistics):.4f}\n"
    log += f"- Std of bootstrap statistics: {np.std(bootstrap_statistics):.4f}\n\n"
    log += f"## {confidence_level * 100}% Confidence Interval:\n"
    log += f"- Lower bound: {ci_lower:.4f}\n"
    log += f"- Upper bound: {ci_upper:.4f}\n"
    log += f"- Interval width: {ci_upper - ci_lower:.4f}\n"

    return log
