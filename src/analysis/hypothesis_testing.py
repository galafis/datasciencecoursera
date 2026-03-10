"""
Hypothesis Testing Module

Provides implementations of common statistical tests including
t-tests, chi-square tests, and ANOVA.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def independent_ttest(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform an independent two-sample t-test.

    Args:
        group1: 1D numpy array for group 1.
        group2: 1D numpy array for group 2.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with t_statistic, p_value, degrees_of_freedom,
        reject_null, mean_diff, and confidence_interval.
    """
    if group1.size < 2 or group2.size < 2:
        raise ValueError("Each group must have at least 2 observations.")

    t_stat, p_value = stats.ttest_ind(group1, group2)
    df = group1.size + group2.size - 2
    mean_diff = float(np.mean(group1) - np.mean(group2))

    se = np.sqrt(np.var(group1, ddof=1) / group1.size + np.var(group2, ddof=1) / group2.size)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(df),
        "reject_null": bool(p_value < alpha),
        "mean_diff": mean_diff,
        "confidence_interval": ci,
        "alpha": alpha,
    }


def paired_ttest(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform a paired t-test.

    Args:
        before: 1D array of measurements before treatment.
        after: 1D array of measurements after treatment.
        alpha: Significance level.

    Returns:
        Dictionary with t_statistic, p_value, reject_null, mean_diff.
    """
    if before.size != after.size:
        raise ValueError("Arrays must have the same length for paired t-test.")
    if before.size < 2:
        raise ValueError("Arrays must have at least 2 observations.")

    t_stat, p_value = stats.ttest_rel(before, after)
    diff = before - after
    mean_diff = float(np.mean(diff))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "mean_diff": mean_diff,
        "alpha": alpha,
    }


def one_sample_ttest(data: np.ndarray, population_mean: float, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform a one-sample t-test against a known population mean.

    Args:
        data: 1D array of sample observations.
        population_mean: The hypothesized population mean.
        alpha: Significance level.

    Returns:
        Dictionary with t_statistic, p_value, reject_null, sample_mean.
    """
    if data.size < 2:
        raise ValueError("Data must have at least 2 observations.")

    t_stat, p_value = stats.ttest_1samp(data, population_mean)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "sample_mean": float(np.mean(data)),
        "population_mean": population_mean,
        "alpha": alpha,
    }


def chi_square_test(observed: np.ndarray, expected: np.ndarray = None, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform a chi-square goodness-of-fit test.

    Args:
        observed: 1D array of observed frequencies.
        expected: 1D array of expected frequencies (uniform if None).
        alpha: Significance level.

    Returns:
        Dictionary with chi2_statistic, p_value, degrees_of_freedom, reject_null.
    """
    observed = np.asarray(observed, dtype=float)
    if expected is None:
        expected = np.full_like(observed, np.mean(observed))
    else:
        expected = np.asarray(expected, dtype=float)

    if observed.shape != expected.shape:
        raise ValueError("Observed and expected arrays must have the same shape.")

    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    df = len(observed) - 1

    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": df,
        "reject_null": bool(p_value < alpha),
        "alpha": alpha,
    }


def chi_square_independence(contingency_table: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform a chi-square test of independence on a contingency table.

    Args:
        contingency_table: 2D array representing the contingency table.
        alpha: Significance level.

    Returns:
        Dictionary with chi2_statistic, p_value, degrees_of_freedom,
        reject_null, and expected_frequencies.
    """
    contingency_table = np.asarray(contingency_table, dtype=float)
    if contingency_table.ndim != 2:
        raise ValueError("Contingency table must be a 2D array.")

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "reject_null": bool(p_value < alpha),
        "expected_frequencies": expected,
        "alpha": alpha,
    }


def one_way_anova(*groups: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """
    Perform a one-way ANOVA test.

    Args:
        *groups: Two or more 1D arrays, one per group.
        alpha: Significance level.

    Returns:
        Dictionary with f_statistic, p_value, reject_null,
        group_means, and overall_mean.
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups.")

    for i, g in enumerate(groups):
        if g.size < 2:
            raise ValueError(f"Group {i} must have at least 2 observations.")

    f_stat, p_value = stats.f_oneway(*groups)
    group_means = [float(np.mean(g)) for g in groups]
    all_data = np.concatenate(groups)
    overall_mean = float(np.mean(all_data))

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "group_means": group_means,
        "overall_mean": overall_mean,
        "n_groups": len(groups),
        "alpha": alpha,
    }
