"""
Exploratory Data Analysis (EDA) Module

Provides functions for descriptive statistics, distribution analysis,
and correlation computation on numerical datasets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics for a 1D numerical array.

    Args:
        data: 1D numpy array of numerical values.

    Returns:
        Dictionary with mean, median, std, min, max, q1, q3, iqr, skewness, kurtosis.
    """
    if data.size == 0:
        raise ValueError("Input array must not be empty.")

    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0

    n = data.size
    if n > 2 and std > 0:
        skewness = float((n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3))
    else:
        skewness = 0.0

    if n > 3 and std > 0:
        kurt = float(
            (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)))
            * np.sum(((data - mean) / std) ** 4)
            - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        )
    else:
        kurt = 0.0

    return {
        "mean": mean,
        "median": float(np.median(data)),
        "std": std,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "skewness": skewness,
        "kurtosis": kurt,
        "count": int(n),
    }


def column_stats(data: np.ndarray, column_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute descriptive statistics for each column of a 2D array.

    Args:
        data: 2D numpy array (rows = observations, columns = features).
        column_names: Optional list of column names.

    Returns:
        Dictionary mapping column name to its descriptive statistics.
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    n_cols = data.shape[1]
    if column_names is None:
        column_names = [f"col_{i}" for i in range(n_cols)]

    if len(column_names) != n_cols:
        raise ValueError("Number of column names must match number of columns.")

    return {name: descriptive_stats(data[:, i]) for i, name in enumerate(column_names)}


def frequency_distribution(data: np.ndarray, bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute frequency distribution (histogram) for a 1D array.

    Args:
        data: 1D numpy array.
        bins: Number of bins.

    Returns:
        Dictionary with 'counts', 'bin_edges', and 'bin_centers'.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return {
        "counts": counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
    }


def correlation_matrix(data: np.ndarray, column_names: Optional[List[str]] = None) -> Dict[str, object]:
    """
    Compute the Pearson correlation matrix for a 2D array.

    Args:
        data: 2D numpy array (rows = observations, columns = features).
        column_names: Optional list of column names.

    Returns:
        Dictionary with 'matrix' (2D array) and 'columns' (list of names).
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    n_cols = data.shape[1]
    if column_names is None:
        column_names = [f"col_{i}" for i in range(n_cols)]

    corr = np.corrcoef(data, rowvar=False)
    return {
        "matrix": corr,
        "columns": column_names,
    }


def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using the IQR method.

    Args:
        data: 1D numpy array.
        factor: Multiplier for IQR to define bounds (default 1.5).

    Returns:
        Tuple of (outlier_mask, outlier_values).
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (data < lower) | (data > upper)
    return mask, data[mask]


def value_counts(data: np.ndarray) -> Dict[str, int]:
    """
    Count occurrences of each unique value.

    Args:
        data: 1D numpy array.

    Returns:
        Dictionary mapping unique values (as strings) to counts.
    """
    unique, counts = np.unique(data, return_counts=True)
    return {str(v): int(c) for v, c in zip(unique, counts)}
