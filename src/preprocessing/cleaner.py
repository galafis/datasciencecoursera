"""
Data Cleaning and Preprocessing Module

Provides utilities for handling missing values, detecting and treating outliers,
normalization, standardization, and encoding.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def handle_missing_values(data: np.ndarray, strategy: str = "mean",
                          fill_value: Optional[float] = None) -> np.ndarray:
    """
    Handle missing values (NaN) in a numerical array.

    Args:
        data: 1D or 2D numpy array that may contain NaN values.
        strategy: Strategy for imputation - 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: Value to use when strategy is 'constant'.

    Returns:
        Array with missing values handled.
    """
    data = np.asarray(data, dtype=float).copy()

    if strategy == "drop":
        if data.ndim == 1:
            return data[~np.isnan(data)]
        else:
            mask = ~np.any(np.isnan(data), axis=1)
            return data[mask]

    if data.ndim == 1:
        mask = np.isnan(data)
        if not np.any(mask):
            return data
        if strategy == "mean":
            data[mask] = np.nanmean(data)
        elif strategy == "median":
            data[mask] = np.nanmedian(data)
        elif strategy == "mode":
            valid = data[~mask]
            if len(valid) > 0:
                values, counts = np.unique(valid, return_counts=True)
                data[mask] = values[np.argmax(counts)]
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be provided for 'constant' strategy.")
            data[mask] = fill_value
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    else:
        for col in range(data.shape[1]):
            col_data = data[:, col]
            mask = np.isnan(col_data)
            if not np.any(mask):
                continue
            if strategy == "mean":
                col_data[mask] = np.nanmean(col_data)
            elif strategy == "median":
                col_data[mask] = np.nanmedian(col_data)
            elif strategy == "mode":
                valid = col_data[~mask]
                if len(valid) > 0:
                    values, counts = np.unique(valid, return_counts=True)
                    col_data[mask] = values[np.argmax(counts)]
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("fill_value must be provided for 'constant' strategy.")
                col_data[mask] = fill_value
            data[:, col] = col_data

    return data


def remove_outliers(data: np.ndarray, method: str = "iqr", factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers from a 1D array.

    Args:
        data: 1D numpy array.
        method: Detection method - 'iqr' or 'zscore'.
        factor: IQR multiplier (default 1.5) or z-score threshold (default 3.0 when method='zscore').

    Returns:
        Array with outliers removed.
    """
    data = np.asarray(data, dtype=float)

    if method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (data >= lower) & (data <= upper)
    elif method == "zscore":
        threshold = factor if factor != 1.5 else 3.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return data
        z_scores = np.abs((data - mean) / std)
        mask = z_scores <= threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    return data[mask]


def normalize(data: np.ndarray, method: str = "minmax",
              feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
    """
    Normalize data using min-max scaling or other methods.

    Args:
        data: 1D or 2D numpy array.
        method: Normalization method - 'minmax', 'standard', 'robust', 'l2'.
        feature_range: Target range for min-max scaling (default (0, 1)).

    Returns:
        Tuple of (normalized_data, params) where params can be used for inverse transform.
    """
    data = np.asarray(data, dtype=float)
    params = {"method": method}

    if method == "minmax":
        if data.ndim == 1:
            dmin, dmax = data.min(), data.max()
            if dmax - dmin == 0:
                return np.zeros_like(data), {"method": method, "min": dmin, "max": dmax, "range": feature_range}
            scaled = (data - dmin) / (dmax - dmin)
            low, high = feature_range
            result = scaled * (high - low) + low
            params.update({"min": dmin, "max": dmax, "range": feature_range})
        else:
            dmin = data.min(axis=0)
            dmax = data.max(axis=0)
            drange = dmax - dmin
            drange[drange == 0] = 1
            scaled = (data - dmin) / drange
            low, high = feature_range
            result = scaled * (high - low) + low
            params.update({"min": dmin.tolist(), "max": dmax.tolist(), "range": feature_range})

    elif method == "standard":
        if data.ndim == 1:
            mean, std = data.mean(), data.std(ddof=1)
            if std == 0:
                std = 1.0
            result = (data - mean) / std
            params.update({"mean": mean, "std": std})
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            result = (data - mean) / std
            params.update({"mean": mean.tolist(), "std": std.tolist()})

    elif method == "robust":
        if data.ndim == 1:
            median = np.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                iqr = 1.0
            result = (data - median) / iqr
            params.update({"median": median, "iqr": iqr})
        else:
            median = np.median(data, axis=0)
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1.0
            result = (data - median) / iqr
            params.update({"median": median.tolist(), "iqr": iqr.tolist()})

    elif method == "l2":
        if data.ndim == 1:
            norm = np.linalg.norm(data)
            if norm == 0:
                norm = 1.0
            result = data / norm
            params.update({"norm": norm})
        else:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = data / norms
            params.update({"norms": norms.flatten().tolist()})
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return result, params


def detect_missing_summary(data: np.ndarray, column_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Generate a summary of missing values per column.

    Args:
        data: 2D numpy array.
        column_names: Optional list of column names.

    Returns:
        Dictionary mapping column names to missing value info.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    n_rows, n_cols = data.shape
    if column_names is None:
        column_names = [f"col_{i}" for i in range(n_cols)]

    summary = {}
    for i, name in enumerate(column_names):
        n_missing = int(np.sum(np.isnan(data[:, i])))
        summary[name] = {
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n_rows, 2) if n_rows > 0 else 0.0,
            "n_present": n_rows - n_missing,
        }
    return summary


def clip_values(data: np.ndarray, lower: Optional[float] = None,
                upper: Optional[float] = None) -> np.ndarray:
    """
    Clip values to a specified range.

    Args:
        data: Numpy array.
        lower: Minimum allowed value (None for no lower bound).
        upper: Maximum allowed value (None for no upper bound).

    Returns:
        Clipped array.
    """
    return np.clip(data, lower, upper)
