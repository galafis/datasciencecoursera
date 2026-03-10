"""
Visualization Module

Matplotlib wrapper functions for common data science plots.
Provides a consistent API for creating publication-quality visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def create_figure(figsize: Tuple[int, int] = (10, 6),
                  title: str = "", xlabel: str = "", ylabel: str = "") -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a new figure with standard formatting.

    Args:
        figsize: Figure size as (width, height).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_histogram(data: np.ndarray, bins: int = 30, title: str = "Histogram",
                   xlabel: str = "Value", ylabel: str = "Frequency",
                   color: str = "#3498db", save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a histogram plot.

    Args:
        data: 1D array of values.
        bins: Number of bins.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Bar color.
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = create_figure(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean: {np.mean(data):.2f}")
    ax.axvline(np.median(data), color="green", linestyle="--", label=f"Median: {np.median(data):.2f}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str = "Scatter Plot",
                 xlabel: str = "X", ylabel: str = "Y",
                 color: str = "#2ecc71", add_trendline: bool = False,
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a scatter plot with optional trendline.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Point color.
        add_trendline: Whether to add a linear trendline.
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = create_figure(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.scatter(x, y, color=color, alpha=0.6, edgecolors="white", s=50)

    if add_trendline and len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(corr_matrix: np.ndarray, labels: Optional[List[str]] = None,
                             title: str = "Correlation Matrix",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a correlation heatmap.

    Args:
        corr_matrix: 2D correlation matrix.
        labels: Column/row labels.
        title: Plot title.
        save_path: Optional file path to save.

    Returns:
        Matplotlib figure.
    """
    n = corr_matrix.shape[0]
    if labels is None:
        labels = [f"Var {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_boxplot(data_groups: List[np.ndarray], labels: Optional[List[str]] = None,
                 title: str = "Box Plot", ylabel: str = "Value",
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a box plot for multiple groups.

    Args:
        data_groups: List of 1D arrays, one per group.
        labels: Group labels.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional file path to save.

    Returns:
        Matplotlib figure.
    """
    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(data_groups))]

    fig, ax = create_figure(title=title, ylabel=ylabel)
    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True)

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]
    for patch, color in zip(bp["boxes"], colors * 10):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bar(categories: List[str], values: np.ndarray, title: str = "Bar Chart",
             xlabel: str = "Category", ylabel: str = "Value",
             color: str = "#9b59b6", horizontal: bool = False,
             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart.

    Args:
        categories: Category labels.
        values: Corresponding values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Bar color.
        horizontal: Whether to create a horizontal bar chart.
        save_path: Optional file path to save.

    Returns:
        Matplotlib figure.
    """
    fig, ax = create_figure(title=title, xlabel=xlabel, ylabel=ylabel)

    if horizontal:
        ax.barh(categories, values, color=color, alpha=0.8, edgecolor="white")
    else:
        ax.bar(categories, values, color=color, alpha=0.8, edgecolor="white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_line(x: np.ndarray, y_series: Dict[str, np.ndarray],
              title: str = "Line Chart", xlabel: str = "X", ylabel: str = "Y",
              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a line chart with multiple series.

    Args:
        x: X-axis values.
        y_series: Dictionary mapping series names to Y values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional file path to save.

    Returns:
        Matplotlib figure.
    """
    fig, ax = create_figure(title=title, xlabel=xlabel, ylabel=ylabel)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (name, y) in enumerate(y_series.items()):
        ax.plot(x, y, label=name, color=colors[idx % len(colors)], linewidth=2, marker="o", markersize=3)

    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
