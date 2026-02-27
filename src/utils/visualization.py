"""Plot helpers for experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_curves(
    curves: Mapping[str, Sequence[float]],
    x_values: Sequence[float],
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    """Plot multiple metric curves on the same axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, values in curves.items():
        plt.plot(x_values, values, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_from_frame(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    output_path: Path,
    title: str,
) -> Path:
    """Plot grouped lines from a tidy DataFrame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for group, group_df in frame.groupby(group_col):
        sorted_df = group_df.sort_values(x_col)
        plt.plot(sorted_df[x_col], sorted_df[y_col], label=str(group))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_metric_curves_with_ci(
    curve_stats: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    alpha_fill: float = 0.2,
) -> Path:
    """Plot mean curve with confidence intervals for each method."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for method, stats in curve_stats.items():
        x = stats["x"]
        mean = stats["mean"]
        lower = stats["lower"]
        upper = stats["upper"]
        plt.plot(x, mean, label=str(method))
        plt.fill_between(x, lower, upper, alpha=alpha_fill)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_heatmap_from_pivot(
    pivot: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str = "viridis",
) -> Path:
    """Plot heatmap from a pivot table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", cmap=cmap)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.colorbar(label="Metric")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
