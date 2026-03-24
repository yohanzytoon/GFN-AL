"""Plotting helpers for experiment comparisons and ablations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_query_curves(
    curves_by_method: dict[str, dict[str, list[float]]],
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    """Plot aggregated query curves with confidence bands."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for method, curve in curves_by_method.items():
        x = curve["x"]
        mean = curve["mean"]
        lower = curve["lower"]
        upper = curve["upper"]
        plt.plot(x, mean, label=method)
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.xlabel("Oracle queries")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
