"""Result aggregation helpers for comparisons and ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.metrics import aggregate_curves_with_ci, pairwise_method_tests


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a summary JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_scalar(values: Iterable[float]) -> dict[str, float]:
    """Compute basic summary statistics for a metric across seeds."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "min": 0.0, "max": 0.0}
    std = array.std(ddof=1) if array.size > 1 else 0.0
    sem = std / np.sqrt(max(array.size, 1))
    return {
        "mean": float(array.mean()),
        "std": float(std),
        "sem": float(sem),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_method_summaries(
    summaries_by_method: dict[str, list[dict[str, Any]]],
    metrics: Iterable[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate scalar summary metrics across methods."""
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for method, summaries in summaries_by_method.items():
        aggregated[method] = {}
        for metric in metrics:
            values = [float(summary.get(metric, 0.0)) for summary in summaries]
            aggregated[method][metric] = summarize_scalar(values)
    return aggregated


def aggregate_method_curves(
    summaries_by_method: dict[str, list[dict[str, Any]]],
    *,
    curve_key: str,
    budget: int,
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per-seed query curves for plotting."""
    aggregated = {}
    for method, summaries in summaries_by_method.items():
        curves = [summary.get("curve", {}) for summary in summaries]
        stats = aggregate_curves_with_ci(curves, budget=budget, key=curve_key)
        aggregated[method] = {
            "x": stats["x"].tolist(),
            "mean": stats["mean"].tolist(),
            "lower": stats["lower"].tolist(),
            "upper": stats["upper"].tolist(),
        }
    return aggregated


def pairwise_metric_tests(
    summaries_by_method: dict[str, list[dict[str, Any]]],
    *,
    metric: str,
    reference_method: str,
) -> dict[str, dict[str, float]]:
    """Run paired tests for a scalar metric across methods."""
    final_scores = {
        method: [float(summary.get(metric, 0.0)) for summary in summaries]
        for method, summaries in summaries_by_method.items()
    }
    return pairwise_method_tests(final_scores, reference_method=reference_method)
