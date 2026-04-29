"""Generate report-ready figures from comparison outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LATEST_RUN = REPO_ROOT / "outputs" / "comparisons" / "2026-03-29_21-12-32"
BASELINE_REFERENCE_RUN = REPO_ROOT / "outputs" / "comparisons" / "2026-03-29_14-51-09"
OUTPUT_DIR = REPO_ROOT / "reports" / "visualizations" / "assets"

METHODS = ["active", "gflownet", "hybrid"]
LABELS = {
    "active": "Actif",
    "gflownet": "GFlowNet",
    "hybrid": "Hybride",
    "baseline": "Supervise (ref.)",
}
COLORS = {
    "active": "#1f77b4",
    "gflownet": "#d62728",
    "hybrid": "#2ca02c",
    "baseline": "#7f7f7f",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_quality_overview(latest_metrics: dict) -> Path:
    metrics = [
        ("best_score", "Best score", 1.0),
        ("top10_score", "Top-10 moyen", 1.0),
        ("mean_score", "Score moyen", 1.0),
        ("valid_word_ratio", "Mots valides (%)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    axes = axes.flatten()
    x = np.arange(len(METHODS))
    labels = [LABELS[m] for m in METHODS]

    for ax, (metric, title, scale) in zip(axes, metrics):
        values = [latest_metrics[m][metric]["mean"] * scale for m in METHODS]
        errs = [latest_metrics[m][metric]["sem"] * scale for m in METHODS]
        colors = [COLORS[m] for m in METHODS]
        ax.bar(x, values, yerr=errs, color=colors, alpha=0.88, capsize=4)
        ax.set_xticks(x, labels, rotation=12)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("Comparaison des performances finales (5 graines)", fontsize=13)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "quality_overview.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cost_tradeoffs(latest_metrics: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    x = np.arange(len(METHODS))
    labels = [LABELS[m] for m in METHODS]

    q90_vals = [latest_metrics[m]["queries_to_90pct_best"]["mean"] for m in METHODS]
    q90_errs = [latest_metrics[m]["queries_to_90pct_best"]["sem"] for m in METHODS]
    axes[0].bar(x, q90_vals, yerr=q90_errs, color=[COLORS[m] for m in METHODS], capsize=4)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_title("Q90\n(requetes oracle)")
    axes[0].grid(axis="y", alpha=0.25)

    runtime_vals = [latest_metrics[m]["runtime_seconds"]["mean"] for m in METHODS]
    runtime_errs = [latest_metrics[m]["runtime_seconds"]["sem"] for m in METHODS]
    axes[1].bar(x, runtime_vals, yerr=runtime_errs, color=[COLORS[m] for m in METHODS], capsize=4)
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_title("Temps mur (s)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25)

    cheap_vals = [latest_metrics[m]["cheap_model_queries"]["mean"] for m in METHODS]
    cheap_plot = [v + 1.0 for v in cheap_vals]
    axes[2].bar(x, cheap_plot, color=[COLORS[m] for m in METHODS], alpha=0.88)
    axes[2].set_xticks(x, labels, rotation=12)
    axes[2].set_title("Requetes bon marche")
    axes[2].set_yscale("log")
    axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_axisbelow(True)

    fig.suptitle("Couts de recherche sous le meme budget reel", fontsize=13)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "cost_tradeoffs.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_curve(ax, curves: dict, title: str) -> None:
    for method in METHODS:
        curve = curves[method]
        x = np.asarray(curve["x"], dtype=float)
        mean = np.asarray(curve["mean"], dtype=float)
        lower = np.asarray(curve["lower"], dtype=float)
        upper = np.asarray(curve["upper"], dtype=float)
        ax.plot(x, mean, label=LABELS[method], color=COLORS[method], linewidth=2)
        ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Requetes oracle")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def plot_query_dynamics(best_curve: dict, top10_curve: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    _plot_curve(axes[0], best_curve, "Meilleur score cumulatif")
    axes[0].set_ylabel("Best score")
    _plot_curve(axes[1], top10_curve, "Top-10 cumulatif")
    axes[1].set_ylabel("Top-10 moyen")
    axes[1].legend(loc="lower right")
    fig.suptitle("Dynamique d'apprentissage (5 graines)", fontsize=13)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "query_dynamics.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summary_csv(latest_metrics: dict, baseline_reference: dict) -> Path:
    output_path = OUTPUT_DIR / "report_metrics_summary.csv"
    rows = []
    for method in METHODS:
        rows.append(
            {
                "method": LABELS[method],
                "source": "latest_5seed_run_2026-03-29_21-12-32",
                "best_score": latest_metrics[method]["best_score"]["mean"],
                "top10_score": latest_metrics[method]["top10_score"]["mean"],
                "mean_score": latest_metrics[method]["mean_score"]["mean"],
                "valid_word_ratio": latest_metrics[method]["valid_word_ratio"]["mean"],
                "topk_edit_distance": latest_metrics[method]["topk_edit_distance"]["mean"],
                "q90": latest_metrics[method]["queries_to_90pct_best"]["mean"],
                "runtime_seconds": latest_metrics[method]["runtime_seconds"]["mean"],
                "cheap_model_queries": latest_metrics[method]["cheap_model_queries"]["mean"],
                "note": "",
            }
        )

    rows.append(
        {
            "method": LABELS["baseline"],
            "source": "reference_3seed_run_2026-03-29_14-51-09",
            "best_score": baseline_reference["best_score"]["mean"],
            "top10_score": baseline_reference["top10_score"]["mean"],
            "mean_score": baseline_reference["mean_score"]["mean"],
            "valid_word_ratio": baseline_reference["valid_word_ratio"]["mean"],
            "topk_edit_distance": baseline_reference["topk_edit_distance"]["mean"],
            "q90": baseline_reference["queries_to_90pct_best"]["mean"],
            "runtime_seconds": baseline_reference["runtime_seconds"]["mean"],
            "cheap_model_queries": baseline_reference["cheap_model_queries"]["mean"],
            "note": "Reference only; not rerun in latest 5-seed tuned sweep.",
        }
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_metrics = load_json(LATEST_RUN / "aggregate_metrics.json")
    best_curve = load_json(LATEST_RUN / "curve_best.json")
    top10_curve = load_json(LATEST_RUN / "curve_top10.json")
    baseline_reference = load_json(BASELINE_REFERENCE_RUN / "aggregate_metrics.json")["baseline"]

    plot_quality_overview(latest_metrics)
    plot_cost_tradeoffs(latest_metrics)
    plot_query_dynamics(best_curve, top10_curve)
    write_summary_csv(latest_metrics, baseline_reference)


if __name__ == "__main__":
    main()
