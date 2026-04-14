#!/usr/bin/env python3
"""Generate a Markdown report from comparison and ablation outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


HIGHER_IS_BETTER = {
    "best_score": True,
    "top10_score": True,
    "mean_score": True,
    "valid_word_ratio": True,
    "mode_coverage": True,
    "topk_edit_distance": False,
    "queries_to_90pct_best": False,
}


DISPLAY_NAMES = {
    "best_score": "Best Score",
    "top10_score": "Top-10 Score",
    "mean_score": "Mean Score",
    "valid_word_ratio": "Valid Word Ratio",
    "mode_coverage": "Mode Coverage",
    "topk_edit_distance": "Top-k Edit Distance",
    "queries_to_90pct_best": "Queries To 90% Best",
}


@dataclass
class RunSelection:
    comparisons_dir: Path | None
    ablations_dir: Path | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_subdir(root: Path) -> Path | None:
    if not root.exists():
        return None
    subdirs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    return subdirs[0] if subdirs else None


def _discover_runs(comparisons_dir: str | None, ablations_dir: str | None) -> RunSelection:
    comparisons = Path(comparisons_dir).resolve() if comparisons_dir else _latest_subdir(OUTPUTS_ROOT / "comparisons")
    ablations = Path(ablations_dir).resolve() if ablations_dir else _latest_subdir(OUTPUTS_ROOT / "ablations")
    return RunSelection(comparisons_dir=comparisons, ablations_dir=ablations)


def _fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _metric_table(comparison_metrics: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in comparison_metrics.items():
        row = {
            "method": method,
            "best_score": float(metrics["best_score"]["mean"]),
            "best_score_sem": float(metrics["best_score"]["sem"]),
            "top10_score": float(metrics["top10_score"]["mean"]),
            "top10_score_sem": float(metrics["top10_score"]["sem"]),
            "mean_score": float(metrics["mean_score"]["mean"]),
            "valid_word_ratio": float(metrics["valid_word_ratio"]["mean"]),
            "queries_to_90pct_best": float(metrics["queries_to_90pct_best"]["mean"]),
            "queries_to_90pct_best_sem": float(metrics["queries_to_90pct_best"]["sem"]),
            "real_oracle_queries": float(
                metrics.get("real_oracle_queries", metrics.get("oracle_queries", {"mean": 0.0}))["mean"]
            ),
            "fake_oracle_queries": float(metrics.get("fake_oracle_queries", {"mean": 0.0})["mean"]),
            "cheap_model_queries": float(metrics.get("cheap_model_queries", {"mean": 0.0})["mean"]),
            "runtime_seconds": float(metrics.get("runtime_seconds", {"mean": 0.0})["mean"]),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["best_score"], reverse=True)
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _winner(metrics: dict[str, dict[str, dict[str, float]]], metric: str) -> tuple[str, float]:
    items = [
        (method, float(values[metric]["mean"]))
        for method, values in metrics.items()
        if metric in values
    ]
    reverse = HIGHER_IS_BETTER.get(metric, True)
    items.sort(key=lambda item: item[1], reverse=reverse)
    return items[0]


def _interpret_comparison(metrics: dict[str, dict[str, dict[str, float]]], pairwise_tests: dict[str, Any]) -> list[str]:
    bullets: list[str] = []

    best_method, best_value = _winner(metrics, "best_score")
    top10_method, top10_value = _winner(metrics, "top10_score")
    q90_method, q90_value = _winner(metrics, "queries_to_90pct_best")

    bullets.append(
        f"`{best_method}` has the highest mean Best Score ({_fmt_float(best_value)}), which is the strongest single-number indicator of final search quality in this report."
    )
    bullets.append(
        f"`{top10_method}` has the highest mean Top-10 Score ({_fmt_float(top10_value)}), which suggests it concentrates more good candidates near the top of its ranked outputs."
    )
    bullets.append(
        f"`{q90_method}` reaches 90% of its own final best score fastest ({_fmt_float(q90_value)} queries on average), but this should only be treated as a speed metric after checking absolute final score."
    )

    significant = []
    weak = []
    for metric, tests in pairwise_tests.items():
        for comparison_name, values in tests.items():
            pvalue = values.get("wilcoxon_pvalue")
            mean_diff = values.get("mean_diff")
            if pvalue is None:
                continue
            record = (metric, comparison_name, float(mean_diff), float(pvalue))
            if pvalue < 0.05:
                significant.append(record)
            else:
                weak.append(record)

    if significant:
        bullets.append(
            "At least one paired Wilcoxon test is below 0.05, so some method differences are supported statistically in this run."
        )
    elif weak:
        bullets.append(
            "None of the paired Wilcoxon tests in `pairwise_tests.json` are below 0.05, so the current seed count is too small to make strong statistical claims."
        )

    hybrid_row = next((values for method, values in metrics.items() if method == "hybrid"), None)
    if hybrid_row is not None and best_method != "hybrid":
        bullets.append(
            "`hybrid` is currently faster to reach its own plateau than some baselines, but it underperforms on absolute score in the latest comparison, so the bottleneck is likely the hybrid setup rather than query efficiency alone."
        )

    real_query_rows = {
        method: float(values.get("real_oracle_queries", values.get("oracle_queries", {"mean": 0.0}))["mean"])
        for method, values in metrics.items()
    }
    if "hybrid" in real_query_rows and real_query_rows["hybrid"] > 0:
        cheapest_method = min(real_query_rows, key=real_query_rows.get)
        bullets.append(
            f"`{cheapest_method}` uses the fewest real-oracle queries on average ({_fmt_float(real_query_rows[cheapest_method])}), which matters most when the expensive oracle dominates cost."
        )

    fake_query_rows = {
        method: float(values.get("fake_oracle_queries", {"mean": 0.0})["mean"])
        for method, values in metrics.items()
    }
    if fake_query_rows.get("hybrid", 0.0) > 0.0:
        bullets.append(
            f"`hybrid` offloads part of its search to a cheap fake oracle ({_fmt_float(fake_query_rows['hybrid'])} fake-oracle queries on average), so its lower real-oracle cost should be interpreted together with that extra surrogate-side work."
        )

    if "gflownet_upper_bound" in metrics:
        bullets.append(
            "`gflownet_upper_bound` is an oracle-rich reference rather than a budget-matched baseline, so it should be read as an upper-bound target, not as a fair cost-controlled comparison."
        )

    return bullets


def _pairwise_table(pairwise_tests: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for metric, tests in pairwise_tests.items():
        for comparison_name, values in tests.items():
            rows.append(
                [
                    DISPLAY_NAMES.get(metric, metric),
                    comparison_name,
                    _fmt_float(values.get("mean_diff")),
                    _fmt_float(values.get("wilcoxon_pvalue")),
                    _fmt_float(values.get("t_pvalue")),
                    str(values.get("n", "-")),
                ]
            )
    return rows


def _cost_table(method_rows: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for row in sorted(method_rows, key=lambda item: item["real_oracle_queries"]):
        score_per_real_query = (
            row["best_score"] / row["real_oracle_queries"]
            if row["real_oracle_queries"] > 0
            else 0.0
        )
        rows.append(
            [
                row["method"],
                _fmt_float(row["real_oracle_queries"]),
                _fmt_float(row["fake_oracle_queries"]),
                _fmt_float(row["cheap_model_queries"]),
                _fmt_float(row["runtime_seconds"]),
                _fmt_float(score_per_real_query, digits=5),
            ]
        )
    return rows


def _count_seed_dirs(root: Path) -> int:
    return len([path for path in root.rglob("seed_*") if path.is_dir()])


def _ablation_best_variant(
    metrics: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> tuple[str, float]:
    return _winner(metrics, metric)


def _interpret_ablation(study_name: str, metrics: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    bullets: list[str] = []
    best_variant, best_score = _ablation_best_variant(metrics, "best_score")
    fast_variant, fast_q90 = _ablation_best_variant(metrics, "queries_to_90pct_best")
    top10_variant, top10_score = _ablation_best_variant(metrics, "top10_score")

    bullets.append(
        f"`{best_variant}` is the strongest setting on Best Score ({_fmt_float(best_score)})."
    )

    if top10_variant != best_variant:
        bullets.append(
            f"`{top10_variant}` leads on Top-10 Score ({_fmt_float(top10_score)}), so it may trade peak performance for stronger average quality in the top-ranked set."
        )

    if fast_variant != best_variant:
        bullets.append(
            f"`{fast_variant}` is the fastest on Queries To 90% Best ({_fmt_float(fast_q90)}), but speed should be interpreted together with final score."
        )

    unique_best = {float(values["best_score"]["mean"]) for values in metrics.values()}
    if len(unique_best) == 1:
        bullets = [
            "All tested settings are effectively identical on Best Score in this study, so the ablated parameter does not appear to be the limiting factor under the current configuration."
        ]

    if study_name.endswith("acquisition") and "thompson" in metrics:
        thompson = float(metrics["thompson"]["best_score"]["mean"])
        leaders = [
            float(values["best_score"]["mean"])
            for key, values in metrics.items()
            if key != "thompson"
        ]
        if leaders and thompson < max(leaders):
            bullets.append(
                "`thompson` is materially worse than the other acquisition choices in this study, so it is a weak default for the current setup."
            )

    return bullets


def _ablation_table(metrics: dict[str, dict[str, dict[str, float]]]) -> list[list[str]]:
    rows = []
    for variant, values in metrics.items():
        rows.append(
            [
                str(variant),
                _fmt_float(values["best_score"]["mean"]),
                _fmt_float(values["best_score"]["sem"]),
                _fmt_float(values["top10_score"]["mean"]),
                _fmt_float(values["top10_score"]["sem"]),
                _fmt_float(values["valid_word_ratio"]["mean"]),
                _fmt_float(values["queries_to_90pct_best"]["mean"]),
            ]
        )
    rows.sort(key=lambda row: float(row[1]), reverse=True)
    return rows


def generate_report(selection: RunSelection) -> str:
    sections: list[str] = ["# Results Report", ""]

    sections.append("## Sources")
    sections.append("")
    if selection.comparisons_dir is not None:
        sections.append(f"- Latest comparison run: `{selection.comparisons_dir}`")
    else:
        sections.append("- Latest comparison run: not found")
    if selection.ablations_dir is not None:
        sections.append(f"- Latest ablation run: `{selection.ablations_dir}`")
    else:
        sections.append("- Latest ablation run: not found")
    sections.append("")

    if selection.comparisons_dir is not None:
        comp_dir = selection.comparisons_dir
        comp_metrics = _load_json(comp_dir / "aggregate_metrics.json")
        pairwise = _load_json(comp_dir / "pairwise_tests.json")
        method_rows = _metric_table(comp_metrics)

        sections.append("## Cross-Method Comparison")
        sections.append("")
        sections.append(
            "Higher is better for `Best Score`, `Top-10 Score`, `Mean Score`, and `Valid Word Ratio`. Lower is better for `Queries To 90% Best`."
        )
        sections.append("")
        sections.append(
            _markdown_table(
                [
                    "Method",
                    "Best Score",
                    "Best SEM",
                    "Top-10 Score",
                    "Top-10 SEM",
                    "Mean Score",
                    "Valid Word Ratio",
                    "Queries To 90% Best",
                    "Q90 SEM",
                ],
                [
                    [
                        row["method"],
                        _fmt_float(row["best_score"]),
                        _fmt_float(row["best_score_sem"]),
                        _fmt_float(row["top10_score"]),
                        _fmt_float(row["top10_score_sem"]),
                        _fmt_float(row["mean_score"]),
                        _fmt_float(row["valid_word_ratio"]),
                        _fmt_float(row["queries_to_90pct_best"]),
                        _fmt_float(row["queries_to_90pct_best_sem"]),
                    ]
                    for row in method_rows
                ],
            )
        )
        sections.append("")
        sections.append("### Interpretation")
        sections.append("")
        for bullet in _interpret_comparison(comp_metrics, pairwise):
            sections.append(f"- {bullet}")
        sections.append("")
        sections.append("### Pairwise Tests")
        sections.append("")
        sections.append(
            _markdown_table(
                ["Metric", "Comparison", "Mean Diff", "Wilcoxon p", "t-test p", "n"],
                _pairwise_table(pairwise),
            )
        )
        sections.append("")
        sections.append("### Cost Comparison")
        sections.append("")
        sections.append(
            _markdown_table(
                [
                    "Method",
                    "Real Oracle Queries",
                    "Fake Oracle Queries",
                    "Cheap Model Queries",
                    "Runtime (s)",
                    "Best Score / Real Query",
                ],
                _cost_table(method_rows),
            )
        )
        sections.append("")
        sections.append("### Plots")
        sections.append("")
        sections.append(f"- `curve_best.png`: `{comp_dir / 'curve_best.png'}`")
        sections.append(f"- `curve_top10.png`: `{comp_dir / 'curve_top10.png'}`")
        sections.append("")

    if selection.ablations_dir is not None:
        abl_dir = selection.ablations_dir
        sections.append("## Ablation Studies")
        sections.append("")
        sections.append(
            "Each ablation table compares settings within a single study. The best setting on absolute score matters more than a low `Queries To 90% Best` value by itself."
        )
        sections.append("")

        study_dirs = sorted(
            path for path in abl_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        for study_dir in study_dirs:
            aggregate_path = study_dir / "aggregate_metrics.json"
            if not aggregate_path.exists():
                continue
            study_metrics = _load_json(aggregate_path)
            sections.append(f"### {study_dir.name}")
            sections.append("")
            sections.append(
                _markdown_table(
                    [
                        "Variant",
                        "Best Score",
                        "Best SEM",
                        "Top-10 Score",
                        "Top-10 SEM",
                        "Valid Word Ratio",
                        "Queries To 90% Best",
                    ],
                    _ablation_table(study_metrics),
                )
            )
            sections.append("")
            sections.append("Interpretation:")
            for bullet in _interpret_ablation(study_dir.name, study_metrics):
                sections.append(f"- {bullet}")
            sections.append(f"- Best-score plot: `{study_dir / 'curve_best.png'}`")
            sections.append(f"- Top-10 plot: `{study_dir / 'curve_top10.png'}`")
            sections.append("")

    sections.append("## Recommended Reading Order")
    sections.append("")
    sections.append("- Start with the cross-method comparison table to see which overall method is strongest.")
    sections.append("- Use the pairwise test table to decide whether the observed gaps are strong enough to claim.")
    sections.append("- Use the ablation tables to decide which knobs matter inside `active` and `hybrid`.")
    sections.append("- Use the PNG curve files when you need to explain query efficiency visually.")
    sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-dir", type=str, default=None)
    parser.add_argument("--ablations-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_ROOT / "reports" / "latest_results_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = _discover_runs(args.comparisons_dir, args.ablations_dir)
    report = generate_report(selection)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
