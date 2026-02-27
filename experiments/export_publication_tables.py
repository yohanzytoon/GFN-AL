"""Export publication-ready LaTeX/Markdown tables from comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _format_mean_ci(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def build_method_table(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create method summary tables in LaTeX and Markdown."""
    frame = pd.read_csv(summary_path)

    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "method": row["method"],
                "best_score": _format_mean_ci(row["best_score_mean"], row["best_score_ci95"]),
                "top10_score": _format_mean_ci(row["top10_score_mean"], row["top10_score_ci95"]),
                "mean_score": _format_mean_ci(row["mean_score_mean"], row["mean_score_ci95"]),
                "valid_ratio": _format_mean_ci(row["valid_word_ratio_mean"], row["valid_word_ratio_ci95"]),
                "div_entropy": _format_mean_ci(row["diversity_entropy_mean"], row["diversity_entropy_ci95"]),
                "queries": _format_mean_ci(row["oracle_queries_mean"], row["oracle_queries_ci95"], digits=1),
            }
        )

    table_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / "table_method_summary.tex"
    md_path = output_dir / "table_method_summary.md"

    latex = table_df.to_latex(index=False, escape=False)
    tex_path.write_text(latex, encoding="utf-8")
    md_path.write_text(table_df.to_markdown(index=False), encoding="utf-8")

    return tex_path, md_path


def build_pairwise_table(tests_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create pairwise tests tables in LaTeX and Markdown."""
    payload = json.loads(tests_path.read_text(encoding="utf-8"))

    rows = []
    for metric, comparisons in payload.items():
        for comparison, values in comparisons.items():
            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "t_pvalue": float(values.get("t_pvalue", 1.0)),
                    "wilcoxon_pvalue": float(values.get("wilcoxon_pvalue", 1.0)),
                    "cohen_d": float(values.get("cohen_d_paired", 0.0)),
                    "mean_diff": float(values.get("mean_diff", 0.0)),
                    "n": int(values.get("n", 0)),
                }
            )

    table_df = pd.DataFrame(rows).sort_values(["metric", "comparison"])

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "table_pairwise_tests.tex"
    md_path = output_dir / "table_pairwise_tests.md"

    latex = table_df.to_latex(index=False, float_format=lambda x: f"{x:.4g}", escape=False)
    tex_path.write_text(latex, encoding="utf-8")
    md_path.write_text(table_df.to_markdown(index=False), encoding="utf-8")

    return tex_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a completed comparisons run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path where tables are written. Default: <run-dir>/tables",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "tables"

    summary_path = run_dir / "method_summary.csv"
    tests_path = run_dir / "pairwise_tests.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing method summary CSV: {summary_path}")
    if not tests_path.exists():
        raise FileNotFoundError(f"Missing pairwise tests JSON: {tests_path}")

    method_tex, method_md = build_method_table(summary_path, output_dir)
    tests_tex, tests_md = build_pairwise_table(tests_path, output_dir)

    print("Generated tables:")
    print(f"- {method_tex}")
    print(f"- {method_md}")
    print(f"- {tests_tex}")
    print(f"- {tests_md}")


if __name__ == "__main__":
    main()
