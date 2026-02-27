"""Run full comparator suite and aggregate results/statistics."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
GFLOWNET_ROOT = (PROJECT_ROOT / "../gflownet").resolve()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GFLOWNET_ROOT) not in sys.path:
    sys.path.insert(0, str(GFLOWNET_ROOT))

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from training.train_active import run_active_learning
from training.train_baseline import run_supervised_baseline
from training.train_hybrid import run_hybrid_learning, run_pure_gflownet
from utils.logging import ExperimentLogger, LoggingConfig
from utils.metrics import (
    aggregate_curves_with_ci,
    auc,
    build_query_curve,
    pairwise_method_tests,
    running_regret,
    search_quality_metrics,
)
from utils.visualization import plot_metric_curves_with_ci


def run_random_search(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Random-search baseline with budgeted oracle access."""
    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    random_cfg = config.get("random_search", {})

    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=int(oracle_cfg["budget"]),
        device=config.get("device", "cpu"),
    )
    oracle = OracleProxy(
        device=config.get("device", "cpu"),
        backend="oracle",
        oracle_budget=int(oracle_cfg["budget"]),
        enforce_budget=True,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
    )
    oracle.setup(env)

    n_queries = int(
        min(oracle_cfg["budget"], random_cfg.get("num_queries", oracle_cfg["budget"]))
    )
    states = env.get_random_terminating_states(
        n_states=n_queries,
        unique=False,
        max_attempts=max(5 * n_queries, 1000),
    )
    scores = oracle(env.states2proxy(states)).detach().cpu().numpy().astype(float).tolist()

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=scores,
        states=states,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    return {
        "method": "random_search",
        "seed": int(config["seed"]),
        **quality,
        "curve": curve,
        "scores": scores,
    }


def _record_result(
    result: dict[str, Any],
    records: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    method = str(result["method"])
    seed = int(result["seed"])
    result_path = run_dir / f"{method}_seed_{seed}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    curve = result.get("curve", {})
    records.append(
        {
            "method": method,
            "seed": seed,
            "oracle_queries": int(result.get("oracle_queries", 0)),
            "best_score": float(result.get("best_score", 0.0)),
            "top10_score": float(result.get("top10_score", 0.0)),
            "mean_score": float(result.get("mean_score", 0.0)),
            "valid_word_ratio": float(result.get("valid_word_ratio", 0.0)),
            "diversity_entropy": float(result.get("diversity_entropy", 0.0)),
            "unique_fraction": float(result.get("unique_fraction", 0.0)),
            "mode_coverage": float(result.get("mode_coverage", 0.0)),
            "topk_edit_distance": float(result.get("topk_edit_distance", 0.0)),
            "queries_to_90pct_best": int(result.get("queries_to_90pct_best", 0)),
            "simple_regret": float(result.get("simple_regret", 0.0)),
            "auc_best": float(auc(curve.get("best", []), curve.get("queries", None))),
            "auc_top10": float(auc(curve.get("top10", []), curve.get("queries", None))),
            "result_path": str(result_path),
        }
    )


def _method_summary_with_ci(frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_df in frame.groupby("method"):
        row: dict[str, Any] = {"method": method, "n_seeds": int(method_df.shape[0])}
        for metric in metric_cols:
            values = method_df[metric].to_numpy(dtype=float)
            if values.size == 0:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0
                row[f"{metric}_ci95"] = 0.0
                continue
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(values.size)) if values.size > 1 else 0.0
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method")


def _build_curve_stats_frame(curve_stats: dict[str, dict[str, np.ndarray]], metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, stats in curve_stats.items():
        x = stats["x"]
        mean = stats["mean"]
        lower = stats["lower"]
        upper = stats["upper"]
        for idx in range(len(x)):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "query": int(x[idx]),
                    "mean": float(mean[idx]),
                    "lower": float(lower[idx]),
                    "upper": float(upper[idx]),
                }
            )
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs/experiments", config_name="comparisons", version_base="1.1")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(
        LoggingConfig(
            output_dir=run_dir,
            run_name="comparisons",
            use_wandb=bool(cfg_dict["logging"].get("use_wandb", False)),
            wandb_project=str(cfg_dict["logging"].get("wandb_project", "gfn-active-learning")),
            wandb_entity=cfg_dict["logging"].get("wandb_entity"),
        )
    )

    methods = [str(m) for m in cfg_dict["comparison"]["methods"]]
    seeds = [int(s) for s in cfg_dict["comparison"]["seeds"]]
    budget = int(cfg_dict["base_config"]["oracle"]["budget"])

    records: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for method in methods:
        for seed in seeds:
            cfg_seed = copy.deepcopy(cfg_dict["base_config"])
            cfg_seed["seed"] = seed
            method_out_dir = run_dir / method / f"seed_{seed}"
            method_out_dir.mkdir(parents=True, exist_ok=True)

            if method == "random_search":
                result = run_random_search(cfg_seed, output_dir=method_out_dir)
            elif method == "supervised_baseline":
                result = run_supervised_baseline(
                    cfg_seed,
                    output_dir=method_out_dir,
                    logger=None,
                )
            elif method == "pure_gflownet":
                result = run_pure_gflownet(
                    cfg_seed,
                    project_root=PROJECT_ROOT,
                    output_dir=method_out_dir,
                    logger=None,
                )
            elif method == "active_learning":
                result = run_active_learning(cfg_seed, output_dir=method_out_dir, logger=None)
            elif method == "hybrid":
                result = run_hybrid_learning(
                    cfg_seed,
                    project_root=PROJECT_ROOT,
                    output_dir=method_out_dir,
                    logger=None,
                )
            else:
                raise KeyError(f"Unknown method '{method}'")

            all_results.append(result)
            _record_result(result=result, records=records, run_dir=run_dir)
            logger.log_metrics(
                step=len(records),
                metrics={
                    "method": method,
                    "seed": seed,
                    "best_score": float(result.get("best_score", 0.0)),
                    "oracle_queries": int(result.get("oracle_queries", 0)),
                    "valid_word_ratio": float(result.get("valid_word_ratio", 0.0)),
                },
            )

    frame = pd.DataFrame(records)
    frame_path = run_dir / "comparison_results.csv"
    frame.to_csv(frame_path, index=False)

    metric_cols = [
        "best_score",
        "top10_score",
        "mean_score",
        "valid_word_ratio",
        "diversity_entropy",
        "unique_fraction",
        "mode_coverage",
        "topk_edit_distance",
        "queries_to_90pct_best",
        "simple_regret",
        "oracle_queries",
    ]
    summary_frame = _method_summary_with_ci(frame, metric_cols=metric_cols)
    summary_path = run_dir / "method_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    # Pairwise tests for publication-table metrics
    test_metrics = [
        str(m)
        for m in cfg_dict["comparison"].get(
            "stat_metrics",
            ["best_score", "top10_score", "mean_score", "valid_word_ratio"],
        )
    ]
    reference_method = str(cfg_dict["comparison"].get("reference_method", "hybrid"))
    pairwise_tests: dict[str, dict[str, dict[str, float]]] = {}
    for metric in test_metrics:
        values_by_method: dict[str, list[float]] = {}
        for method in methods:
            values_by_method[method] = (
                frame[frame["method"] == method].sort_values("seed")[metric].tolist()
            )
        pairwise_tests[metric] = pairwise_method_tests(
            values_by_method,
            reference_method=reference_method,
        )

    tests_path = run_dir / "pairwise_tests.json"
    with tests_path.open("w", encoding="utf-8") as f:
        json.dump(pairwise_tests, f, indent=2)

    # Build regret curves against empirical global best across all runs
    global_best = float(frame["best_score"].max()) if not frame.empty else 0.0
    for result in all_results:
        curve = result.get("curve", {})
        if "regret" not in curve:
            best_curve = np.asarray(curve.get("best", []), dtype=float)
            if best_curve.size > 0:
                curve["regret"] = running_regret(best_curve, optimum_score=global_best).tolist()
            else:
                curve["regret"] = []

    curve_stats_best: dict[str, dict[str, np.ndarray]] = {}
    curve_stats_top10: dict[str, dict[str, np.ndarray]] = {}
    curve_stats_regret: dict[str, dict[str, np.ndarray]] = {}

    for method in methods:
        method_curves = [r.get("curve", {}) for r in all_results if r["method"] == method]
        if len(method_curves) == 0:
            continue
        curve_stats_best[method] = aggregate_curves_with_ci(method_curves, budget=budget, key="best")
        curve_stats_top10[method] = aggregate_curves_with_ci(method_curves, budget=budget, key="top10")
        curve_stats_regret[method] = aggregate_curves_with_ci(
            method_curves,
            budget=budget,
            key="regret",
        )

    plot_metric_curves_with_ci(
        curve_stats=curve_stats_best,
        output_path=run_dir / "best_vs_queries_ci.png",
        xlabel="Oracle Queries",
        ylabel="Best Score",
        title="Best Score vs Oracle Queries (95% CI)",
    )
    plot_metric_curves_with_ci(
        curve_stats=curve_stats_top10,
        output_path=run_dir / "top10_vs_queries_ci.png",
        xlabel="Oracle Queries",
        ylabel="Top-10 Mean Score",
        title="Top-10 Score vs Oracle Queries (95% CI)",
    )
    plot_metric_curves_with_ci(
        curve_stats=curve_stats_regret,
        output_path=run_dir / "regret_vs_queries_ci.png",
        xlabel="Oracle Queries",
        ylabel="Simple Regret",
        title="Simple Regret vs Oracle Queries (95% CI)",
    )

    curve_stats_frame = pd.concat(
        [
            _build_curve_stats_frame(curve_stats_best, metric="best"),
            _build_curve_stats_frame(curve_stats_top10, metric="top10"),
            _build_curve_stats_frame(curve_stats_regret, metric="regret"),
        ],
        axis=0,
        ignore_index=True,
    )
    curve_stats_path = run_dir / "curve_statistics.csv"
    curve_stats_frame.to_csv(curve_stats_path, index=False)

    summary = {
        "results_csv": str(frame_path),
        "summary_csv": str(summary_path),
        "curve_stats_csv": str(curve_stats_path),
        "tests_json": str(tests_path),
        "n_runs": len(records),
        "methods": methods,
        "seeds": seeds,
        "reference_method": reference_method,
        "global_best_score": global_best,
    }
    logger.dump_summary(summary, filename="summary_comparisons.json")
    logger.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
