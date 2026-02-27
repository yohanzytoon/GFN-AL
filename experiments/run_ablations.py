"""Run systematic ablation sweeps for Active Learning and Hybrid methods."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
GFLOWNET_ROOT = (PROJECT_ROOT / "../gflownet").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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
from utils.metrics import build_query_curve, search_quality_metrics


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


def _apply_ablation_override(cfg_seed: dict[str, Any], sweep_name: str, sweep_value: Any) -> None:
    if sweep_name == "acquisition":
        cfg_seed["active"]["acquisition"] = sweep_value
    elif sweep_name == "surrogate_type":
        cfg_seed["active"]["surrogate_type"] = sweep_value
    elif sweep_name == "gflownet_temperature":
        cfg_seed["gflownet"]["temperature_logits"] = float(sweep_value)
    elif sweep_name == "batch_size":
        cfg_seed["active"]["batch_size"] = int(sweep_value)
    elif sweep_name == "initial_size":
        cfg_seed["active"]["initial_size"] = int(sweep_value)
    elif sweep_name == "retrain_frequency":
        cfg_seed["hybrid"]["retrain_frequency"] = int(sweep_value)
    else:
        raise KeyError(f"Unknown sweep name: {sweep_name}")


def _run_method(method: str, cfg_seed: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if method == "random_search":
        return run_random_search(cfg_seed, output_dir=output_dir)
    if method == "supervised_baseline":
        return run_supervised_baseline(cfg_seed, output_dir=output_dir, logger=None)
    if method == "pure_gflownet":
        return run_pure_gflownet(
            cfg_seed,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            logger=None,
        )
    if method == "active_learning":
        return run_active_learning(cfg_seed, output_dir=output_dir, logger=None)
    if method == "hybrid":
        return run_hybrid_learning(
            cfg_seed,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            logger=None,
        )
    raise KeyError(f"Unknown method: {method}")


def _is_sweep_applicable(method: str, sweep_name: str) -> bool:
    """Return whether a sweep is meaningful for a method."""
    if sweep_name == "retrain_frequency":
        return method == "hybrid"
    if sweep_name == "gflownet_temperature":
        return method in {"pure_gflownet", "hybrid"}
    return True


def _plot_ablation_curve(frame: pd.DataFrame, sweep_name: str, metric: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agg = (
        frame.groupby(["method", "sweep_value"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["method", "sweep_value"])
    )
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
    categories = sorted(frame["sweep_value"].astype(str).unique().tolist())
    x_map = {value: idx for idx, value in enumerate(categories)}

    plt.figure(figsize=(8, 5))
    for method, method_df in agg.groupby("method"):
        x_vals = method_df["sweep_value"].astype(str).tolist()
        mean_vals = method_df["mean"].to_numpy(dtype=float)
        ci_vals = method_df["ci95"].to_numpy(dtype=float)
        x_idx = np.asarray([x_map[value] for value in x_vals], dtype=float)
        plt.plot(x_idx, mean_vals, marker="o", label=str(method))
        plt.fill_between(x_idx, mean_vals - ci_vals, mean_vals + ci_vals, alpha=0.2)
    plt.xticks(np.arange(len(categories)), categories)

    plt.xlabel(sweep_name)
    plt.ylabel(metric)
    plt.title(f"Ablation: {sweep_name} vs {metric}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@hydra.main(config_path="../configs/experiments", config_name="ablations", version_base="1.1")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(
        LoggingConfig(
            output_dir=run_dir,
            run_name="ablations",
            use_wandb=bool(cfg_dict["logging"].get("use_wandb", False)),
            wandb_project=str(cfg_dict["logging"].get("wandb_project", "gfn-active-learning")),
            wandb_entity=cfg_dict["logging"].get("wandb_entity"),
        )
    )

    methods = [str(m) for m in cfg_dict["ablation"]["methods"]]
    seeds = [int(s) for s in cfg_dict["ablation"]["seeds"]]
    sweeps = cfg_dict["ablation"]["sweeps"]
    primary_metric = str(cfg_dict["ablation"].get("primary_metric", "best_score"))

    records: list[dict[str, Any]] = []

    for sweep_name, sweep_values in sweeps.items():
        for sweep_value in sweep_values:
            for method in methods:
                if not _is_sweep_applicable(method=method, sweep_name=sweep_name):
                    continue
                for seed in seeds:
                    cfg_seed = copy.deepcopy(cfg_dict["base_config"])
                    cfg_seed["seed"] = seed
                    _apply_ablation_override(cfg_seed, sweep_name, sweep_value)

                    output_dir = run_dir / sweep_name / str(sweep_value) / method / f"seed_{seed}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    result = _run_method(method=method, cfg_seed=cfg_seed, output_dir=output_dir)
                    record = {
                        "sweep_name": sweep_name,
                        "sweep_value": sweep_value,
                        "method": method,
                        "seed": seed,
                        "best_score": float(result.get("best_score", 0.0)),
                        "top10_score": float(result.get("top10_score", 0.0)),
                        "mean_score": float(result.get("mean_score", 0.0)),
                        "valid_word_ratio": float(result.get("valid_word_ratio", 0.0)),
                        "diversity_entropy": float(result.get("diversity_entropy", 0.0)),
                        "oracle_queries": int(result.get("oracle_queries", 0)),
                        "result_path": str(output_dir / "result.json"),
                    }
                    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    records.append(record)

                    logger.log_metrics(
                        step=len(records),
                        metrics={
                            "sweep_name": sweep_name,
                            "sweep_value": str(sweep_value),
                            "method": method,
                            "seed": seed,
                            "best_score": record["best_score"],
                            "oracle_queries": record["oracle_queries"],
                        },
                    )

    frame = pd.DataFrame(records)
    results_path = run_dir / "ablation_results.csv"
    frame.to_csv(results_path, index=False)

    summary = (
        frame.groupby(["sweep_name", "sweep_value", "method"])[
            ["best_score", "top10_score", "mean_score", "valid_word_ratio", "diversity_entropy", "oracle_queries"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_path = run_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    for sweep_name in frame["sweep_name"].unique():
        subset = frame[frame["sweep_name"] == sweep_name].copy()
        _plot_ablation_curve(
            frame=subset,
            sweep_name=sweep_name,
            metric=primary_metric,
            output_path=run_dir / f"ablation_{sweep_name}_{primary_metric}.png",
        )

    payload = {
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
        "n_runs": int(frame.shape[0]),
        "methods": methods,
        "seeds": seeds,
        "primary_metric": primary_metric,
    }
    logger.dump_summary(payload, filename="summary_ablations.json")
    logger.close()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
