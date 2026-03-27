"""Run multi-seed method comparisons and aggregate results."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
GFLOWNET_ROOT = (PROJECT_ROOT / "../gflownet").resolve()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GFLOWNET_ROOT) not in sys.path:
    sys.path.insert(0, str(GFLOWNET_ROOT))

from training.dataset import generate_random_dataset
from training.train_active import run_active_learning
from training.train_baseline import run_supervised_baseline
from training.train_gflownet import run_oracle_gflownet
from training.train_hybrid import run_hybrid_gflownet_active
from utils.results import aggregate_method_curves, aggregate_method_summaries, pairwise_metric_tests
from utils.visualization import plot_query_curves


def _seed_config(base_cfg: dict, seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    return cfg


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@hydra.main(config_path="../configs", config_name="comparisons", version_base="1.1")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    methods = set(cfg_dict["comparison"]["methods"])
    summaries_by_method: dict[str, list[dict]] = {
        "baseline": [],
        "active": [],
        "gflownet": [],
        "hybrid": [],
    }

    for seed in cfg_dict["seeds"]:
        seed_cfg = _seed_config(cfg_dict, seed)

        dataset_dir = run_dir / "dataset" / f"seed_{seed}"
        dataset_result = generate_random_dataset(seed_cfg, output_dir=dataset_dir, logger=None)

        if "baseline" in methods:
            baseline_cfg = copy.deepcopy(seed_cfg)
            baseline_cfg["dataset"] = copy.deepcopy(seed_cfg["dataset"])
            baseline_cfg["dataset"]["path"] = dataset_result["dataset_path"]
            baseline_dir = run_dir / "baseline" / f"seed_{seed}"
            summaries_by_method["baseline"].append(
                run_supervised_baseline(baseline_cfg, output_dir=baseline_dir, logger=None)
            )

        if "active" in methods:
            active_dir = run_dir / "active" / f"seed_{seed}"
            summaries_by_method["active"].append(
                run_active_learning(seed_cfg, output_dir=active_dir, logger=None)
            )

        if "gflownet" in methods:
            gflownet_dir = run_dir / "gflownet" / f"seed_{seed}"
            summaries_by_method["gflownet"].append(
                run_oracle_gflownet(seed_cfg, output_dir=gflownet_dir, logger=None)
            )

        if "hybrid" in methods:
            hybrid_dir = run_dir / "hybrid" / f"seed_{seed}"
            summaries_by_method["hybrid"].append(
                run_hybrid_gflownet_active(seed_cfg, output_dir=hybrid_dir, logger=None)
            )

    summaries_by_method = {
        method: summaries
        for method, summaries in summaries_by_method.items()
        if len(summaries) > 0
    }

    metrics = cfg_dict["comparison"]["metrics"]
    aggregated_metrics = aggregate_method_summaries(summaries_by_method, metrics=metrics)
    _save_json(run_dir / "aggregate_metrics.json", aggregated_metrics)

    budget = int(cfg_dict["oracle"]["budget"])
    for curve_key in cfg_dict["comparison"]["curve_keys"]:
        curves = aggregate_method_curves(
            summaries_by_method,
            curve_key=curve_key,
            budget=budget,
        )
        _save_json(run_dir / f"curve_{curve_key}.json", curves)
        plot_query_curves(
            curves,
            title=f"{curve_key} vs oracle queries",
            ylabel=curve_key,
            output_path=run_dir / f"curve_{curve_key}.png",
        )

    reference_method = str(cfg_dict["comparison"]["reference_method"])
    pairwise_tests = {}
    for metric in metrics:
        if reference_method not in summaries_by_method:
            continue
        pairwise_tests[metric] = pairwise_metric_tests(
            summaries_by_method,
            metric=metric,
            reference_method=reference_method,
        )
    _save_json(run_dir / "pairwise_tests.json", pairwise_tests)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "methods": list(summaries_by_method.keys()),
                "aggregate_metrics": aggregated_metrics,
                "pairwise_tests": pairwise_tests,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
