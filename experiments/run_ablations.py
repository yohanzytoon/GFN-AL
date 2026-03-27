"""Run ablation studies and aggregate results."""

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

from training.train_active import run_active_learning
from training.train_hybrid import run_hybrid_gflownet_active
from utils.results import aggregate_method_curves, aggregate_method_summaries
from utils.visualization import plot_query_curves


def _set_nested(config: dict, dotted_key: str, value) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor[key]
    cursor[parts[-1]] = value


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@hydra.main(config_path="../configs", config_name="ablations", version_base="1.1")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for study in cfg_dict["ablations"]:
        study_name = str(study["name"])
        method = str(study["method"])
        parameter = str(study["parameter"])
        values = list(study["values"])
        study_dir = run_dir / study_name
        summaries_by_value: dict[str, list[dict]] = {}

        for value in values:
            value_key = str(value)
            summaries_by_value[value_key] = []
            for seed in cfg_dict["seeds"]:
                seed_cfg = copy.deepcopy(cfg_dict)
                seed_cfg["seed"] = int(seed)
                _set_nested(seed_cfg, parameter, value)
                value_dir = study_dir / value_key / f"seed_{seed}"
                if method == "active":
                    summary = run_active_learning(seed_cfg, output_dir=value_dir, logger=None)
                elif method == "hybrid":
                    summary = run_hybrid_gflownet_active(seed_cfg, output_dir=value_dir, logger=None)
                else:
                    raise ValueError(f"Unsupported ablation method: {method}")
                summaries_by_value[value_key].append(summary)

        aggregated_metrics = aggregate_method_summaries(
            summaries_by_value,
            metrics=cfg_dict["analysis"]["metrics"],
        )
        _save_json(study_dir / "aggregate_metrics.json", aggregated_metrics)

        budget = int(cfg_dict["oracle"]["budget"])
        for curve_key in cfg_dict["analysis"]["curve_keys"]:
            curves = aggregate_method_curves(
                summaries_by_value,
                curve_key=curve_key,
                budget=budget,
            )
            _save_json(study_dir / f"curve_{curve_key}.json", curves)
            plot_query_curves(
                curves,
                title=f"{study_name}: {curve_key}",
                ylabel=curve_key,
                output_path=study_dir / f"curve_{curve_key}.png",
            )

    print(json.dumps({"run_dir": str(run_dir), "studies": cfg_dict["ablations"]}, indent=2))


if __name__ == "__main__":
    main()
