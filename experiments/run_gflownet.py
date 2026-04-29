"""Run oracle-access GFlowNet experiment."""

from __future__ import annotations

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

from training.train_gflownet import run_oracle_gflownet
from utils.logging import ExperimentLogger, LoggingConfig, print_result_summary


@hydra.main(config_path="../configs", config_name="gflownet", version_base="1.1")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    logger = ExperimentLogger(
        LoggingConfig(
            output_dir=run_dir,
            run_name=f"gflownet_seed_{cfg_dict['seed']}",
        )
    )
    result = run_oracle_gflownet(cfg_dict, output_dir=run_dir, logger=logger)
    logger.dump_metrics("metrics_gflownet.csv")
    logger.dump_summary(result, filename="summary_gflownet.json")
    logger.close()

    print_result_summary(result)


if __name__ == "__main__":
    main()
