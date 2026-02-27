"""Lightweight experiment logging utilities."""

from __future__ import annotations

import json
import logging as pylogging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(slots=True)
class LoggingConfig:
    """Configuration for experiment logging."""

    output_dir: Path
    run_name: str
    use_wandb: bool = False
    wandb_project: str = "gfn-active-learning"
    wandb_entity: str | None = None


class ExperimentLogger:
    """Structured logger that writes tabular metrics and optional WandB logs."""

    def __init__(self, config: LoggingConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

        self._logger = pylogging.getLogger(config.run_name)
        self._logger.setLevel(pylogging.INFO)
        self._logger.handlers.clear()
        file_handler = pylogging.FileHandler(self.output_dir / "run.log")
        file_handler.setFormatter(
            pylogging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        self._logger.addHandler(file_handler)

        self._wandb = None
        if config.use_wandb:
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.run_name,
                    dir=str(self.output_dir),
                )
            except Exception as exc:  # pragma: no cover - optional dependency path
                self._logger.warning("WandB disabled due to import/init failure: %s", exc)

    def info(self, message: str) -> None:
        """Log free-form informational message."""
        self._logger.info(message)

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics for a given step."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": int(step),
            **metrics,
        }
        self._records.append(payload)
        self._logger.info("step=%s metrics=%s", step, metrics)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def to_frame(self) -> pd.DataFrame:
        """Return logged metrics as a DataFrame."""
        return pd.DataFrame(self._records)

    def dump_metrics(self, filename: str = "metrics.csv") -> Path:
        """Persist tabular metrics to disk."""
        path = self.output_dir / filename
        self.to_frame().to_csv(path, index=False)
        return path

    def dump_summary(self, summary: dict[str, Any], filename: str = "summary.json") -> Path:
        """Persist summary artifacts to disk."""
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self._wandb is not None:
            self._wandb.summary.update(summary)
        return path

    def close(self) -> None:
        """Finalize logger resources."""
        if self._wandb is not None:
            self._wandb.finish()


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
