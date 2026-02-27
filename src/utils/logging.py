"""Lightweight experiment logging utilities."""

from __future__ import annotations

import csv
import json
import logging as pylogging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class LoggingConfig:
    """Configuration for experiment logging."""

    output_dir: Path
    run_name: str


class ExperimentLogger:
    """Structured logger that writes local files only."""

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

    def dump_metrics(self, filename: str = "metrics.csv") -> Path:
        """Persist tabular metrics to disk."""
        path = self.output_dir / filename
        if not self._records:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = list(self._records[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        return path

    def dump_summary(self, summary: dict[str, Any], filename: str = "summary.json") -> Path:
        """Persist summary artifacts to disk."""
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return path

    def close(self) -> None:
        """Finalize logger resources."""
        return None


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
