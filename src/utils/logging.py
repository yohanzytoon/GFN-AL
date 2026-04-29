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
        self._logger.propagate = False
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
        """Record metrics for a given step (stored in memory for CSV export, not logged to file)."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": int(step),
            **metrics,
        }
        self._records.append(payload)

    def dump_metrics(self, filename: str = "metrics.csv") -> Path:
        """Persist tabular metrics to disk."""
        path = self.output_dir / filename
        if not self._records:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames: list[str] = []
        for record in self._records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
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
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


_BULKY_RESULT_KEYS = frozenset({
    "scores", "round_logs", "curve", "sampled_states", "sampled_scores",
    "bootstrap_logs", "optimum_words", "gflownet_round_dirs",
    "surrogate_path", "checkpoint_dir",
})


def print_result_summary(result: dict[str, Any]) -> None:
    """Print a concise terminal summary, omitting large arrays and logs."""
    slim = {k: v for k, v in result.items() if k not in _BULKY_RESULT_KEYS}
    import json as _json
    print(_json.dumps(slim, indent=2))


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
