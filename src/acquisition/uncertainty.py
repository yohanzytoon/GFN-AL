"""Uncertainty-sampling acquisition."""

from __future__ import annotations

import numpy as np


def select_uncertainty(std: np.ndarray, batch_size: int) -> np.ndarray:
    """Select the most uncertain candidates."""
    values = np.asarray(std, dtype=float)
    order = np.argsort(values)[::-1]
    return order[:batch_size]
