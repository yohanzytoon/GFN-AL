"""Uncertainty-sampling acquisition."""

from __future__ import annotations

import numpy as np


def select_uncertainty(std: np.ndarray, batch_size: int) -> np.ndarray:
    """Select the most uncertain candidates."""
    values = np.asarray(std, dtype=float)

    batch_size = min(batch_size, len(values))

    idx = np.argpartition(values, -batch_size)[-batch_size:]
    idx = idx[np.argsort(values[idx])[::-1]]

    return idx
