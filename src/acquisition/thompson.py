"""Thompson-sampling acquisition helpers."""

from __future__ import annotations

import numpy as np


def select_thompson(
    draws: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Select top candidates from one Thompson draw."""
    values = np.asarray(draws, dtype=float).reshape(-1)

    batch_size = min(batch_size, len(values))

    idx = np.argpartition(values, -batch_size)[-batch_size:]
    idx = idx[np.argsort(values[idx])[::-1]]

    return idx
