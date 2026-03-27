"""Thompson-sampling acquisition helpers."""

from __future__ import annotations

import numpy as np


def select_thompson(
    draws: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Select top candidates from one Thompson draw."""
    values = np.asarray(draws, dtype=float).reshape(-1)
    order = np.argsort(values)[::-1]
    return order[:batch_size]
