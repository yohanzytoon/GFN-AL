"""Thompson sampling acquisition."""

from __future__ import annotations

import numpy as np


def thompson_draw(
    mean: np.ndarray,
    std: np.ndarray,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Single Thompson draw from independent Gaussian marginals."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    rng = random_state or np.random.default_rng()
    return rng.normal(loc=mean, scale=np.maximum(std, 1e-9))


def select_thompson(
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Select top batch_size candidates based on Thompson draws."""
    draws = thompson_draw(mean=mean, std=std, random_state=random_state)
    order = np.argsort(draws)[::-1]
    return order[:batch_size]
