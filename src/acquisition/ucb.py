"""Upper Confidence Bound acquisition."""

from __future__ import annotations

import numpy as np


def ucb_scores(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Compute UCB score for each candidate."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return mean + beta * std


def select_ucb(
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    beta: float = 2.0,
) -> np.ndarray:
    """Select top batch_size candidates by UCB."""
    scores = ucb_scores(mean=mean, std=std, beta=beta)
    order = np.argsort(scores)[::-1]
    return order[:batch_size]
