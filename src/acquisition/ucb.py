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
    """Select top candidates by UCB."""
    scores = ucb_scores(mean, std, beta)

    batch_size = min(batch_size, len(scores))

    idx = np.argpartition(scores, -batch_size)[-batch_size:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    return idx
