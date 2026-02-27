"""Expected Improvement acquisition."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute EI for each candidate."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    std_safe = np.maximum(std, 1e-9)
    improvement = mean - best_observed - xi
    z = improvement / std_safe
    ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
    ei[std <= 0] = 0.0
    return ei


def select_ei(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    batch_size: int,
    xi: float = 0.01,
) -> np.ndarray:
    """Select top batch_size candidates by EI."""
    scores = expected_improvement(
        mean=mean,
        std=std,
        best_observed=best_observed,
        xi=xi,
    )
    order = np.argsort(scores)[::-1]
    return order[:batch_size]
