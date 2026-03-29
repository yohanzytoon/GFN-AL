"""Expected Improvement acquisition."""

from __future__ import annotations

import math
import numpy as np


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Compute Expected Improvement for each candidate."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    safe_std = np.maximum(std, 1e-12)

    improvement = mean - best_f - xi
    z = improvement / safe_std

    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.erf(z / math.sqrt(2.0)))

    ei = improvement * cdf + safe_std * pdf
    ei[std <= 1e-12] = np.maximum(improvement[std <= 1e-12], 0.0)

    return ei


def select_ei(
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    best_f: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Select top candidates by Expected Improvement."""
    scores = expected_improvement(mean, std, best_f, xi)

    batch_size = min(batch_size, len(scores))

    idx = np.argpartition(scores, -batch_size)[-batch_size:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    return idx
