"""Factory helpers for acquisition-function selection."""

from __future__ import annotations

from typing import Any
import numpy as np

from acquisition.ei import select_ei
from acquisition.thompson import select_thompson
from acquisition.ucb import select_ucb
from acquisition.uncertainty import select_uncertainty


def select_acquisition_batch(
    acquisition: str,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    best_f: float,
    surrogate: Any | None = None,
    candidate_states: np.ndarray | None = None,
    beta: float = 2.0,
    xi: float = 0.0,
    thompson_samples: int = 1,
) -> np.ndarray:
    """Dispatch batch selection to the requested acquisition function."""

    name = acquisition.lower()

    if name == "ucb":
        return select_ucb(mean, std, batch_size, beta)

    if name in {"uncertainty", "uncertainty_sampling"}:
        return select_uncertainty(std, batch_size)

    if name == "ei":
        return select_ei(mean, std, batch_size, best_f, xi)

    if name == "thompson":
        if surrogate is None or candidate_states is None:
            raise ValueError("Thompson requires surrogate and candidate_states.")

        draws = surrogate.sample(
            np.asarray(candidate_states, dtype=np.int64),
            n_samples=max(thompson_samples, 1),
        )

        return select_thompson(draws[0], batch_size)

    raise ValueError(
        f"Unsupported acquisition function: {acquisition}. "
        "Use one of {'ucb', 'uncertainty', 'ei', 'thompson'}."
    )
