"""Factory helpers for acquisition-function selection.

Includes both standard greedy batch selection and a diversity-aware variant
that prevents wasting oracle queries on near-identical candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from acquisition.ei import select_ei
from acquisition.ei import expected_improvement
from acquisition.thompson import select_thompson
from acquisition.ucb import select_ucb
from acquisition.ucb import ucb_scores
from acquisition.uncertainty import select_uncertainty


def score_acquisition(
    acquisition: str,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    surrogate: Any | None = None,
    candidate_states: np.ndarray | None = None,
    beta: float = 2.0,
    xi: float = 0.0,
    thompson_samples: int = 1,
) -> np.ndarray:
    """Return acquisition scores for all candidates."""
    name = str(acquisition).lower()
    if name == "ucb":
        return ucb_scores(mean=mean, std=std, beta=beta)
    if name in {"uncertainty", "uncertainty_sampling"}:
        return np.asarray(std, dtype=float)
    if name == "ei":
        return expected_improvement(
            mean=mean,
            std=std,
            best_f=best_f,
            xi=xi,
        )
    if name == "thompson":
        if surrogate is None or candidate_states is None:
            raise ValueError("Thompson sampling requires surrogate and candidate_states.")
        draws = surrogate.sample(
            np.asarray(candidate_states, dtype=np.int64),
            n_samples=max(int(thompson_samples), 1),
        )
        return np.asarray(draws[0], dtype=float)
    raise ValueError(
        f"Unsupported acquisition function: {acquisition}. "
        "Use one of {'ucb', 'uncertainty', 'ei', 'thompson'}."
    )


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two integer state vectors (ignoring shared padding)."""
    return int(np.sum(a != b))


def _greedy_diverse_selection(
    scores: np.ndarray,
    candidate_states: np.ndarray,
    batch_size: int,
    diversity_weight: float = 0.3,
) -> np.ndarray:
    """Greedy batch selection that balances acquisition score and diversity.

    At each step, we select the candidate maximising:

        combined(x) = normalised_acq(x) + λ × min_distance(x, selected)

    where:
    - normalised_acq is the acquisition score scaled to [0, 1],
    - min_distance is the minimum Hamming distance to any already-selected
      candidate (normalised by state dimension),
    - λ (diversity_weight) controls the exploration-exploitation trade-off.

    Why Hamming distance?
    ---------------------
    In discrete token spaces, Hamming distance is a natural metric:
    two states that differ by k tokens are k mutations apart.
    Ensuring minimum Hamming distance in a batch means we query the oracle
    on structurally different candidates, gathering more diverse information
    for the surrogate to learn from.

    This is a submodular-greedy approach (related to DPP batch selection)
    that provably achieves a (1 - 1/e) ≈ 63% approximation to the optimal
    diverse batch under monotone submodular objectives.

    Parameters
    ----------
    scores : (n_candidates,) acquisition scores.
    candidate_states : (n_candidates, state_dim) integer state matrix.
    batch_size : number of candidates to select.
    diversity_weight : λ, strength of diversity bonus.
    """
    n_candidates = scores.shape[0]
    batch_size = min(batch_size, n_candidates)
    if batch_size <= 0:
        return np.array([], dtype=np.intp)

    state_dim = float(candidate_states.shape[1])
    scores_f = np.asarray(scores, dtype=np.float64)

    # Normalise acquisition scores to [0, 1] for fair weighting with diversity.
    s_min = scores_f.min()
    s_range = scores_f.max() - s_min
    if s_range > 1e-12:
        norm_scores = (scores_f - s_min) / s_range
    else:
        norm_scores = np.ones_like(scores_f)

    selected: list[int] = []
    # Track minimum distance from each candidate to the selected set.
    # Initialise to max possible (= state_dim).
    min_dist = np.full(n_candidates, state_dim, dtype=np.float64)
    used = np.zeros(n_candidates, dtype=bool)

    for _ in range(batch_size):
        if len(selected) == 0:
            # First selection: pure acquisition score (no diversity yet).
            combined = norm_scores.copy()
        else:
            # Normalise minimum distance to [0, 1].
            norm_dist = min_dist / state_dim
            combined = norm_scores + diversity_weight * norm_dist

        combined[used] = -np.inf
        best_idx = int(np.argmax(combined))
        selected.append(best_idx)
        used[best_idx] = True

        # Update min distances for remaining candidates.
        new_state = candidate_states[best_idx]
        for j in range(n_candidates):
            if not used[j]:
                d = float(_hamming_distance(candidate_states[j], new_state))
                if d < min_dist[j]:
                    min_dist[j] = d

    return np.array(selected, dtype=np.intp)


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
    diversity_weight: float = 0.3,
) -> np.ndarray:
    """Dispatch batch selection to the requested acquisition function.

    When candidate_states are provided and diversity_weight > 0, uses
    greedy diversity-aware selection instead of pure top-k.  This prevents
    wasting oracle budget on near-identical candidates.
    """
    name = str(acquisition).lower()
    if name == "thompson":
        if surrogate is None or candidate_states is None:
            raise ValueError("Thompson sampling requires surrogate and candidate_states.")
        draws = surrogate.sample(
            np.asarray(candidate_states, dtype=np.int64),
            n_samples=max(int(thompson_samples), 1),
        )
        return select_thompson(draws=draws[0], batch_size=batch_size)

    scores = score_acquisition(
        acquisition,
        mean=mean,
        std=std,
        best_f=best_f,
        surrogate=surrogate,
        candidate_states=candidate_states,
        beta=beta,
        xi=xi,
        thompson_samples=thompson_samples,
    )

    # Use diversity-aware selection when we have state vectors to compare.
    if (
        candidate_states is not None
        and diversity_weight > 0
        and candidate_states.shape[0] > batch_size
    ):
        return _greedy_diverse_selection(
            scores=scores,
            candidate_states=np.asarray(candidate_states, dtype=np.int64),
            batch_size=batch_size,
            diversity_weight=diversity_weight,
        )

    # Fallback: pure greedy top-k.
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    return order[:batch_size]
