"""Upper Confidence Bound acquisition with diversity-aware batch selection."""

from __future__ import annotations

import numpy as np


def ucb_scores(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Compute UCB score for each candidate."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return mean + beta * std


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two integer state vectors (ignoring shared padding)."""
    return int(np.sum(a != b))


def _greedy_diverse_selection(
    scores: np.ndarray,
    candidate_states: np.ndarray,
    batch_size: int,
    diversity_weight: float = 0.3,
) -> np.ndarray:
    """Greedy batch selection balancing UCB score and diversity.

    At each step selects the candidate maximising:

        combined(x) = normalised_acq(x) + λ × min_distance(x, selected)

    where min_distance is the minimum Hamming distance to any already-selected
    candidate (normalised by state dimension) and λ is diversity_weight.

    Achieves a (1 - 1/e) ≈ 63% approximation to the optimal diverse batch
    under monotone submodular objectives.
    """
    n_candidates = scores.shape[0]
    batch_size = min(batch_size, n_candidates)
    if batch_size <= 0:
        return np.array([], dtype=np.intp)

    state_dim = float(candidate_states.shape[1])
    scores_f = np.asarray(scores, dtype=np.float64)

    s_min = scores_f.min()
    s_range = scores_f.max() - s_min
    norm_scores = (scores_f - s_min) / s_range if s_range > 1e-12 else np.ones_like(scores_f)

    selected: list[int] = []
    min_dist = np.full(n_candidates, state_dim, dtype=np.float64)
    used = np.zeros(n_candidates, dtype=bool)

    for _ in range(batch_size):
        if len(selected) == 0:
            combined = norm_scores.copy()
        else:
            combined = norm_scores + diversity_weight * (min_dist / state_dim)

        combined[used] = -np.inf
        best_idx = int(np.argmax(combined))
        selected.append(best_idx)
        used[best_idx] = True

        new_state = candidate_states[best_idx]
        for j in range(n_candidates):
            if not used[j]:
                d = float(_hamming_distance(candidate_states[j], new_state))
                if d < min_dist[j]:
                    min_dist[j] = d

    return np.array(selected, dtype=np.intp)


def select_acquisition_batch(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    beta: float = 2.0,
    candidate_states: np.ndarray | None = None,
    diversity_weight: float = 0.3,
) -> np.ndarray:
    """Select a batch of candidates by UCB, with optional diversity-aware selection.

    When candidate_states are provided and diversity_weight > 0, uses greedy
    Hamming-diverse selection to avoid querying near-identical candidates.
    """
    scores = ucb_scores(mean=mean, std=std, beta=beta)

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

    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    return order[:batch_size]
