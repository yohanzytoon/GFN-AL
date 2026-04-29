"""Shared helpers for experiment training pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from surrogate.factory import build_surrogate
from utils.scrabble import ngram_plausibility, vocabulary_bigram_model


def build_surrogate_from_config(
    surrogate_cfg: dict[str, Any],
    *,
    max_length: int,
    num_tokens: int,
    device: str,
    gflownet_root: str | None = None,
):
    """Instantiate a surrogate backend from config."""
    surrogate_cfg = dict(surrogate_cfg)
    if gflownet_root is not None:
        surrogate_cfg.setdefault("gflownet_root", gflownet_root)
    return build_surrogate(
        surrogate_cfg,
        max_length=max_length,
        num_tokens=num_tokens,
        device=device,
    )


def filter_new_states(
    candidate_states: Iterable[Iterable[int]],
    seen_states: Iterable[Iterable[int]],
) -> list[list[int]]:
    """Remove candidate states already present in the seen set while preserving order."""
    seen = {tuple(int(x) for x in state) for state in seen_states}
    filtered: list[list[int]] = []
    added: set[tuple[int, ...]] = set()
    for state in candidate_states:
        key = tuple(int(x) for x in state)
        if key in seen or key in added:
            continue
        filtered.append(list(key))
        added.add(key)
    return filtered


def flatten_oracle_history(
    call_history: list[dict[str, Any]],
) -> tuple[list[list[int] | str], list[float]]:
    """Flatten oracle call history into per-query states and scores."""
    states: list[list[int] | str] = []
    scores: list[float] = []
    for record in call_history:
        record_states = record.get("states", [])
        record_scores = record.get("scores", [])
        if record_states and len(record_states) == len(record_scores):
            states.extend(record_states)
            scores.extend(float(score) for score in record_scores)
    return states, scores


def query_oracle_scores(oracle, states: Iterable[Iterable[int]]) -> list[float]:
    """Query a batch of integer states and return Python float scores."""
    oracle_states = (
        states
        if isinstance(states, np.ndarray) or hasattr(states, "detach")
        else np.asarray(list(states), dtype=np.int64)
    )
    return (
        oracle(oracle_states)
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )


def compute_plausibility_bonus(
    candidate_matrix: np.ndarray,
    max_length: int,
    gflownet_root: str | None = None,
    weight: float = 2.0,
) -> np.ndarray:
    """Compute a word-plausibility bonus for candidate states.

    The GP surrogate struggles to learn from sparse rewards (~93% of queries
    return score=0 in the Scrabble domain).  Without guidance, UCB acquisition
    picks the most *uncertain* candidates, which are typically far from any
    training data and almost never valid words.  This creates a vicious cycle:
    bad candidates → zero scores → surrogate stays bad.

    The plausibility prior breaks this cycle by adding a bonus proportional to
    bigram log-probability (how "English word-like" the candidate is).  This
    biases acquisition toward candidates that at least resemble real words,
    dramatically increasing the fraction of valid words found per round.

    The bonus is additive to the surrogate mean, acting as a Bayesian prior:

        adjusted_mean(x) = surrogate_mean(x) + weight × plausibility(x)

    As the surrogate accumulates real positive examples, its predictions
    dominate and the fixed plausibility bonus becomes relatively less
    influential — a natural form of prior annealing.
    """
    bigram_model = vocabulary_bigram_model(
        max_length=max_length, gflownet_root=gflownet_root,
    )
    if bigram_model is None:
        return np.zeros(candidate_matrix.shape[0], dtype=np.float64)

    plaus = ngram_plausibility(candidate_matrix, bigram_model)
    # Normalise to [0, 1]: typical raw values are [-8, -1.5].
    # Real English words usually score > -3, garbage scores < -5.
    plaus_norm = (plaus + 6.0).clip(0.0, 5.0) / 5.0
    return plaus_norm * weight
