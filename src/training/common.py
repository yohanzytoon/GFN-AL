"""Shared helpers for experiment training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from acquisition.factory import score_acquisition
from surrogate.factory import build_surrogate
from utils.scrabble import ngram_plausibility, vocabulary_bigram_model


def build_surrogate_from_config(
    surrogate_cfg: dict[str, Any],
    *,
    max_length: int,
    num_tokens: int,
    device: str,
):
    """Instantiate a surrogate backend from config."""
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
        else:
            scores.extend(float(score) for score in record_scores)
    return states, scores


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


def propose_local_search_candidates(
    *,
    env,
    surrogate,
    acquisition_name: str,
    best_f: float,
    anchor_states: list[list[int]],
    proposal_size: int,
    beam_width: int,
    n_steps: int,
    neighbors_per_step: int,
    sampling_strategy: str,
    min_length: int,
    candidate_unique: bool,
    seen_states: Iterable[Iterable[int]],
    seed: int,
    beta: float = 2.0,
    xi: float = 0.0,
    thompson_samples: int = 1,
    mutation_edits: int = 1,
    max_length: int | None = None,
    gflownet_root: str | None = None,
    plausibility_weight: float = 2.0,
) -> tuple[list[list[int]], dict[str, int]]:
    """Optimize the acquisition over a discrete neighborhood by multi-start local search."""
    from training.dataset import sample_mutated_states

    if (
        proposal_size <= 0
        or beam_width <= 0
        or n_steps <= 0
        or neighbors_per_step <= 0
        or len(anchor_states) == 0
    ):
        return [], {"surrogate_queries": 0, "evaluated_candidates": 0}

    beam_width = max(int(beam_width), 1)
    n_steps = max(int(n_steps), 1)
    neighbors_per_step = max(int(neighbors_per_step), 1)
    proposal_size = max(int(proposal_size), beam_width)

    local_seen = {tuple(int(x) for x in state) for state in seen_states}
    scored_candidates: dict[tuple[int, ...], float] = {}
    surrogate_queries = 0
    evaluated_candidates = 0

    beam = [list(state) for state in anchor_states[:beam_width]]
    for step in range(n_steps):
        neighbor_states = sample_mutated_states(
            env,
            beam,
            n_states=beam_width * neighbors_per_step,
            sampling_strategy=sampling_strategy,
            min_length=min_length,
            unique=candidate_unique,
            seed=seed + step,
            max_mutations=mutation_edits,
        )
        filtered_neighbors = filter_new_states(neighbor_states, local_seen)
        if len(filtered_neighbors) == 0:
            break

        candidate_matrix = np.asarray(filtered_neighbors, dtype=np.int64)
        mean, std = surrogate.predict(candidate_matrix, return_std=True)
        if max_length is not None and plausibility_weight != 0.0:
            mean = mean + compute_plausibility_bonus(
                candidate_matrix,
                int(max_length),
                gflownet_root=gflownet_root,
                weight=float(plausibility_weight),
            )
        scores = score_acquisition(
            acquisition_name,
            mean=mean,
            std=std,
            best_f=best_f,
            surrogate=surrogate,
            candidate_states=candidate_matrix,
            beta=beta,
            xi=xi,
            thompson_samples=thompson_samples,
        )
        surrogate_queries += int(candidate_matrix.shape[0])
        evaluated_candidates += int(candidate_matrix.shape[0])

        order = np.argsort(np.asarray(scores, dtype=float))[::-1]
        beam = []
        for idx in order[:beam_width]:
            state = candidate_matrix[int(idx)].tolist()
            key = tuple(int(x) for x in state)
            local_seen.add(key)
            scored_candidates[key] = float(scores[int(idx)])
            beam.append(state)

        # Keep additional high-scoring neighbors as final proposals even if they
        # are not retained in the beam for the next search step.
        keep = min(max(beam_width * 2, 1), int(candidate_matrix.shape[0]))
        for idx in order[:keep]:
            key = tuple(int(x) for x in candidate_matrix[int(idx)].tolist())
            local_seen.add(key)
            scored_candidates[key] = max(
                float(scores[int(idx)]),
                scored_candidates.get(key, float("-inf")),
            )

    ranked = sorted(scored_candidates.items(), key=lambda item: item[1], reverse=True)
    proposals = [list(key) for key, _ in ranked[:proposal_size]]
    return proposals, {
        "surrogate_queries": int(surrogate_queries),
        "evaluated_candidates": int(evaluated_candidates),
    }
