"""Metrics utilities for active-learning experiments."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def running_best(scores: Sequence[float]) -> np.ndarray:
    """Running maximum over a score sequence."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    return np.maximum.accumulate(values)


def running_topk_mean(scores: Sequence[float], k: int = 10) -> np.ndarray:
    """Running mean of top-k scores observed so far."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    out = np.zeros_like(values)
    for idx in range(values.size):
        prefix = np.sort(values[: idx + 1])
        top_k = prefix[-min(k, prefix.size) :]
        out[idx] = float(np.mean(top_k))
    return out


def running_regret(scores: Sequence[float], optimum_score: float) -> np.ndarray:
    """Running simple regret against a known optimum score."""
    best = running_best(scores)
    if best.size == 0:
        return best
    return np.maximum(float(optimum_score) - best, 0.0)


def canonicalize_state(state: Sequence[int] | str, pad_value: int | str = 0) -> tuple:
    """Convert a state to a canonical tuple without padding."""
    if isinstance(state, str):
        raw = state.replace(" ", "")
        return tuple(raw)
    seq = list(state)
    if pad_value in seq:
        seq = seq[: seq.index(pad_value)]
    return tuple(seq)


def unique_fraction(states: Sequence[Sequence[int] | str]) -> float:
    """Fraction of unique states in a state collection."""
    if len(states) == 0:
        return 0.0
    canonical = [canonicalize_state(state) for state in states]
    return float(len(set(canonical)) / len(canonical))


def diversity_entropy(states: Sequence[Sequence[int] | str], pad_value: int | str = 0) -> float:
    """Token-level Shannon entropy over sampled states."""
    tokens: list[str] = []
    for state in states:
        if isinstance(state, str):
            for char in state.replace(" ", ""):
                tokens.append(char)
            continue
        for token in state:
            if token == pad_value:
                continue
            tokens.append(str(token))
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    probs = np.asarray(list(counts.values()), dtype=float)
    probs /= probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def valid_word_ratio(scores: Sequence[float], threshold: float = 0.0) -> float:
    """Fraction of sampled candidates receiving positive oracle score."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > float(threshold)))


def mode_coverage(
    states: Sequence[Sequence[int] | str],
    scores: Sequence[float],
    pad_value: int | str = 0,
    quantile: float = 0.9,
) -> int:
    """Number of unique states in the high-score tail (proxy for mode coverage)."""
    if len(states) == 0:
        return 0
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0
    threshold = float(np.quantile(values, quantile))
    selected = [
        canonicalize_state(state, pad_value=pad_value)
        for state, score in zip(states, values)
        if score >= threshold
    ]
    return len(set(selected))


def _levenshtein_distance(seq_a: Sequence, seq_b: Sequence) -> int:
    """Token-level Levenshtein distance."""
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)
    dp = np.zeros((len(seq_a) + 1, len(seq_b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(seq_a) + 1)
    dp[0, :] = np.arange(len(seq_b) + 1)
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(seq_a), len(seq_b)])


def mean_pairwise_edit_distance(
    states: Sequence[Sequence[int] | str],
    pad_value: int | str = 0,
) -> float:
    """Mean pairwise edit distance across a list of states."""
    if len(states) < 2:
        return 0.0
    canonical = [canonicalize_state(state, pad_value=pad_value) for state in states]
    distances: list[int] = []
    for i in range(len(canonical)):
        for j in range(i + 1, len(canonical)):
            distances.append(_levenshtein_distance(canonical[i], canonical[j]))
    if not distances:
        return 0.0
    return float(np.mean(distances))


def topk_edit_distance(
    states: Sequence[Sequence[int] | str],
    scores: Sequence[float],
    k: int = 10,
    pad_value: int | str = 0,
) -> float:
    """Mean pairwise edit distance for top-k scored states."""
    if len(states) == 0:
        return 0.0
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.0
    k_eff = int(min(max(k, 1), values.size))
    top_indices = np.argsort(values)[::-1][:k_eff]
    top_states = [states[int(i)] for i in top_indices]
    return mean_pairwise_edit_distance(top_states, pad_value=pad_value)


def queries_to_reach_target(scores: Sequence[float], target_score: float) -> int:
    """1-indexed number of oracle queries required to reach a target score."""
    best = running_best(scores)
    if best.size == 0:
        return 0
    reached = np.where(best >= float(target_score))[0]
    if reached.size == 0:
        return int(best.size)
    return int(reached[0] + 1)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Standard regression metrics for surrogate quality."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if y_t.size == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
    mse = np.mean(np.square(y_t - y_p))
    mae = np.mean(np.abs(y_t - y_p))
    var = np.var(y_t)
    r2 = 1.0 - mse / (var + 1e-12)
    return {"rmse": float(np.sqrt(mse)), "mae": float(mae), "r2": float(r2)}


def build_query_curve(
    scores: Sequence[float],
    optimum_score: float | None = None,
) -> dict[str, list[float]]:
    """Convenience helper returning sample-efficiency curves."""
    score_values = np.asarray(scores, dtype=float)
    queries = np.arange(1, score_values.size + 1)
    curve = {
        "queries": queries.tolist(),
        "best": running_best(score_values).tolist(),
        "top10": running_topk_mean(score_values, k=10).tolist(),
    }
    if optimum_score is not None:
        curve["regret"] = running_regret(score_values, optimum_score).tolist()
    return curve


def search_quality_metrics(
    scores: Sequence[float],
    states: Sequence[Sequence[int] | str] | None = None,
    oracle_queries: int | None = None,
    optimum_score: float | None = None,
    top_k: int = 10,
    pad_value: int | str = 0,
) -> dict[str, float | int]:
    """Compute summary metrics used in result tables."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return {
            "best_score": 0.0,
            "top10_score": 0.0,
            "mean_score": 0.0,
            "valid_word_ratio": 0.0,
            "mode_coverage": 0,
            "topk_edit_distance": 0.0,
            "queries_to_90pct_best": 0,
            "queries_to_90pct_optimum": 0,
            "simple_regret": 0.0,
            "oracle_queries": int(oracle_queries or 0),
            "diversity_entropy": 0.0,
            "unique_fraction": 0.0,
        }

    best_score = float(np.max(values))
    top10_score = float(np.mean(np.sort(values)[-min(10, values.size) :]))
    mean_score = float(np.mean(values))
    valid_ratio = float(valid_word_ratio(values))

    metrics: dict[str, float | int] = {
        "best_score": best_score,
        "top10_score": top10_score,
        "mean_score": mean_score,
        "valid_word_ratio": valid_ratio,
        "queries_to_90pct_best": int(queries_to_reach_target(values, 0.9 * best_score)),
        "oracle_queries": int(oracle_queries if oracle_queries is not None else values.size),
    }

    if optimum_score is not None:
        metrics["simple_regret"] = float(max(optimum_score - best_score, 0.0))
        metrics["queries_to_90pct_optimum"] = int(
            queries_to_reach_target(values, 0.9 * float(optimum_score))
        )
    else:
        metrics["simple_regret"] = 0.0
        metrics["queries_to_90pct_optimum"] = 0

    if states is not None and len(states) > 0:
        metrics["diversity_entropy"] = float(diversity_entropy(states, pad_value=pad_value))
        metrics["unique_fraction"] = float(unique_fraction(states))
        if len(states) == len(values):
            metrics["mode_coverage"] = int(
                mode_coverage(states, values, pad_value=pad_value, quantile=0.9)
            )
            metrics["topk_edit_distance"] = float(
                topk_edit_distance(states, values, k=top_k, pad_value=pad_value)
            )
        else:
            metrics["mode_coverage"] = 0
            metrics["topk_edit_distance"] = 0.0
    else:
        metrics["diversity_entropy"] = 0.0
        metrics["unique_fraction"] = 0.0
        metrics["mode_coverage"] = 0
        metrics["topk_edit_distance"] = 0.0

    return metrics
