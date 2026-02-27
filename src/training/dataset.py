"""Dataset generation utilities for the preliminary milestone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, search_quality_metrics

_ENGLISH_LETTER_FREQUENCIES = {
    "A": 8.17,
    "B": 1.49,
    "C": 2.78,
    "D": 4.25,
    "E": 12.70,
    "F": 2.23,
    "G": 2.02,
    "H": 6.09,
    "I": 6.97,
    "J": 0.15,
    "K": 0.77,
    "L": 4.03,
    "M": 2.41,
    "N": 6.75,
    "O": 7.51,
    "P": 1.93,
    "Q": 0.10,
    "R": 5.99,
    "S": 6.33,
    "T": 9.06,
    "U": 2.76,
    "V": 0.98,
    "W": 2.36,
    "X": 0.15,
    "Y": 1.97,
    "Z": 0.07,
}


def load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved dataset of Scrabble states and oracle scores."""
    payload = np.load(Path(dataset_path), allow_pickle=False)
    states = np.asarray(payload["states"], dtype=np.int64)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    return states, scores


def deduplicate_state_scores(
    states: list[list[int]] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate states while preserving order."""
    states_np = np.asarray(states, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)

    if states_np.shape[0] != scores_np.shape[0]:
        raise ValueError(
            "states and scores must have the same number of rows for deduplication"
        )

    dedup_scores: dict[tuple[int, ...], float] = {}
    order: list[tuple[int, ...]] = []
    for state, score in zip(states_np, scores_np):
        key = tuple(int(x) for x in state.tolist())
        if key not in dedup_scores:
            order.append(key)
            dedup_scores[key] = float(score)
        else:
            dedup_scores[key] = max(dedup_scores[key], float(score))

    unique_states = np.asarray(order, dtype=np.int64)
    unique_scores = np.asarray([dedup_scores[key] for key in order], dtype=np.float32)
    return unique_states, unique_scores


def sample_terminating_states(
    env: ScrabbleOracleEnv,
    n_states: int,
    *,
    sampling_strategy: str = "uniform",
    min_length: int = 3,
    unique: bool = True,
    seed: int | None = None,
) -> list[list[int]]:
    """Sample terminating Scrabble states with a simple configurable strategy."""
    if n_states <= 0:
        return []

    strategy = str(sampling_strategy).lower()
    if strategy == "uniform":
        return env.get_random_terminating_states(
            n_states=n_states,
            unique=unique,
            max_attempts=max(5 * max(n_states, 1), 1000),
        )
    if strategy != "frequency":
        raise ValueError(
            f"Unsupported sampling strategy: {sampling_strategy}. Use 'frequency' or 'uniform'."
        )

    rng = np.random.default_rng(seed)
    min_length = int(max(1, min(min_length, env.max_length)))
    lengths = np.arange(min_length, env.max_length + 1, dtype=np.int64)
    center = min(max(5, min_length), env.max_length)
    length_weights = 1.0 / (1.0 + np.abs(lengths - center))
    length_probs = length_weights / length_weights.sum()

    letter_weights = np.asarray(
        [
            _ENGLISH_LETTER_FREQUENCIES.get(str(token).upper(), 1.0)
            for token in env.letters
        ],
        dtype=np.float64,
    )
    letter_probs = letter_weights / letter_weights.sum()
    letter_indices = np.arange(1, env.n_letters + 1, dtype=np.int64)

    states: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(10 * n_states, 1000)

    for _ in range(max_attempts):
        if len(states) >= n_states:
            break
        length = int(rng.choice(lengths, p=length_probs))
        state = np.zeros(env.max_length, dtype=np.int64)
        state[:length] = rng.choice(letter_indices, size=length, p=letter_probs)
        key = tuple(int(x) for x in state.tolist())
        if unique and key in seen:
            continue
        seen.add(key)
        states.append(list(key))

    if len(states) < n_states:
        fallback = env.get_random_terminating_states(
            n_states=n_states - len(states),
            unique=False,
            max_attempts=max(5 * max(n_states - len(states), 1), 1000),
        )
        for state in fallback:
            key = tuple(int(x) for x in state)
            if unique and key in seen:
                continue
            seen.add(key)
            states.append(list(key))
            if len(states) >= n_states:
                break

    return states[:n_states]


def generate_random_dataset(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Sample a random oracle-labeled dataset and save it to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    dataset_cfg = config["dataset"]
    device = config.get("device", "cpu")

    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=int(oracle_cfg["budget"]),
        track_oracle_history=True,
        device=device,
    )
    oracle = OracleProxy(
        device=device,
        float_precision=32,
        oracle_budget=int(oracle_cfg["budget"]),
        enforce_budget=True,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
    )
    oracle.setup(env)

    n_queries = int(min(oracle_cfg["budget"], dataset_cfg.get("num_queries", oracle_cfg["budget"])))
    sampled_states = sample_terminating_states(
        env,
        n_queries,
        sampling_strategy=str(dataset_cfg.get("sampling_strategy", "uniform")),
        min_length=int(dataset_cfg.get("min_length", 3)),
        unique=bool(dataset_cfg.get("unique", True)),
        seed=seed,
    )
    proxy_states = env.states2proxy(sampled_states)
    scores = oracle(proxy_states).detach().cpu().numpy().astype(np.float32)
    states, scores = deduplicate_state_scores(sampled_states, scores)

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(dataset_path, states=states, scores=scores)

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=scores.tolist(),
        states=states.tolist(),
        oracle_queries=int(states.shape[0]),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    result = {
        "method": "dataset_generation",
        "seed": seed,
        "dataset_path": str(dataset_path),
        "num_samples": int(states.shape[0]),
        "sampling_strategy": str(dataset_cfg.get("sampling_strategy", "uniform")),
        **quality,
        "curve": curve,
        "scores": scores.tolist(),
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_dataset.json")

    return result
