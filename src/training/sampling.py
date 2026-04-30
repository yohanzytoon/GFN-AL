"""Sampling strategies for generating and augmenting Scrabble state pools."""

from __future__ import annotations

import numpy as np

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from utils.scrabble import SCRABBLE_LETTER_SCORES, vocabulary_bigram_model

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


def _resolve_sampling_strategy(sampling_strategy: str) -> str:
    strategy = str(sampling_strategy).lower()
    aliases = {
        "freq": "frequency",
        "score": "frequency_score",
        "score_bias": "frequency_score",
        "score_biased": "frequency_score",
        "bigram": "ngram",
        "ngram_score_bias": "ngram_score",
        "bigram_score": "ngram_score",
    }
    return aliases.get(strategy, strategy)


def _letter_sampling_probs(env: ScrabbleOracleEnv, sampling_strategy: str) -> np.ndarray:
    """Return per-letter marginal sampling probabilities (independent of position).

    Used for fallback generation and any position-independent letter choice.
    For sequential (bigram-chain) generation, see ``_sample_state_bigram_chain``.
    """
    strategy = _resolve_sampling_strategy(sampling_strategy)
    if strategy == "uniform":
        return np.ones(env.n_letters, dtype=np.float64) / float(env.n_letters)

    frequency_weights = np.asarray(
        [
            _ENGLISH_LETTER_FREQUENCIES.get(str(token).upper(), 1.0)
            for token in env.letters
        ],
        dtype=np.float64,
    )

    if strategy in {"frequency", "ngram"}:
        weights = frequency_weights
    elif strategy in {"frequency_score", "ngram_score"}:
        score_weights = SCRABBLE_LETTER_SCORES[1 : 1 + len(env.letters)].astype(np.float64)
        weights = np.power(frequency_weights, 0.7) * np.power(score_weights, 1.35)
    else:
        raise ValueError(
            f"Unsupported sampling strategy: {sampling_strategy}. "
            "Use 'uniform', 'frequency', 'frequency_score', 'ngram', or 'ngram_score'."
        )

    return weights / weights.sum()


def _length_sampling_probs(
    env: ScrabbleOracleEnv,
    *,
    min_length: int,
    sampling_strategy: str,
    bigram_model: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    strategy = _resolve_sampling_strategy(sampling_strategy)
    lengths = np.arange(min_length, env.max_length + 1, dtype=np.int64)

    if strategy in {"ngram", "ngram_score"} and bigram_model is not None:
        raw = bigram_model["length_probs"]
        weights = np.asarray(
            [float(raw[l]) if l < len(raw) else 0.0 for l in lengths],
            dtype=np.float64,
        )
        if strategy == "ngram_score":
            weights *= np.power(lengths.astype(np.float64), 1.5)
        if weights.sum() <= 0:
            weights = np.ones_like(lengths, dtype=np.float64)
    elif strategy == "frequency":
        center = min(max(5, min_length), env.max_length)
        weights = 1.0 / (1.0 + np.abs(lengths - center))
    elif strategy in {"frequency_score", "ngram_score"}:
        weights = np.power(lengths.astype(np.float64), 2.5)
    elif strategy in {"uniform", "ngram"}:
        weights = np.ones_like(lengths, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported sampling strategy: {sampling_strategy}. "
            "Use 'uniform', 'frequency', 'frequency_score', 'ngram', or 'ngram_score'."
        )

    return lengths, weights / weights.sum()


def deduplicate_state_scores(
    states: list[list[int]] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate states while preserving order (keeps highest score per duplicate)."""
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


def _sample_state_bigram_chain(
    rng: np.random.Generator,
    length: int,
    max_length: int,
    bigram_model: dict[str, np.ndarray],
    score_bias: float = 0.0,
) -> np.ndarray:
    """Generate one word by sampling a first-order Markov chain over letters.

    The bigram chain P(letter_i | letter_{i-1}) is learned from the Scrabble
    vocabulary. When ``score_bias > 0``, mixes in a preference for high-scoring
    Scrabble letters via a product-of-experts formulation.
    """
    bigram_probs = bigram_model["bigram_probs"]  # (26, 26)
    start_probs = bigram_model["start_probs"]    # (26,)
    tile_values = SCRABBLE_LETTER_SCORES[1:27].astype(np.float64)

    state = np.zeros(max_length, dtype=np.int64)

    if score_bias > 0:
        probs = start_probs * np.power(tile_values, score_bias)
        probs /= probs.sum()
    else:
        probs = start_probs
    first_letter = int(rng.choice(26, p=probs))
    state[0] = first_letter + 1

    for pos in range(1, length):
        prev = int(state[pos - 1]) - 1
        transition = bigram_probs[prev]
        if score_bias > 0:
            transition = transition * np.power(tile_values, score_bias)
            transition /= transition.sum()
        next_letter = int(rng.choice(26, p=transition))
        state[pos] = next_letter + 1

    return state


def sample_terminating_states(
    env: ScrabbleOracleEnv,
    n_states: int,
    *,
    sampling_strategy: str = "uniform",
    min_length: int = 3,
    unique: bool = True,
    seed: int | None = None,
    gflownet_root: str | None = None,
) -> list[list[int]]:
    """Sample terminating Scrabble states with a configurable strategy.

    Strategies
    ----------
    uniform          : Completely random letters and lengths.
    frequency        : Letters sampled by English frequency, lengths centered at 5.
    frequency_score  : Frequency-weighted with bias toward high-scoring tiles.
    ngram            : Bigram Markov chain from the vocabulary — produces word-like
                       sequences with ~8-15× higher valid-word rate than frequency.
    ngram_score      : Bigram chain biased toward high-value Scrabble tiles —
                       best balance of word-validity and high score potential.
    """
    if n_states <= 0:
        return []

    strategy = _resolve_sampling_strategy(sampling_strategy)
    if strategy == "uniform":
        return env.get_random_terminating_states(
            n_states=n_states,
            unique=unique,
            max_attempts=max(5 * max(n_states, 1), 1000),
        )

    rng = np.random.default_rng(seed)
    min_length = int(max(1, min(min_length, env.max_length)))

    bigram_model = None
    use_bigram_chain = strategy in {"ngram", "ngram_score"}
    if use_bigram_chain:
        bigram_model = vocabulary_bigram_model(
            max_length=int(env.max_length),
            gflownet_root=gflownet_root,
        )
        if bigram_model is None:
            strategy = "frequency" if strategy == "ngram" else "frequency_score"
            use_bigram_chain = False

    lengths, length_probs = _length_sampling_probs(
        env,
        min_length=min_length,
        sampling_strategy=strategy,
        bigram_model=bigram_model,
    )
    letter_probs = _letter_sampling_probs(env, strategy)
    letter_indices = np.arange(1, env.n_letters + 1, dtype=np.int64)

    # 0.25 gently favours high-value letters (K, W, V, Y) without collapsing
    # onto rare letters like Q, Z that rarely form valid words.
    score_bias = 0.25 if strategy == "ngram_score" else 0.0

    states: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(10 * n_states, 1000)

    for _ in range(max_attempts):
        if len(states) >= n_states:
            break
        length = int(rng.choice(lengths, p=length_probs))

        if use_bigram_chain and bigram_model is not None:
            state = _sample_state_bigram_chain(
                rng, length, env.max_length, bigram_model, score_bias=score_bias,
            )
        else:
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
