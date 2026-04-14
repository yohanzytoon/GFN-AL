"""Dataset generation utilities for the preliminary milestone.

Includes multiple sampling strategies ranging from uniform random to
vocabulary-derived bigram models that dramatically increase the
probability of generating valid English words.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from utils.device import resolve_device
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, search_quality_metrics
from utils.scrabble import SCRABBLE_LETTER_SCORES, resolve_scrabble_optimum, vocabulary_bigram_model

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

    Used for mutations, fallback generation, and any position-independent
    letter choice.  For sequential (bigram-chain) generation, see
    ``_sample_state_bigram_chain`` which uses position-dependent transitions.
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
        # For ngram strategy, the per-letter marginals come from English
        # frequencies.  The sequential structure is handled by the bigram chain.
        weights = frequency_weights
    elif strategy in {"frequency_score", "ngram_score"}:
        # SCRABBLE_LETTER_SCORES indices 1-26 = A-Z, matching env.letters order.
        score_weights = SCRABBLE_LETTER_SCORES[1 : 1 + len(env.letters)].astype(np.float64)
        # Mix word-likeness with a controlled preference for high-value letters.
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
        # Use the vocabulary-derived length distribution.
        # This ensures we generate words of the lengths that actually appear
        # in the dictionary, biased toward longer words for higher scores.
        raw = bigram_model["length_probs"]
        weights = np.asarray(
            [float(raw[l]) if l < len(raw) else 0.0 for l in lengths],
            dtype=np.float64,
        )
        if strategy == "ngram_score":
            # Bias toward longer words: longer words = more tiles = higher score.
            weights *= np.power(lengths.astype(np.float64), 1.5)
        if weights.sum() <= 0:
            weights = np.ones_like(lengths, dtype=np.float64)
    elif strategy == "frequency":
        center = min(max(5, min_length), env.max_length)
        weights = 1.0 / (1.0 + np.abs(lengths - center))
    elif strategy in {"frequency_score", "ngram_score"}:
        # Fallback when bigram model is unavailable
        weights = np.power(lengths.astype(np.float64), 2.5)
    elif strategy in {"uniform", "ngram"}:
        weights = np.ones_like(lengths, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported sampling strategy: {sampling_strategy}. "
            "Use 'uniform', 'frequency', 'frequency_score', 'ngram', or 'ngram_score'."
        )

    return lengths, weights / weights.sum()


def _state_length(state: list[int] | np.ndarray) -> int:
    array = np.asarray(state, dtype=np.int64).reshape(-1)
    zero_idx = np.where(array == 0)[0]
    if zero_idx.size == 0:
        return int(array.shape[0])
    return int(zero_idx[0])


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


def _sample_state_bigram_chain(
    rng: np.random.Generator,
    length: int,
    max_length: int,
    bigram_model: dict[str, np.ndarray],
    score_bias: float = 0.0,
) -> np.ndarray:
    """Generate one word by sampling a first-order Markov chain over letters.

    The bigram chain P(letter_i | letter_{i-1}) is learned from the Scrabble
    vocabulary.  This produces sequences that *look like* English words,
    dramatically increasing the probability of generating a valid word
    compared to independent letter sampling.

    When ``score_bias > 0``, we mix in a preference for high-scoring Scrabble
    letters.  The combined distribution at each step is:

        P_mix(j | i) ∝ P_bigram(j | i) × score(j)^score_bias

    This is a product-of-experts formulation: the bigram model provides
    "word-likeness" and the score weights provide "value-seeking".

    Parameters
    ----------
    rng : Generator
        Numpy random generator for reproducibility.
    length : int
        Desired word length.
    max_length : int
        State vector length (with zero-padding).
    bigram_model : dict
        Output of ``vocabulary_bigram_model()``.
    score_bias : float
        Exponent controlling Scrabble-score bias.  0 = pure bigram model,
        higher values = prefer high-value tiles.
    """
    bigram_probs = bigram_model["bigram_probs"]  # (26, 26)
    start_probs = bigram_model["start_probs"]  # (26,)
    # Scrabble tile values for A-Z (0-indexed), used for score biasing.
    tile_values = SCRABBLE_LETTER_SCORES[1:27].astype(np.float64)

    state = np.zeros(max_length, dtype=np.int64)

    # Sample first letter from the vocabulary start distribution.
    if score_bias > 0:
        probs = start_probs * np.power(tile_values, score_bias)
        probs /= probs.sum()
    else:
        probs = start_probs
    first_letter = int(rng.choice(26, p=probs))  # 0-indexed
    state[0] = first_letter + 1  # Convert to 1-indexed token

    # Sample subsequent letters using bigram transitions.
    for pos in range(1, length):
        prev = int(state[pos - 1]) - 1  # Back to 0-indexed
        transition = bigram_probs[prev]  # P(next | prev)
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

    # Try to load the vocabulary bigram model for ngram strategies.
    bigram_model = None
    use_bigram_chain = strategy in {"ngram", "ngram_score"}
    if use_bigram_chain:
        bigram_model = vocabulary_bigram_model(
            max_length=int(env.max_length),
            gflownet_root=gflownet_root,
        )
        if bigram_model is None:
            # Graceful fallback: ngram→frequency, ngram_score→frequency_score
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

    # Score bias for ngram_score: mild bias toward high-value tiles.
    # A value of 0.25 gently favours letters like K, W, V, Y without
    # collapsing onto rare letters like Q, Z that rarely form valid words.
    score_bias = 0.25 if strategy == "ngram_score" else 0.0

    states: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(10 * n_states, 1000)

    for _ in range(max_attempts):
        if len(states) >= n_states:
            break
        length = int(rng.choice(lengths, p=length_probs))

        if use_bigram_chain and bigram_model is not None:
            # Generate using the vocabulary-derived Markov chain.
            state = _sample_state_bigram_chain(
                rng, length, env.max_length, bigram_model, score_bias=score_bias,
            )
        else:
            # Independent letter sampling (original behaviour).
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


def sample_mutated_states(
    env: ScrabbleOracleEnv,
    base_states: list[list[int]] | np.ndarray,
    n_states: int,
    *,
    sampling_strategy: str = "frequency_score",
    min_length: int = 3,
    unique: bool = True,
    seed: int | None = None,
    max_mutations: int = 2,
    gflownet_root: str | None = None,
) -> list[list[int]]:
    """Sample new states by locally mutating promising existing states.

    Operations
    ----------
    replace   (55%) : Swap one letter for another (bigram-aware if model available).
    insert    (15%) : Insert a letter at a random position.
    delete    (10%) : Remove a letter.
    crossover (20%) : Combine prefix of one anchor with suffix of another.
                      This is a standard genetic-algorithm operator that
                      recombines building blocks of good solutions.

    When a bigram model is available, replacement letters are sampled from
    the conditional distribution P(letter | neighbor context) rather than
    the marginal distribution. This keeps mutations "word-like".
    """
    if n_states <= 0:
        return []

    anchors = np.asarray(base_states, dtype=np.int64)
    if anchors.size == 0 or anchors.ndim != 2:
        return []

    rng = np.random.default_rng(seed)
    min_length = int(max(1, min(min_length, env.max_length)))
    max_mutations = max(int(max_mutations), 1)
    letter_probs = _letter_sampling_probs(env, sampling_strategy)
    letter_indices = np.arange(1, env.n_letters + 1, dtype=np.int64)

    # Load bigram model for context-aware mutations if available.
    strategy = _resolve_sampling_strategy(sampling_strategy)
    bigram_model = None
    if strategy in {"ngram", "ngram_score", "frequency_score"}:
        bigram_model = vocabulary_bigram_model(
            max_length=int(env.max_length),
            gflownet_root=gflownet_root,
        )

    states: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(25 * n_states, 1000)

    for _ in range(max_attempts):
        if len(states) >= n_states:
            break

        # Decide: point mutation or crossover?
        # Crossover combines successful building blocks from two parents,
        # a powerful exploration operator from evolutionary algorithms.
        use_crossover = anchors.shape[0] >= 2 and rng.random() < 0.20

        if use_crossover:
            # --- Crossover: prefix of parent A + suffix of parent B ---
            idx_a = int(rng.integers(0, anchors.shape[0]))
            idx_b = int(rng.integers(0, anchors.shape[0]))
            if idx_b == idx_a:
                idx_b = (idx_a + 1) % anchors.shape[0]

            state_a = anchors[idx_a]
            state_b = anchors[idx_b]
            len_a = _state_length(state_a)
            len_b = _state_length(state_b)
            if len_a < 2 or len_b < 2:
                continue

            # Choose a crossover point: take first k letters from A,
            # remaining letters from B.
            cut = int(rng.integers(1, min(len_a, len_b)))
            new_length = min(cut + (len_b - cut), env.max_length)
            state = np.zeros(env.max_length, dtype=np.int64)
            state[:cut] = state_a[:cut]
            remaining = min(len_b - cut, env.max_length - cut)
            state[cut : cut + remaining] = state_b[cut : cut + remaining]
            length = cut + remaining
        else:
            # --- Point mutation ---
            state = anchors[int(rng.integers(0, anchors.shape[0]))].copy()
            length = _state_length(state)
            if length <= 0:
                continue

            n_edits = int(rng.integers(1, max_mutations + 1))
            for _ in range(n_edits):
                operations = ["replace"]
                op_weights = [0.55]
                if length < env.max_length:
                    operations.append("insert")
                    op_weights.append(0.15)
                if length > min_length:
                    operations.append("delete")
                    op_weights.append(0.10)
                op_probs = np.asarray(op_weights, dtype=np.float64)
                op_probs = op_probs / op_probs.sum()
                operation = str(rng.choice(np.asarray(operations, dtype=object), p=op_probs))

                if operation == "replace":
                    position = int(rng.integers(0, length))

                    # Context-aware replacement: if we have a bigram model,
                    # sample the replacement letter conditioned on the
                    # previous letter.  This keeps the mutated word
                    # structurally similar to real English words.
                    if bigram_model is not None and position > 0:
                        prev_letter = int(state[position - 1]) - 1  # 0-indexed
                        if 0 <= prev_letter < 26:
                            ctx_probs = bigram_model["bigram_probs"][prev_letter].copy()
                            # Mild score bias for ngram_score strategy
                            if strategy == "ngram_score":
                                tile_vals = SCRABBLE_LETTER_SCORES[1:27].astype(np.float64)
                                ctx_probs *= np.power(tile_vals, 0.4)
                            ctx_probs /= ctx_probs.sum()
                            state[position] = int(rng.choice(26, p=ctx_probs)) + 1
                        else:
                            state[position] = int(rng.choice(letter_indices, p=letter_probs))
                    else:
                        state[position] = int(rng.choice(letter_indices, p=letter_probs))

                elif operation == "insert":
                    position = int(rng.integers(0, length + 1))
                    state[position + 1 : length + 1] = state[position:length]
                    state[position] = int(rng.choice(letter_indices, p=letter_probs))
                    length += 1
                else:
                    position = int(rng.integers(0, length))
                    state[position : length - 1] = state[position + 1 : length]
                    state[length - 1] = 0
                    length -= 1

        if length < min_length:
            continue

        state[length:] = 0
        key = tuple(int(x) for x in state.tolist())
        if unique and key in seen:
            continue
        seen.add(key)
        states.append(list(key))

    return states[:n_states]


def generate_random_dataset(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Sample a random oracle-labeled dataset and save it to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    seed = int(config["seed"])
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    dataset_cfg = config["dataset"]
    device = resolve_device(config.get("device", "cpu"))
    optimum_info = resolve_scrabble_optimum(
        max_length=int(env_cfg["max_length"]),
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        configured_optimum_score=config.get("metrics", {}).get("optimum_score"),
        gflownet_root=config.get("gflownet_root"),
    )
    optimum_score = float(optimum_info["optimum_score"])

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
        enforce_budget=bool(oracle_cfg.get("enforce_budget", True)),
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
        optimum_score=optimum_score,
    )
    quality = search_quality_metrics(
        scores=scores.tolist(),
        states=states.tolist(),
        oracle_queries=int(states.shape[0]),
        optimum_score=optimum_score,
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
        "real_oracle_queries": int(states.shape[0]),
        "fake_oracle_queries": 0,
        "cheap_model_queries": 0,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "optimum_score": optimum_score,
    }
    if optimum_info.get("optimum_words"):
        result["optimum_words"] = list(optimum_info["optimum_words"])
        result["optimum_word_count"] = int(optimum_info["optimum_word_count"])
        result["optimum_source"] = str(optimum_info["optimum_source"])

    if logger is not None:
        logger.dump_summary(result, filename="summary_dataset.json")

    return result
