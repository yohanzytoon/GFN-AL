"""Dataset generation utilities for the preliminary milestone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, search_quality_metrics


def load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved dataset of Scrabble states and oracle scores."""
    payload = np.load(Path(dataset_path), allow_pickle=False)
    states = np.asarray(payload["states"], dtype=np.int64)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    return states, scores


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
    random_states = env.get_random_terminating_states(
        n_states=n_queries,
        unique=bool(dataset_cfg.get("unique", False)),
        max_attempts=max(5 * max(n_queries, 1), 1000),
    )
    proxy_states = env.states2proxy(random_states)
    scores = oracle(proxy_states).detach().cpu().numpy().astype(np.float32)
    states = np.asarray(random_states, dtype=np.int64)

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(dataset_path, states=states, scores=scores)

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=scores.tolist(),
        states=random_states,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    result = {
        "method": "dataset_generation",
        "seed": seed,
        "dataset_path": str(dataset_path),
        "num_samples": int(states.shape[0]),
        **quality,
        "curve": curve,
        "scores": scores.tolist(),
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_dataset.json")

    return result
