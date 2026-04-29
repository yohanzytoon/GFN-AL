"""Pure active-learning training loop on Scrabble with budgeted oracle access."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.ucb import select_acquisition_batch
from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from training.common import (
    build_surrogate_from_config,
    filter_new_states,
    query_oracle_scores,
)
from training.dataset import (
    deduplicate_state_scores,
    sample_terminating_states,
)
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import (
    build_query_curve,
    regression_metrics,
    running_best,
    search_quality_metrics,
)
from utils.device import resolve_device
from utils.scrabble import resolve_scrabble_optimum


def run_active_learning(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run pure active learning and return experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    seed = int(config["seed"])
    device = resolve_device(config.get("device", "cpu"))
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    active_cfg = config["active"]

    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=int(oracle_cfg["budget"]),
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

    initial_size = int(min(active_cfg.get("initial_size", 50), oracle_cfg["budget"]))
    batch_size = int(active_cfg.get("batch_size", 16))
    candidate_pool_size = int(active_cfg.get("candidate_pool_size", 256))
    max_rounds = int(active_cfg.get("max_rounds", 200))
    acquisition_cfg = dict(active_cfg.get("acquisition", {}))
    acquisition_name = str(acquisition_cfg.get("name", "ucb")).lower()
    if acquisition_name != "ucb":
        raise ValueError(f"Unsupported acquisition: {acquisition_name}. Use 'ucb'.")
    beta = float(acquisition_cfg.get("beta", 2.0))
    beta_min = float(acquisition_cfg.get("beta_min", beta))
    surrogate_cfg = dict(active_cfg.get("surrogate", {}))

    sampling_strategy = str(active_cfg.get("sampling_strategy", "frequency"))
    initial_sampling_strategy = str(
        active_cfg.get("initial_sampling_strategy", sampling_strategy)
    )
    candidate_sampling_strategy = str(
        active_cfg.get("candidate_sampling_strategy", sampling_strategy)
    )
    fallback_sampling_strategy = str(
        active_cfg.get("fallback_sampling_strategy", "frequency")
    )
    min_length = int(active_cfg.get("min_length", 3))
    candidate_unique = bool(active_cfg.get("candidate_unique", True))
    diversity_weight = float(active_cfg.get("diversity_weight", 0.3))
    gflownet_root = config.get("gflownet_root")
    num_tokens = int(env_cfg.get("num_tokens", 27))
    optimum_info = resolve_scrabble_optimum(
        max_length=int(env_cfg["max_length"]),
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        configured_optimum_score=config.get("metrics", {}).get("optimum_score"),
        gflownet_root=config.get("gflownet_root"),
    )
    optimum_score = float(optimum_info["optimum_score"])

    states = sample_terminating_states(
        env,
        initial_size,
        sampling_strategy=initial_sampling_strategy,
        min_length=min_length,
        unique=True,
        seed=seed,
        gflownet_root=gflownet_root,
    )
    scores = query_oracle_scores(oracle, env.states2proxy(states))

    round_logs: list[dict[str, Any]] = []
    surrogate = None
    candidate_pool_queries = 0

    for round_idx in range(max_rounds):
        if oracle.remaining_budget <= 0:
            break

        # Linearly decay beta from beta (explore) to beta_min (exploit) over rounds.
        round_beta = beta - (beta - beta_min) * round_idx / max(max_rounds - 1, 1)

        train_states, train_scores = deduplicate_state_scores(states, scores)
        surrogate = build_surrogate_from_config(
            surrogate_cfg,
            max_length=int(env_cfg["max_length"]),
            num_tokens=num_tokens,
            device=device,
            gflownet_root=gflownet_root,
        )
        surrogate.fit(train_states, train_scores)

        candidate_states = sample_terminating_states(
            env,
            candidate_pool_size,
            sampling_strategy=candidate_sampling_strategy,
            min_length=min_length,
            unique=candidate_unique,
            seed=seed + round_idx + 1,
            gflownet_root=gflownet_root,
        )
        filtered_candidates = filter_new_states(candidate_states, states)
        if len(filtered_candidates) < batch_size:
            filtered_candidates.extend(
                filter_new_states(
                    sample_terminating_states(
                        env,
                        candidate_pool_size,
                        sampling_strategy=fallback_sampling_strategy,
                        min_length=min_length,
                        unique=False,
                        seed=seed + 10_000 + round_idx,
                        gflownet_root=gflownet_root,
                    ),
                    states,
                )
            )
            filtered_candidates = filter_new_states(filtered_candidates, states)

        candidate_matrix = np.asarray(filtered_candidates, dtype=np.int64)
        if candidate_matrix.size == 0:
            break
        candidate_pool_queries += int(candidate_matrix.shape[0])
        mean, std = surrogate.predict(candidate_matrix, return_std=True)

        this_batch = int(min(batch_size, oracle.remaining_budget, candidate_matrix.shape[0]))
        if this_batch <= 0:
            break

        selected_idx = select_acquisition_batch(
            mean=mean,
            std=std,
            batch_size=this_batch,
            candidate_states=candidate_matrix,
            beta=round_beta,
            diversity_weight=diversity_weight,
        )
        selected_states = candidate_matrix[selected_idx].tolist()
        queried_scores = query_oracle_scores(oracle, selected_states)

        states.extend(selected_states)
        scores.extend(float(s) for s in queried_scores)

        train_pred = surrogate.predict(np.asarray(states, dtype=np.int64), return_std=False)
        train_metrics = regression_metrics(np.asarray(scores, dtype=float), np.asarray(train_pred, dtype=float))
        round_quality = search_quality_metrics(
            scores=scores,
            states=states,
            oracle_queries=int(oracle.call_count),
            optimum_score=optimum_score,
            top_k=10,
            pad_value=0,
        )

        round_payload = {
            "round": int(round_idx),
            "oracle_queries": int(round_quality["oracle_queries"]),
            "best_score": float(round_quality["best_score"]),
            "top10_score": float(round_quality["top10_score"]),
            "diversity_entropy": float(round_quality["diversity_entropy"]),
            "valid_word_ratio": float(round_quality["valid_word_ratio"]),
            "mode_coverage": int(round_quality["mode_coverage"]),
            "surrogate_rmse": float(train_metrics["rmse"]),
            "acquisition": acquisition_name,
            "surrogate_type": getattr(surrogate, "surrogate_type", "unknown"),
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

    curve = build_query_curve(
        scores,
        optimum_score=optimum_score,
    )

    surrogate_path = output_dir / "active_final_surrogate.pt"
    if surrogate is not None:
        surrogate.save(surrogate_path)

    best_curve = running_best(scores)
    quality = search_quality_metrics(
        scores=scores,
        states=states,
        oracle_queries=int(oracle.call_count),
        optimum_score=optimum_score,
        top_k=10,
        pad_value=0,
    )
    result = {
        "method": "active_learning",
        "seed": seed,
        **quality,
        "best_score": float(best_curve[-1]) if best_curve.size > 0 else float(quality["best_score"]),
        "curve": curve,
        "round_logs": round_logs,
        "surrogate_path": str(surrogate_path) if surrogate is not None else None,
        "acquisition": acquisition_name,
        "surrogate_type": getattr(surrogate, "surrogate_type", "unknown")
        if surrogate is not None
        else "unknown",
        "sampling_strategy": initial_sampling_strategy,
        "candidate_sampling_strategy": candidate_sampling_strategy,
        "scores": [float(s) for s in scores],
        "real_oracle_queries": int(oracle.call_count),
        "fake_oracle_queries": 0,
        "candidate_pool_queries": int(candidate_pool_queries),
        "cheap_model_queries": int(candidate_pool_queries),
        "surrogate_fit_count": int(len(round_logs)),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "optimum_score": optimum_score,
    }
    if optimum_info.get("optimum_words"):
        result["optimum_words"] = list(optimum_info["optimum_words"])
        result["optimum_word_count"] = int(optimum_info["optimum_word_count"])
        result["optimum_source"] = str(optimum_info["optimum_source"])

    return result
