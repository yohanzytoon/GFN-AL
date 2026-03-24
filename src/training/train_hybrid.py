"""Hybrid GFlowNet + Active Learning training loop."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acquisition.factory import select_acquisition_batch
from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from training.common import build_surrogate_from_config, filter_new_states
from training.dataset import deduplicate_state_scores, sample_terminating_states
from training.train_gflownet import _train_upstream_gflownet, sample_gflownet_terminating_states
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, regression_metrics, running_best, search_quality_metrics


def run_hybrid_gflownet_active(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the hybrid pipeline: surrogate -> GFlowNet -> acquisition -> oracle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    hybrid_cfg = config["hybrid"]

    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=int(oracle_cfg["budget"]),
        device=device,
    )
    oracle = OracleProxy(
        device=device,
        float_precision=32,
        oracle_budget=int(oracle_cfg["budget"]),
        enforce_budget=True,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        stats_output_path=str(output_dir / "oracle_stats.json"),
    )
    oracle.setup(env)

    initial_size = int(min(hybrid_cfg.get("initial_size", 64), oracle_cfg["budget"]))
    batch_size = int(hybrid_cfg.get("batch_size", 16))
    candidate_pool_size = int(hybrid_cfg.get("candidate_pool_size", 256))
    fallback_random_pool_size = int(hybrid_cfg.get("fallback_random_pool_size", candidate_pool_size))
    max_rounds = int(hybrid_cfg.get("max_rounds", 50))
    sampling_strategy = str(hybrid_cfg.get("sampling_strategy", "uniform"))
    min_length = int(hybrid_cfg.get("min_length", 3))
    candidate_unique = bool(hybrid_cfg.get("candidate_unique", True))

    acquisition_cfg = dict(hybrid_cfg.get("acquisition", {}))
    acquisition_name = str(acquisition_cfg.get("name", "ucb")).lower()
    beta = float(acquisition_cfg.get("beta", 2.0))
    xi = float(acquisition_cfg.get("xi", 0.0))
    thompson_samples = int(acquisition_cfg.get("thompson_samples", 1))

    surrogate_cfg = dict(hybrid_cfg.get("surrogate", {}))
    gflownet_schedule_cfg = dict(hybrid_cfg.get("gflownet", {}))
    retrain_every = max(int(gflownet_schedule_cfg.get("retrain_every", 1)), 1)
    gflownet_sample_size = int(gflownet_schedule_cfg.get("sample_size", candidate_pool_size))
    num_tokens = int(env_cfg.get("num_tokens", 27))

    states = sample_terminating_states(
        env,
        initial_size,
        sampling_strategy=sampling_strategy,
        min_length=min_length,
        unique=True,
        seed=seed,
    )
    scores = oracle(env.states2proxy(states)).detach().cpu().numpy().astype(float).tolist()

    surrogate = None
    gfn = None
    gflownet_round_dirs: list[str] = []
    round_logs: list[dict[str, Any]] = []

    for round_idx in range(max_rounds):
        if oracle.remaining_budget <= 0:
            break

        train_states, train_scores = deduplicate_state_scores(states, scores)
        surrogate = build_surrogate_from_config(
            surrogate_cfg,
            max_length=int(env_cfg["max_length"]),
            num_tokens=num_tokens,
            device=device,
        )
        surrogate.fit(train_states, train_scores)
        surrogate_path = output_dir / f"surrogate_round_{round_idx:03d}.pt"
        surrogate.save(surrogate_path)

        if gfn is None or round_idx % retrain_every == 0:
            if gfn is not None and hasattr(gfn, "logger"):
                gfn.logger.end()
            round_gflownet_dir = output_dir / f"gflownet_round_{round_idx:03d}"
            round_gflownet_dir.mkdir(parents=True, exist_ok=True)
            gflownet_round_dirs.append(str(round_gflownet_dir))

            gflownet_config = copy.deepcopy(config)
            gflownet_config["gflownet"] = copy.deepcopy(gflownet_schedule_cfg)
            gfn, _ = _train_upstream_gflownet(
                gflownet_config,
                output_dir=round_gflownet_dir,
                proxy_target="proxies.surrogate_proxy.SurrogateProxy",
                proxy_kwargs={
                    "surrogate_path": str(surrogate_path),
                    "prediction_mode": str(gflownet_schedule_cfg.get("prediction_mode", "mean")),
                    "exploration_beta": float(gflownet_schedule_cfg.get("exploration_beta", 1.0)),
                    "reward_transform": str(gflownet_schedule_cfg.get("reward_transform", "softplus")),
                    "reward_function": "identity",
                    "reward_min": float(gflownet_schedule_cfg.get("reward_min", 1e-4)),
                    "do_clip_rewards": bool(gflownet_schedule_cfg.get("do_clip_rewards", True)),
                },
            )

        gflownet_candidates = sample_gflownet_terminating_states(gfn, gflownet_sample_size)
        filtered_candidates = filter_new_states(gflownet_candidates, states)
        if len(filtered_candidates) < batch_size:
            filtered_candidates.extend(
                filter_new_states(
                    sample_terminating_states(
                        env,
                        fallback_random_pool_size,
                        sampling_strategy="uniform",
                        min_length=min_length,
                        unique=candidate_unique,
                        seed=seed + 50_000 + round_idx,
                    ),
                    states,
                )
            )
            filtered_candidates = filter_new_states(filtered_candidates, states)

        candidate_matrix = np.asarray(filtered_candidates, dtype=np.int64)
        if candidate_matrix.size == 0:
            break

        mean, std = surrogate.predict(candidate_matrix, return_std=True)
        this_batch = int(min(batch_size, oracle.remaining_budget, candidate_matrix.shape[0]))
        if this_batch <= 0:
            break

        selected_idx = select_acquisition_batch(
            acquisition_name,
            mean=mean,
            std=std,
            batch_size=this_batch,
            best_f=float(np.max(train_scores)) if len(train_scores) > 0 else 0.0,
            surrogate=surrogate,
            candidate_states=candidate_matrix,
            beta=beta,
            xi=xi,
            thompson_samples=thompson_samples,
        )
        selected_states = candidate_matrix[selected_idx].tolist()
        queried_scores = (
            oracle(np.asarray(selected_states, dtype=np.int64)).detach().cpu().numpy().tolist()
        )

        states.extend(selected_states)
        scores.extend(float(score) for score in queried_scores)

        train_pred = surrogate.predict(np.asarray(states, dtype=np.int64), return_std=False)
        train_metrics = regression_metrics(np.asarray(scores, dtype=float), np.asarray(train_pred, dtype=float))
        round_quality = search_quality_metrics(
            scores=scores,
            states=states,
            oracle_queries=int(oracle.call_count),
            optimum_score=config.get("metrics", {}).get("optimum_score"),
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
            "surrogate_type": getattr(surrogate, "surrogate_type", "unknown"),
            "acquisition": acquisition_name,
            "gflownet_candidates": int(len(gflownet_candidates)),
            "unique_candidates_after_filter": int(candidate_matrix.shape[0]),
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    best_curve = running_best(scores)
    quality = search_quality_metrics(
        scores=scores,
        states=states,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    result = {
        "method": "hybrid_gflownet_active",
        "seed": seed,
        **quality,
        "best_score": float(best_curve[-1]) if best_curve.size > 0 else float(quality["best_score"]),
        "curve": curve,
        "round_logs": round_logs,
        "oracle_queries": int(oracle.call_count),
        "acquisition": acquisition_name,
        "surrogate_type": getattr(surrogate, "surrogate_type", "unknown")
        if surrogate is not None
        else "unknown",
        "gflownet_round_dirs": gflownet_round_dirs,
        "scores": [float(score) for score in scores],
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_hybrid.json")

    if gfn is not None and hasattr(gfn, "logger"):
        gfn.logger.end()
    return result
