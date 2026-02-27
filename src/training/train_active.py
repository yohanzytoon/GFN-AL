"""Pure active-learning training loop on Scrabble with budgeted oracle access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from acquisition.ei import select_ei
from acquisition.thompson import select_thompson
from acquisition.ucb import select_ucb
from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from surrogate import build_surrogate
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import (
    build_query_curve,
    regression_metrics,
    running_best,
    search_quality_metrics,
)


def _select_batch(
    acquisition: str,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    best_observed: float,
    beta: float,
    xi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    acquisition = acquisition.lower()
    if acquisition == "ucb":
        return select_ucb(mean=mean, std=std, batch_size=batch_size, beta=beta)
    if acquisition == "ei":
        return select_ei(
            mean=mean,
            std=std,
            best_observed=best_observed,
            batch_size=batch_size,
            xi=xi,
        )
    if acquisition == "thompson":
        return select_thompson(
            mean=mean,
            std=std,
            batch_size=batch_size,
            random_state=rng,
        )
    if acquisition == "uncertainty":
        return np.argsort(std)[::-1][:batch_size]
    raise KeyError(f"Unknown acquisition function: {acquisition}")


def run_active_learning(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run pure active learning and return experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

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
        enforce_budget=True,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        backend="oracle",
    )
    oracle.setup(env)

    initial_size = int(min(active_cfg.get("initial_size", 50), oracle_cfg["budget"]))
    batch_size = int(active_cfg.get("batch_size", 16))
    candidate_pool_size = int(active_cfg.get("candidate_pool_size", 256))
    max_rounds = int(active_cfg.get("max_rounds", 200))
    acquisition = str(active_cfg.get("acquisition", "ucb"))
    beta = float(active_cfg.get("acquisition_beta", 2.0))
    xi = float(active_cfg.get("ei_xi", 0.01))
    surrogate_type = str(active_cfg.get("surrogate_type", "gp"))

    surrogate_kwargs = dict(active_cfg.get("surrogate", {}))
    surrogate_kwargs.setdefault("max_length", int(env_cfg["max_length"]))
    surrogate_kwargs.setdefault("device", device)

    states = env.get_random_terminating_states(
        n_states=initial_size,
        unique=False,
        max_attempts=max(5 * initial_size, 1000),
    )
    scores = oracle(env.states2proxy(states)).detach().cpu().numpy().astype(float).tolist()

    round_logs: list[dict[str, Any]] = []
    surrogate = None

    for round_idx in range(max_rounds):
        if oracle.remaining_budget <= 0:
            break

        surrogate = build_surrogate(surrogate_type=surrogate_type, **surrogate_kwargs)
        train_states = np.asarray(states, dtype=np.int64)
        train_scores = np.asarray(scores, dtype=np.float32)
        surrogate.fit(train_states, train_scores)

        candidate_states = env.get_random_terminating_states(
            n_states=candidate_pool_size,
            unique=False,
            max_attempts=max(5 * candidate_pool_size, 1000),
        )
        candidate_matrix = np.asarray(candidate_states, dtype=np.int64)
        mean, std = surrogate.predict(candidate_matrix, return_std=True)

        this_batch = int(min(batch_size, oracle.remaining_budget, candidate_matrix.shape[0]))
        if this_batch <= 0:
            break

        selected_idx = _select_batch(
            acquisition=acquisition,
            mean=mean,
            std=std,
            batch_size=this_batch,
            best_observed=float(np.max(train_scores)),
            beta=beta,
            xi=xi,
            rng=rng,
        )
        selected_states = candidate_matrix[selected_idx].tolist()
        queried_scores = (
            oracle(np.asarray(selected_states, dtype=np.int64)).detach().cpu().numpy().tolist()
        )

        states.extend(selected_states)
        scores.extend(float(s) for s in queried_scores)

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
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )

    surrogate_path = output_dir / "active_final_surrogate.pt"
    if surrogate is not None:
        surrogate.save(surrogate_path)

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
        "method": "active_learning",
        "seed": seed,
        **quality,
        "best_score": float(best_curve[-1]) if best_curve.size > 0 else float(quality["best_score"]),
        "curve": curve,
        "round_logs": round_logs,
        "surrogate_path": str(surrogate_path) if surrogate is not None else None,
        "acquisition": acquisition,
        "surrogate_type": surrogate_type,
        "scores": [float(s) for s in scores],
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_active.json")

    return result
