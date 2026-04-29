"""Hybrid GFlowNet + Active Learning — GFlowNet-only candidate generation.

The only candidates fed to the acquisition function are states sampled directly
from a GFlowNet trained on the surrogate as a proxy reward.

Before the first GFlowNet trains, the surrogate is bootstrapped with a
heuristic oracle-labelled warm-up phase so it is not learned from near-random
zero-heavy data alone.
"""

from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.ucb import select_acquisition_batch
from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from training.common import (
    build_surrogate_from_config,
    compute_plausibility_bonus,
    filter_new_states,
)
from training.dataset import (
    deduplicate_state_scores,
    sample_terminating_states,
)
from training.train_gflownet import (
    _train_upstream_gflownet,
    sample_gflownet_terminating_states,
)
from utils.device import resolve_device
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, regression_metrics, running_best, search_quality_metrics
from utils.scrabble import resolve_scrabble_optimum


def _sample_gfn_only_candidates(
    *,
    gfn,
    seen_states: list[list[int]],
    gflownet_sample_size: int,
    gflownet_sample_batches: int,
    gflownet_max_sample_batches: int,
    min_required: int,
    min_length: int = 0,
    max_length: int | None = None,
    gflownet_root: str | None = None,
    min_plausibility: float | None = None,
) -> tuple[list[list[int]], dict[str, int]]:
    """Sample candidates exclusively from the GFlowNet.

    If the GFlowNet collapses to duplicates, keep drawing additional batches up
    to a configured cap so acquisition sees a non-trivial unique pool.
    """
    if gfn is None or gflownet_sample_size <= 0:
        return [], {
            "gflownet_candidates": 0,
            "gflownet_sample_batches_used": 0,
        }

    requested_batches = max(int(gflownet_sample_batches), 1)
    max_batches = max(int(gflownet_max_sample_batches), requested_batches)
    gflownet_candidates: list[list[int]] = []
    filtered: list[list[int]] = []
    previous_unique_count = -1
    batches_used = 0

    while (
        batches_used < requested_batches
        or (
            len(filtered) < min_required
            and batches_used < max_batches
            and len(filtered) > previous_unique_count
        )
    ):
        previous_unique_count = len(filtered)
        gflownet_candidates.extend(
            sample_gflownet_terminating_states(gfn, gflownet_sample_size)
        )
        batches_used += 1
        filtered = filter_new_states(gflownet_candidates, seen_states)
        filtered = _filter_gfn_candidates(
            filtered,
            min_length=min_length,
            max_length=max_length,
            gflownet_root=gflownet_root,
            min_plausibility=min_plausibility,
        )

    return filtered, {
        "gflownet_candidates": int(len(gflownet_candidates)),
        "gflownet_sample_batches_used": int(batches_used),
        "unique_candidates_after_quality_filter": int(len(filtered)),
    }


def _state_length(state: list[int] | np.ndarray) -> int:
    array = np.asarray(state, dtype=np.int64).reshape(-1)
    zero_idx = np.where(array == 0)[0]
    if zero_idx.size == 0:
        return int(array.shape[0])
    return int(zero_idx[0])


def _filter_gfn_candidates(
    candidate_states: list[list[int]],
    *,
    min_length: int = 0,
    max_length: int | None = None,
    gflownet_root: str | None = None,
    min_plausibility: float | None = None,
) -> list[list[int]]:
    """Drop GFlowNet samples that cannot be useful oracle queries."""
    if not candidate_states:
        return []

    filtered = [
        state
        for state in candidate_states
        if int(_state_length(state)) >= int(min_length)
    ]
    if (
        not filtered
        or min_plausibility is None
        or float(min_plausibility) <= 0.0
        or max_length is None
    ):
        return filtered

    candidate_matrix = np.asarray(filtered, dtype=np.int64)
    plausibility = compute_plausibility_bonus(
        candidate_matrix,
        int(max_length),
        gflownet_root=gflownet_root,
        weight=1.0,
    )
    # If the vocabulary model is unavailable, compute_plausibility_bonus returns
    # all zeros. In that case, do not accidentally filter the whole GFN pool.
    if np.max(plausibility) <= 0.0:
        return filtered
    keep = plausibility >= float(min_plausibility)
    return candidate_matrix[keep].tolist()


def _adaptive_proxy_score_max(
    scores: list[float] | np.ndarray,
    gflownet_cfg: dict[str, Any],
    config: dict[str, Any],
) -> float:
    """Scale exp_beta rewards to the score range observed so far."""
    configured = float(
        gflownet_cfg.get(
            "score_max",
            config.get("metrics", {}).get("optimum_score", 45.0),
        )
    )
    if not bool(gflownet_cfg.get("adaptive_score_max", False)):
        return max(configured, 1e-6)

    values = np.asarray(scores, dtype=float)
    positives = values[values > 0.0]
    floor = float(gflownet_cfg.get("score_max_floor", 10.0))
    quantile = float(gflownet_cfg.get("score_max_quantile", 0.9))
    if positives.size == 0:
        return max(floor, 1e-6)

    adaptive = max(
        floor,
        float(np.quantile(positives, np.clip(quantile, 0.0, 1.0))),
        float(np.max(positives)),
    )
    ceiling = gflownet_cfg.get("score_max_ceiling")
    if ceiling is not None:
        adaptive = min(adaptive, float(ceiling))
    return max(adaptive, 1e-6)


def _write_offline_train_dataset(
    path: Path,
    states: np.ndarray,
    scores: np.ndarray,
    *,
    max_states: int,
    min_score: float,
) -> Path | None:
    """Write high-value labelled states for upstream backward-dataset sampling."""
    if max_states <= 0 or states.size == 0:
        return None

    scores_np = np.asarray(scores, dtype=float).reshape(-1)
    states_np = np.asarray(states, dtype=np.int64)
    ranked = np.argsort(scores_np)[::-1]
    keep = [int(idx) for idx in ranked if scores_np[int(idx)] > float(min_score)]
    if not keep:
        keep = [int(idx) for idx in ranked[: min(max_states, len(ranked))]]
    keep = keep[: int(max_states)]
    if not keep:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"samples": states_np[keep].tolist()}, f)
    return path


def _bootstrap_oracle_dataset(
    *,
    env: ScrabbleOracleEnv,
    oracle: OracleProxy,
    initial_states: list[list[int]],
    initial_scores: list[float],
    bootstrap_pool_size: int,
    bootstrap_batch_size: int,
    bootstrap_sampling_strategy: str,
    min_length: int,
    candidate_unique: bool,
    final_oracle_reserve: int,
    target_min_rounds: int,
    target_positive_count: int,
    seed: int,
    gflownet_root: str | None,
    max_bootstrap_queries: int,
) -> tuple[list[list[int]], list[float], list[dict[str, int]], int]:
    """Collect heuristic oracle labels before handing off to the GFlowNet.

    Bootstrap is hard-capped at `max_bootstrap_queries` so it cannot starve the
    active-learning rounds — the GFlowNet needs budget to drive the loop.
    """
    states = list(initial_states)
    scores = [float(score) for score in initial_scores]
    logs: list[dict[str, int]] = []
    bootstrap_rounds = 0
    bootstrap_queries_used = 0

    while oracle.remaining_budget > final_oracle_reserve:
        positive_count = int(np.sum(np.asarray(scores, dtype=float) > 0.0))
        if (
            bootstrap_rounds >= int(target_min_rounds)
            and positive_count >= int(target_positive_count)
        ):
            break
        if bootstrap_queries_used >= int(max_bootstrap_queries):
            break

        query_budget = int(
            min(
                bootstrap_batch_size,
                max(int(oracle.remaining_budget) - int(final_oracle_reserve), 0),
                max(int(max_bootstrap_queries) - bootstrap_queries_used, 0),
            )
        )
        if query_budget <= 0:
            break

        candidate_states = sample_terminating_states(
            env,
            bootstrap_pool_size,
            sampling_strategy=bootstrap_sampling_strategy,
            min_length=min_length,
            unique=candidate_unique,
            seed=seed + bootstrap_rounds,
            gflownet_root=gflownet_root,
        )
        filtered_candidates = filter_new_states(candidate_states, states)
        if len(filtered_candidates) == 0:
            break

        queried_states = filtered_candidates[:query_budget]
        queried_scores = (
            oracle(np.asarray(queried_states, dtype=np.int64))
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )
        states.extend(queried_states)
        scores.extend(float(score) for score in queried_scores)
        bootstrap_rounds += 1
        bootstrap_queries_used += int(len(queried_states))
        logs.append(
            {
                "bootstrap_round": int(bootstrap_rounds),
                "queried_states": int(len(queried_states)),
                "positive_count": int(np.sum(np.asarray(scores, dtype=float) > 0.0)),
                "remaining_oracle_budget": int(oracle.remaining_budget),
            }
        )

    return states, scores, logs, bootstrap_rounds


def run_hybrid_gfn_only(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the GFlowNet-only hybrid pipeline.

    Surrogate is trained each round on all oracle-queried (state, score) pairs.
    The GFlowNet is trained with the surrogate as a reward proxy.
    Candidates for the acquisition function come *only* from GFlowNet samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    seed = int(config["seed"])
    device = resolve_device(config.get("device", "cpu"))
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    hybrid_cfg = config["hybrid"]
    oracle_budget = int(oracle_cfg["budget"])

    optimum_info = resolve_scrabble_optimum(
        max_length=int(env_cfg["max_length"]),
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        configured_optimum_score=config.get("metrics", {}).get("optimum_score"),
        gflownet_root=config.get("gflownet_root"),
    )
    optimum_score = float(optimum_info["optimum_score"])

    env = ScrabbleOracleEnv(
        max_length=int(env_cfg["max_length"]),
        oracle_budget=oracle_budget,
        device=device,
    )
    oracle = OracleProxy(
        device=device,
        float_precision=32,
        oracle_budget=oracle_budget,
        enforce_budget=bool(oracle_cfg.get("enforce_budget", True)),
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        stats_output_path=str(output_dir / "oracle_stats.json"),
    )
    oracle.setup(env)

    # ------------------------------------------------------------------ config
    initial_size = int(min(hybrid_cfg.get("initial_size", 64), oracle_budget))
    batch_size = int(hybrid_cfg.get("batch_size", 16))
    max_rounds = int(hybrid_cfg.get("max_rounds", 50))

    initial_sampling_strategy = str(
        hybrid_cfg.get(
            "initial_sampling_strategy",
            hybrid_cfg.get("sampling_strategy", "frequency"),
        )
    )
    fallback_sampling_strategy = str(
        hybrid_cfg.get("fallback_sampling_strategy", "frequency")
    )
    min_length = int(hybrid_cfg.get("min_length", 3))
    candidate_unique = bool(hybrid_cfg.get("candidate_unique", True))

    acquisition_cfg = dict(hybrid_cfg.get("acquisition", {}))
    acquisition_name = str(acquisition_cfg.get("name", "ucb")).lower()
    beta = float(acquisition_cfg.get("beta", 2.0))
    beta_min = float(acquisition_cfg.get("beta_min", beta))
    final_beta = float(acquisition_cfg.get("final_beta", 0.0))
    final_acquisition_name = str(
        acquisition_cfg.get("final_name", acquisition_name)
    ).lower()

    surrogate_cfg = dict(hybrid_cfg.get("surrogate", {}))
    gflownet_schedule_cfg = dict(hybrid_cfg.get("gflownet", {}))
    retrain_every = max(int(gflownet_schedule_cfg.get("retrain_every", 1)), 1)
    gflownet_sample_size = int(gflownet_schedule_cfg.get("sample_size", 256))
    gflownet_sample_batches = int(gflownet_schedule_cfg.get("sample_batches", 1))
    gflownet_max_sample_batches = int(
        gflownet_schedule_cfg.get(
            "max_sample_batches",
            max(gflownet_sample_batches, 1),
        )
    )
    refresh_before_final_selection = bool(
        gflownet_schedule_cfg.get("refresh_before_final_selection", True)
    )
    warm_start = bool(gflownet_schedule_cfg.get("warm_start", True))
    gflownet_steps_per_refresh = int(gflownet_schedule_cfg.get("n_train_steps", 1000))
    gflownet_start_round = int(gflownet_schedule_cfg.get("start_round", 2))
    gflownet_min_positive_count = int(
        gflownet_schedule_cfg.get("min_positive_count", max(batch_size // 2, 8))
    )
    offline_train_max_states = int(
        gflownet_schedule_cfg.get("offline_train_max_states", 256)
    )
    offline_train_min_score = float(
        gflownet_schedule_cfg.get("offline_train_min_score", 0.0)
    )
    gflownet_candidate_min_plausibility_cfg = gflownet_schedule_cfg.get(
        "candidate_min_plausibility",
        hybrid_cfg.get("gflownet_candidate_min_plausibility"),
    )
    gflownet_candidate_min_plausibility = (
        None
        if gflownet_candidate_min_plausibility_cfg is None
        else float(gflownet_candidate_min_plausibility_cfg)
    )
    final_candidate_min_plausibility_cfg = gflownet_schedule_cfg.get(
        "final_candidate_min_plausibility",
        gflownet_candidate_min_plausibility,
    )
    final_candidate_min_plausibility = (
        None
        if final_candidate_min_plausibility_cfg is None
        else float(final_candidate_min_plausibility_cfg)
    )
    final_gate_min_score_fraction = float(
        gflownet_schedule_cfg.get("final_gate_min_score_fraction", 0.75)
    )

    bootstrap_pool_size = int(
        hybrid_cfg.get(
            "bootstrap_pool_size",
            hybrid_cfg.get("fallback_pool_size", max(batch_size * 32, 512)),
        )
    )
    bootstrap_batch_size = int(hybrid_cfg.get("bootstrap_batch_size", batch_size))
    bootstrap_sampling_strategy = str(
        hybrid_cfg.get("bootstrap_sampling_strategy", fallback_sampling_strategy)
    )
    bootstrap_min_rounds = int(
        hybrid_cfg.get("bootstrap_min_rounds", gflownet_start_round)
    )
    bootstrap_min_positive_count = int(
        hybrid_cfg.get("bootstrap_min_positive_count", gflownet_min_positive_count)
    )
    bootstrap_budget_fraction = float(
        hybrid_cfg.get("bootstrap_budget_fraction", 0.35)
    )

    post_initial_budget = max(oracle_budget - initial_size, 0)
    configured_final_oracle_reserve = hybrid_cfg.get("final_oracle_reserve")
    default_final_oracle_reserve = min(
        max(batch_size * 8, int(0.2 * oracle_budget)),
        post_initial_budget // 2 if post_initial_budget > 0 else 0,
    )
    final_oracle_reserve = int(
        default_final_oracle_reserve
        if configured_final_oracle_reserve is None
        else configured_final_oracle_reserve
    )
    final_oracle_reserve = max(0, min(final_oracle_reserve, post_initial_budget))
    final_candidate_pool_size = int(
        hybrid_cfg.get(
            "final_candidate_pool_size",
            max(gflownet_sample_size * 4, batch_size * 8),
        )
    )

    num_tokens = int(env_cfg.get("num_tokens", 27))
    diversity_weight = float(hybrid_cfg.get("diversity_weight", 0.3))
    plausibility_bonus_weight = float(hybrid_cfg.get("plausibility_bonus_weight", 2.0))
    gflownet_root = config.get("gflownet_root")

    # --------------------------------------------------------------- warm-up
    initial_states = sample_terminating_states(
        env,
        initial_size,
        sampling_strategy=initial_sampling_strategy,
        min_length=min_length,
        unique=True,
        seed=seed,
        gflownet_root=gflownet_root,
    )
    initial_scores = (
        oracle(env.states2proxy(initial_states))
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )
    max_bootstrap_queries = max(
        int(bootstrap_budget_fraction * post_initial_budget),
        int(bootstrap_batch_size),
    )
    states, scores, bootstrap_logs, bootstrap_rounds = _bootstrap_oracle_dataset(
        env=env,
        oracle=oracle,
        initial_states=initial_states,
        initial_scores=initial_scores,
        bootstrap_pool_size=bootstrap_pool_size,
        bootstrap_batch_size=bootstrap_batch_size,
        bootstrap_sampling_strategy=bootstrap_sampling_strategy,
        min_length=min_length,
        candidate_unique=candidate_unique,
        final_oracle_reserve=final_oracle_reserve,
        target_min_rounds=bootstrap_min_rounds,
        target_positive_count=bootstrap_min_positive_count,
        seed=seed + 100_000,
        gflownet_root=gflownet_root,
        max_bootstrap_queries=max_bootstrap_queries,
    )
    bootstrap_oracle_queries = int(len(scores))

    surrogate = None
    gfn = None
    gflownet_round_dirs: list[str] = []
    round_logs: list[dict[str, Any]] = []
    final_selection_log: dict[str, Any] | None = None

    surrogate_fit_count = 0
    candidate_pool_queries = 0
    fake_oracle_queries = 0
    gflownet_train_calls = 0
    gflownet_total_train_steps = 0
    last_gflownet_stop_reason = None

    # ----------------------------------------------------------------- rounds
    for round_idx in range(max_rounds):
        remaining_for_rounds = max(0, oracle.remaining_budget - final_oracle_reserve)
        if remaining_for_rounds <= 0:
            break

        round_beta = beta - (beta - beta_min) * round_idx / max(max_rounds - 1, 1)
        effective_round = bootstrap_rounds + round_idx

        train_states, train_scores = deduplicate_state_scores(states, scores)
        surrogate = build_surrogate_from_config(
            surrogate_cfg,
            max_length=int(env_cfg["max_length"]),
            num_tokens=num_tokens,
            device=device,
        )
        surrogate.fit(train_states, train_scores)
        surrogate_fit_count += 1
        surrogate_path = output_dir / "surrogate_current.pt"
        surrogate.save(surrogate_path)

        positive_count = int(np.sum(np.asarray(train_scores, dtype=float) > 0.0))
        gflownet_ready = (
            effective_round >= gflownet_start_round
            and positive_count >= gflownet_min_positive_count
        )
        if gflownet_ready and (gfn is None or round_idx % retrain_every == 0):
            previous_gfn = gfn if warm_start else None
            if gfn is not None and hasattr(gfn, "logger"):
                gfn.logger.end()

            round_gflownet_dir = output_dir / f"gflownet_round_{round_idx:03d}"
            round_gflownet_dir.mkdir(parents=True, exist_ok=True)
            gflownet_round_dirs.append(str(round_gflownet_dir))

            gflownet_config = copy.deepcopy(config)
            gflownet_config["gflownet"] = copy.deepcopy(gflownet_schedule_cfg)
            offline_train_path = _write_offline_train_dataset(
                round_gflownet_dir / "offline_train.pkl",
                train_states,
                train_scores,
                max_states=offline_train_max_states,
                min_score=offline_train_min_score,
            )
            if offline_train_path is not None:
                gflownet_config["gflownet"]["train_data_path"] = str(offline_train_path)
            proxy_score_max = _adaptive_proxy_score_max(
                train_scores,
                gflownet_schedule_cfg,
                config,
            )
            gfn, stop_reason = _train_upstream_gflownet(
                gflownet_config,
                output_dir=round_gflownet_dir,
                proxy_target="proxies.surrogate_proxy.SurrogateProxy",
                proxy_kwargs={
                    "surrogate_path": str(surrogate_path),
                    # Use surrogate mean, not UCB — UCB biases the GFlowNet
                    # toward uncertain regions rather than high-value regions.
                    "prediction_mode": str(
                        gflownet_schedule_cfg.get("prediction_mode", "mean")
                    ),
                    "exploration_beta": float(
                        gflownet_schedule_cfg.get("exploration_beta", 1.0)
                    ),
                    # exp_beta gives a sharp Boltzmann reward; see SurrogateProxy.
                    "reward_transform": str(
                        gflownet_schedule_cfg.get("reward_transform", "exp_beta")
                    ),
                    "beta_scale": float(
                        gflownet_schedule_cfg.get("beta_scale", 5.0)
                    ),
                    "score_max": float(proxy_score_max),
                    "plausibility_mode": str(
                        gflownet_schedule_cfg.get("plausibility_mode", "multiplicative")
                    ),
                    "reward_function": "identity",
                    "reward_min": float(gflownet_schedule_cfg.get("reward_min", 1e-4)),
                    "do_clip_rewards": bool(
                        gflownet_schedule_cfg.get("do_clip_rewards", True)
                    ),
                    "max_length": int(env_cfg["max_length"]),
                    "min_length": int(min_length),
                    "gflownet_root": gflownet_root,
                    "plausibility_weight": float(plausibility_bonus_weight),
                    "stats_output_path": str(round_gflownet_dir / "surrogate_stats.json"),
                },
                warm_start_from=previous_gfn,
            )
            gflownet_train_calls += 1
            gflownet_total_train_steps += gflownet_steps_per_refresh
            fake_oracle_queries += int(getattr(getattr(gfn, "proxy", None), "call_count", 0))
            last_gflownet_stop_reason = stop_reason

        # Candidate generation: GFlowNet only.
        sampled_candidates, candidate_breakdown = _sample_gfn_only_candidates(
            gfn=gfn,
            seen_states=states,
            gflownet_sample_size=gflownet_sample_size,
            gflownet_sample_batches=gflownet_sample_batches,
            gflownet_max_sample_batches=gflownet_max_sample_batches,
            min_required=batch_size,
            min_length=min_length,
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            min_plausibility=gflownet_candidate_min_plausibility,
        )

        candidate_matrix = np.asarray(sampled_candidates, dtype=np.int64)
        if candidate_matrix.size == 0:
            break

        candidate_pool_queries += int(candidate_matrix.shape[0])
        mean, std = surrogate.predict(candidate_matrix, return_std=True)
        mean = mean + compute_plausibility_bonus(
            candidate_matrix,
            int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            weight=plausibility_bonus_weight,
        )

        this_batch = int(min(batch_size, remaining_for_rounds, candidate_matrix.shape[0]))
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
        queried_scores = (
            oracle(np.asarray(selected_states, dtype=np.int64)).detach().cpu().numpy().tolist()
        )

        states.extend(selected_states)
        scores.extend(float(s) for s in queried_scores)

        train_pred = surrogate.predict(np.asarray(states, dtype=np.int64), return_std=False)
        train_metrics = regression_metrics(
            np.asarray(scores, dtype=float),
            np.asarray(train_pred, dtype=float),
        )
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
            "surrogate_type": getattr(surrogate, "surrogate_type", "unknown"),
            "acquisition": acquisition_name,
            "effective_round": int(effective_round),
            "positive_count": int(positive_count),
            "gflownet_ready": bool(gflownet_ready),
            "gflownet_candidates": int(candidate_breakdown["gflownet_candidates"]),
            "gflownet_sample_batches_used": int(
                candidate_breakdown["gflownet_sample_batches_used"]
            ),
            "unique_candidates_after_filter": int(candidate_matrix.shape[0]),
            "reserved_final_oracle_budget": int(final_oracle_reserve),
            "remaining_oracle_budget": int(oracle.remaining_budget),
            "fake_oracle_queries_total": int(fake_oracle_queries),
            "candidate_pool_queries_total": int(candidate_pool_queries),
            "gflownet_train_calls": int(gflownet_train_calls),
            "gflownet_total_train_steps": int(gflownet_total_train_steps),
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

    # --------------------------------------------------------- final selection
    if oracle.remaining_budget > 0:
        train_states, train_scores = deduplicate_state_scores(states, scores)
        surrogate = build_surrogate_from_config(
            surrogate_cfg,
            max_length=int(env_cfg["max_length"]),
            num_tokens=num_tokens,
            device=device,
        )
        surrogate.fit(train_states, train_scores)
        surrogate_fit_count += 1
        final_surrogate_path = output_dir / "surrogate_final.pt"
        surrogate.save(final_surrogate_path)

        if refresh_before_final_selection:
            previous_gfn = gfn if warm_start else None
            if gfn is not None and hasattr(gfn, "logger"):
                gfn.logger.end()

            final_gflownet_dir = output_dir / "gflownet_final_selection"
            final_gflownet_dir.mkdir(parents=True, exist_ok=True)
            gflownet_round_dirs.append(str(final_gflownet_dir))

            gflownet_config = copy.deepcopy(config)
            gflownet_config["gflownet"] = copy.deepcopy(gflownet_schedule_cfg)
            offline_train_path = _write_offline_train_dataset(
                final_gflownet_dir / "offline_train.pkl",
                train_states,
                train_scores,
                max_states=offline_train_max_states,
                min_score=offline_train_min_score,
            )
            if offline_train_path is not None:
                gflownet_config["gflownet"]["train_data_path"] = str(offline_train_path)
            proxy_score_max = _adaptive_proxy_score_max(
                train_scores,
                gflownet_schedule_cfg,
                config,
            )
            gfn, stop_reason = _train_upstream_gflownet(
                gflownet_config,
                output_dir=final_gflownet_dir,
                proxy_target="proxies.surrogate_proxy.SurrogateProxy",
                proxy_kwargs={
                    "surrogate_path": str(final_surrogate_path),
                    # Final selection: always use mean (pure exploitation).
                    "prediction_mode": "mean",
                    "exploration_beta": 0.0,
                    "reward_transform": str(
                        gflownet_schedule_cfg.get("reward_transform", "exp_beta")
                    ),
                    "beta_scale": float(
                        gflownet_schedule_cfg.get("beta_scale", 5.0)
                    ),
                    "score_max": float(proxy_score_max),
                    "plausibility_mode": str(
                        gflownet_schedule_cfg.get("plausibility_mode", "multiplicative")
                    ),
                    "reward_function": "identity",
                    "reward_min": float(gflownet_schedule_cfg.get("reward_min", 1e-4)),
                    "do_clip_rewards": bool(
                        gflownet_schedule_cfg.get("do_clip_rewards", True)
                    ),
                    "max_length": int(env_cfg["max_length"]),
                    "min_length": int(min_length),
                    "gflownet_root": gflownet_root,
                    "plausibility_weight": float(plausibility_bonus_weight),
                    "stats_output_path": str(final_gflownet_dir / "surrogate_stats.json"),
                },
                warm_start_from=previous_gfn,
            )
            gflownet_train_calls += 1
            gflownet_total_train_steps += gflownet_steps_per_refresh
            fake_oracle_queries += int(getattr(getattr(gfn, "proxy", None), "call_count", 0))
            last_gflownet_stop_reason = stop_reason

        final_gflownet_sample_size = int(
            gflownet_schedule_cfg.get(
                "final_gflownet_sample_size",
                hybrid_cfg.get("final_candidate_pool_size", final_candidate_pool_size),
            )
        )
        final_gflownet_sample_batches = int(
            gflownet_schedule_cfg.get(
                "final_sample_batches",
                gflownet_sample_batches,
            )
        )
        final_gflownet_max_sample_batches = int(
            gflownet_schedule_cfg.get(
                "final_max_sample_batches",
                max(final_gflownet_sample_batches, gflownet_max_sample_batches),
            )
        )
        final_candidates, candidate_breakdown = _sample_gfn_only_candidates(
            gfn=gfn,
            seen_states=states,
            gflownet_sample_size=final_gflownet_sample_size,
            gflownet_sample_batches=final_gflownet_sample_batches,
            gflownet_max_sample_batches=final_gflownet_max_sample_batches,
            min_required=int(min(oracle.remaining_budget, final_candidate_pool_size)),
            min_length=min_length,
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            min_plausibility=final_candidate_min_plausibility,
        )
        candidate_matrix = np.asarray(final_candidates, dtype=np.int64)
        if candidate_matrix.size > 0:
            candidate_pool_queries += int(candidate_matrix.shape[0])
            mean, std = surrogate.predict(candidate_matrix, return_std=True)
            raw_mean = np.asarray(mean, dtype=float).copy()
            mean = mean + compute_plausibility_bonus(
                candidate_matrix,
                int(env_cfg["max_length"]),
                gflownet_root=gflownet_root,
                weight=plausibility_bonus_weight,
            )

            final_batch = int(min(oracle.remaining_budget, candidate_matrix.shape[0]))
            score_values = np.asarray(scores, dtype=float)
            current_top = np.sort(score_values)[-min(10, score_values.size) :]
            final_gate_threshold = (
                float(np.mean(current_top)) * final_gate_min_score_fraction
                if current_top.size > 0
                else 0.0
            )
            if (
                final_batch > 0
                and final_gate_threshold > 0.0
                and float(np.max(raw_mean)) < final_gate_threshold
            ):
                final_selection_log = {
                    "skipped": True,
                    "reason": "candidate_predictions_below_gate",
                    "oracle_queries_before": int(oracle.call_count),
                    "oracle_queries_after": int(oracle.call_count),
                    "selected_candidates": 0,
                    "best_selected_score": 0.0,
                    "mean_selected_score": 0.0,
                    "max_predicted_score": float(np.max(raw_mean)),
                    "gate_threshold": float(final_gate_threshold),
                    "gflownet_candidates": int(candidate_breakdown["gflownet_candidates"]),
                    "gflownet_sample_batches_used": int(
                        candidate_breakdown["gflownet_sample_batches_used"]
                    ),
                    "candidate_pool_size": int(candidate_matrix.shape[0]),
                }
            elif final_batch > 0:
                selected_idx = select_acquisition_batch(
                    mean=mean,
                    std=std,
                    batch_size=final_batch,
                    candidate_states=candidate_matrix,
                    beta=final_beta,
                    diversity_weight=max(diversity_weight * 0.5, 0.05),
                )
                selected_states = candidate_matrix[selected_idx].tolist()
                queried_scores = (
                    oracle(np.asarray(selected_states, dtype=np.int64)).detach().cpu().numpy().tolist()
                )
                states.extend(selected_states)
                scores.extend(float(s) for s in queried_scores)
                final_selection_log = {
                    "skipped": False,
                    "oracle_queries_before": int(oracle.call_count - len(selected_states)),
                    "oracle_queries_after": int(oracle.call_count),
                    "selected_candidates": int(len(selected_states)),
                    "best_selected_score": float(max(queried_scores)) if queried_scores else 0.0,
                    "mean_selected_score": float(np.mean(queried_scores)) if queried_scores else 0.0,
                    "max_predicted_score": float(np.max(raw_mean)),
                    "gate_threshold": float(final_gate_threshold),
                    "gflownet_candidates": int(candidate_breakdown["gflownet_candidates"]),
                    "gflownet_sample_batches_used": int(
                        candidate_breakdown["gflownet_sample_batches_used"]
                    ),
                    "candidate_pool_size": int(candidate_matrix.shape[0]),
                }
                if logger is not None:
                    logger.log_metrics(
                        step=len(round_logs),
                        metrics={"final_selection_queries": int(len(selected_states))},
                    )

    # --------------------------------------------------------------- summary
    curve = build_query_curve(scores, optimum_score=optimum_score)
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
        "method": "hybrid_gfn_only",
        "seed": seed,
        **quality,
        "best_score": float(best_curve[-1]) if best_curve.size > 0 else float(quality["best_score"]),
        "curve": curve,
        "bootstrap_logs": bootstrap_logs,
        "bootstrap_rounds": int(bootstrap_rounds),
        "bootstrap_oracle_queries": int(bootstrap_oracle_queries),
        "round_logs": round_logs,
        "final_selection_log": final_selection_log,
        "oracle_queries": int(oracle.call_count),
        "real_oracle_queries": int(oracle.call_count),
        "fake_oracle_queries": int(fake_oracle_queries),
        "candidate_pool_queries": int(candidate_pool_queries),
        "cheap_model_queries": int(candidate_pool_queries + fake_oracle_queries),
        "unused_real_oracle_budget": int(oracle.remaining_budget),
        "reserved_final_oracle_budget": int(final_oracle_reserve),
        "acquisition": acquisition_name,
        "surrogate_type": getattr(surrogate, "surrogate_type", "unknown")
        if surrogate is not None
        else "unknown",
        "surrogate_fit_count": int(surrogate_fit_count),
        "gflownet_train_calls": int(gflownet_train_calls),
        "gflownet_total_train_steps": int(gflownet_total_train_steps),
        "gflownet_round_dirs": gflownet_round_dirs,
        "scores": [float(s) for s in scores],
        "runtime_seconds": float(time.perf_counter() - start_time),
        "optimum_score": optimum_score,
        "stopped_reason": last_gflownet_stop_reason,
        "sampling_strategy": initial_sampling_strategy,
        "candidate_sampling_strategy": "gflownet_only",
    }
    if optimum_info.get("optimum_words"):
        result["optimum_words"] = list(optimum_info["optimum_words"])
        result["optimum_word_count"] = int(optimum_info["optimum_word_count"])
        result["optimum_source"] = str(optimum_info["optimum_source"])

    if logger is not None:
        logger.dump_summary(result, filename="summary_hybrid_gfn_only.json")

    if gfn is not None and hasattr(gfn, "logger"):
        gfn.logger.end()

    return result
