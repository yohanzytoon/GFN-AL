"""Hybrid GFlowNet + Active Learning training loop."""

from __future__ import annotations

import copy
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
    propose_local_search_candidates,
)
from training.dataset import (
    deduplicate_state_scores,
    sample_mutated_states,
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


def _sample_hybrid_candidates(
    *,
    env: ScrabbleOracleEnv,
    gfn,
    seen_states: list[list[int]],
    gflownet_sample_size: int,
    random_pool_size: int,
    fallback_random_pool_size: int,
    min_required: int,
    min_length: int,
    candidate_unique: bool,
    seed: int,
    sampling_strategy: str,
    fallback_sampling_strategy: str,
    local_search_candidates: list[list[int]],
    mutation_anchor_states: list[list[int]],
    mutation_pool_size: int,
    mutation_edits: int,
    mutation_sampling_strategy: str,
    gflownet_root: str | None = None,
) -> tuple[list[list[int]], dict[str, int]]:
    """Sample a mixed GFlowNet + random candidate pool."""
    gflownet_candidates = (
        sample_gflownet_terminating_states(gfn, gflownet_sample_size)
        if gfn is not None and gflownet_sample_size > 0
        else []
    )
    random_candidates = (
        sample_terminating_states(
            env,
            random_pool_size,
            sampling_strategy=sampling_strategy,
            min_length=min_length,
            unique=candidate_unique,
            seed=seed,
            gflownet_root=gflownet_root,
        )
        if random_pool_size > 0
        else []
    )
    mutation_candidates = (
        sample_mutated_states(
            env,
            mutation_anchor_states,
            mutation_pool_size,
            sampling_strategy=mutation_sampling_strategy,
            min_length=min_length,
            unique=candidate_unique,
            seed=seed + 17,
            max_mutations=mutation_edits,
            gflownet_root=gflownet_root,
        )
        if mutation_pool_size > 0 and len(mutation_anchor_states) > 0
        else []
    )

    filtered_candidates = filter_new_states(
        gflownet_candidates + random_candidates + mutation_candidates + local_search_candidates,
        seen_states,
    )
    if len(filtered_candidates) < min_required and fallback_random_pool_size > 0:
        fallback_candidates = sample_terminating_states(
            env,
            fallback_random_pool_size,
            sampling_strategy=fallback_sampling_strategy,
            min_length=min_length,
            unique=False,
            seed=seed + 1,
            gflownet_root=gflownet_root,
        )
        filtered_candidates = filter_new_states(
            filtered_candidates + fallback_candidates,
            seen_states,
        )

    return filtered_candidates, {
        "gflownet_candidates": int(len(gflownet_candidates)),
        "random_candidates": int(len(random_candidates)),
        "local_search_candidates": int(len(local_search_candidates)),
        "mutation_candidates": int(len(mutation_candidates)),
    }


def _augment_local_search_anchors_with_gflownet(
    *,
    gfn,
    surrogate,
    base_anchor_states: list[list[int]],
    seen_states: list[list[int]],
    pool_size: int,
    top_k: int,
    beta: float,
    diversity_weight: float,
    max_length: int,
    gflownet_root: str | None,
    plausibility_weight: float,
) -> tuple[list[list[int]], int]:
    """Use the current GFlowNet to seed local search around promising unseen states."""
    if gfn is None or pool_size <= 0 or top_k <= 0:
        return base_anchor_states, 0

    sampled_states = sample_gflownet_terminating_states(gfn, int(pool_size))
    filtered_states = filter_new_states(sampled_states, seen_states)
    if len(filtered_states) == 0:
        return base_anchor_states, 0

    candidate_matrix = np.asarray(filtered_states, dtype=np.int64)
    mean, std = surrogate.predict(candidate_matrix, return_std=True)
    mean = mean + compute_plausibility_bonus(
        candidate_matrix,
        int(max_length),
        gflownet_root=gflownet_root,
        weight=float(plausibility_weight),
    )
    selected_idx = select_acquisition_batch(
        mean=mean,
        std=std,
        batch_size=min(int(top_k), int(candidate_matrix.shape[0])),
        candidate_states=candidate_matrix,
        beta=float(beta),
        diversity_weight=max(float(diversity_weight) * 0.5, 0.0),
    )
    selected_states = candidate_matrix[selected_idx].tolist()
    return filter_new_states(base_anchor_states + selected_states, []), int(candidate_matrix.shape[0])


def run_hybrid_gflownet_active(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the hybrid pipeline: fake-oracle GFlowNet -> acquisition -> real oracle."""
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

    initial_size = int(min(hybrid_cfg.get("initial_size", 64), oracle_budget))
    batch_size = int(hybrid_cfg.get("batch_size", 16))
    candidate_pool_size = int(hybrid_cfg.get("candidate_pool_size", 256))
    fallback_random_pool_size = int(
        hybrid_cfg.get("fallback_random_pool_size", candidate_pool_size)
    )
    max_rounds = int(hybrid_cfg.get("max_rounds", 50))
    sampling_strategy = str(hybrid_cfg.get("sampling_strategy", "frequency"))
    initial_sampling_strategy = str(
        hybrid_cfg.get("initial_sampling_strategy", sampling_strategy)
    )
    candidate_sampling_strategy = str(
        hybrid_cfg.get("candidate_sampling_strategy", sampling_strategy)
    )
    fallback_sampling_strategy = str(
        hybrid_cfg.get("fallback_sampling_strategy", "frequency")
    )
    min_length = int(hybrid_cfg.get("min_length", 3))
    candidate_unique = bool(hybrid_cfg.get("candidate_unique", True))
    always_include_random_pool = bool(hybrid_cfg.get("always_include_random_pool", True))
    random_pool_size = int(
        hybrid_cfg.get(
            "random_pool_size",
            fallback_random_pool_size if always_include_random_pool else 0,
        )
    )
    mutation_pool_size = int(hybrid_cfg.get("mutation_pool_size", 0))
    mutation_top_k = int(hybrid_cfg.get("mutation_top_k", 16))
    mutation_edits = int(hybrid_cfg.get("mutation_edits", 2))
    mutation_sampling_strategy = str(
        hybrid_cfg.get("mutation_sampling_strategy", candidate_sampling_strategy)
    )
    local_search_pool_size = int(hybrid_cfg.get("local_search_pool_size", 0))
    local_search_beam_width = int(hybrid_cfg.get("local_search_beam_width", 12))
    local_search_steps = int(hybrid_cfg.get("local_search_steps", 3))
    local_search_neighbors_per_step = int(
        hybrid_cfg.get("local_search_neighbors_per_step", 64)
    )
    local_search_sampling_strategy = str(
        hybrid_cfg.get("local_search_sampling_strategy", candidate_sampling_strategy)
    )
    local_search_mutation_edits = int(
        hybrid_cfg.get("local_search_mutation_edits", 1)
    )
    final_candidate_pool_size = int(
        hybrid_cfg.get(
            "final_candidate_pool_size",
            max(candidate_pool_size * 4, batch_size * 8),
        )
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
    gflownet_sample_size = int(gflownet_schedule_cfg.get("sample_size", candidate_pool_size))
    refresh_before_final_selection = bool(
        gflownet_schedule_cfg.get("refresh_before_final_selection", True)
    )
    warm_start = bool(gflownet_schedule_cfg.get("warm_start", True))
    gflownet_steps_per_refresh = int(gflownet_schedule_cfg.get("n_train_steps", 1000))
    gflownet_start_round = int(gflownet_schedule_cfg.get("start_round", 2))
    gflownet_min_positive_count = int(
        gflownet_schedule_cfg.get("min_positive_count", max(batch_size // 2, 8))
    )
    num_tokens = int(env_cfg.get("num_tokens", 27))
    diversity_weight = float(hybrid_cfg.get("diversity_weight", 0.3))
    plausibility_bonus_weight = float(hybrid_cfg.get("plausibility_bonus_weight", 2.0))
    gflownet_root = config.get("gflownet_root")
    gflownet_local_anchor_pool_size = int(
        gflownet_schedule_cfg.get("local_search_anchor_pool_size", 0)
    )
    gflownet_local_anchor_top_k = int(
        gflownet_schedule_cfg.get("local_search_anchor_top_k", 0)
    )

    states = sample_terminating_states(
        env,
        initial_size,
        sampling_strategy=initial_sampling_strategy,
        min_length=min_length,
        unique=True,
        seed=seed,
        gflownet_root=gflownet_root,
    )
    scores = oracle(env.states2proxy(states)).detach().cpu().numpy().astype(float).tolist()

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
    proposal_surrogate_queries = 0

    for round_idx in range(max_rounds):
        remaining_for_rounds = max(0, oracle.remaining_budget - final_oracle_reserve)
        if remaining_for_rounds <= 0:
            break

        # Linearly decay beta from beta (explore) to beta_min (exploit) over rounds.
        round_beta = beta - (beta - beta_min) * round_idx / max(max_rounds - 1, 1)

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
            round_idx >= gflownet_start_round
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
            gfn, stop_reason = _train_upstream_gflownet(
                gflownet_config,
                output_dir=round_gflownet_dir,
                proxy_target="proxies.surrogate_proxy.SurrogateProxy",
                proxy_kwargs={
                    "surrogate_path": str(surrogate_path),
                    "prediction_mode": str(
                        gflownet_schedule_cfg.get("prediction_mode", "mean")
                    ),
                    "exploration_beta": float(
                        gflownet_schedule_cfg.get("exploration_beta", 1.0)
                    ),
                    "reward_transform": str(
                        gflownet_schedule_cfg.get("reward_transform", "softplus")
                    ),
                    "reward_function": "identity",
                    "reward_min": float(gflownet_schedule_cfg.get("reward_min", 1e-4)),
                    "do_clip_rewards": bool(
                        gflownet_schedule_cfg.get("do_clip_rewards", True)
                    ),
                    "max_length": int(env_cfg["max_length"]),
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

        ranked_indices = np.argsort(np.asarray(scores, dtype=float))[::-1]
        mutation_anchor_states = [
            states[idx]
            for idx in ranked_indices[: max(mutation_top_k, 0)]
        ]
        local_search_anchor_states, gflownet_anchor_surrogate_queries = _augment_local_search_anchors_with_gflownet(
            gfn=gfn,
            surrogate=surrogate,
            base_anchor_states=mutation_anchor_states,
            seen_states=states,
            pool_size=gflownet_local_anchor_pool_size,
            top_k=gflownet_local_anchor_top_k,
            beta=round_beta,
            diversity_weight=diversity_weight,
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            plausibility_weight=plausibility_bonus_weight,
        )
        proposal_surrogate_queries += int(gflownet_anchor_surrogate_queries)
        local_search_candidates, local_search_stats = propose_local_search_candidates(
            env=env,
            surrogate=surrogate,
            anchor_states=local_search_anchor_states,
            proposal_size=local_search_pool_size,
            beam_width=local_search_beam_width,
            n_steps=local_search_steps,
            neighbors_per_step=local_search_neighbors_per_step,
            sampling_strategy=local_search_sampling_strategy,
            min_length=min_length,
            candidate_unique=candidate_unique,
            seen_states=states,
            seed=seed + 60_000 + round_idx,
            beta=round_beta,
            mutation_edits=local_search_mutation_edits,
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            plausibility_weight=plausibility_bonus_weight,
        )
        proposal_surrogate_queries += int(local_search_stats["surrogate_queries"])

        sampled_candidates, candidate_breakdown = _sample_hybrid_candidates(
            env=env,
            gfn=gfn,
            seen_states=states,
            gflownet_sample_size=gflownet_sample_size,
            random_pool_size=random_pool_size,
            fallback_random_pool_size=fallback_random_pool_size,
            min_required=batch_size,
            min_length=min_length,
            candidate_unique=candidate_unique,
            seed=seed + 50_000 + (2 * round_idx),
            sampling_strategy=candidate_sampling_strategy,
            fallback_sampling_strategy=fallback_sampling_strategy,
            local_search_candidates=local_search_candidates,
            mutation_anchor_states=mutation_anchor_states,
            mutation_pool_size=mutation_pool_size,
            mutation_edits=mutation_edits,
            mutation_sampling_strategy=mutation_sampling_strategy,
            gflownet_root=gflownet_root,
        )
        candidate_matrix = np.asarray(sampled_candidates, dtype=np.int64)
        if candidate_matrix.size == 0:
            break

        candidate_pool_queries += int(candidate_matrix.shape[0])
        mean, std = surrogate.predict(candidate_matrix, return_std=True)

        # Plausibility prior: bias toward word-like candidates.
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
        scores.extend(float(score) for score in queried_scores)

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
            "positive_count": int(positive_count),
            "gflownet_ready": bool(gflownet_ready),
            "gflownet_candidates": int(candidate_breakdown["gflownet_candidates"]),
            "random_candidates": int(candidate_breakdown["random_candidates"]),
            "local_search_candidates": int(candidate_breakdown["local_search_candidates"]),
            "mutation_candidates": int(candidate_breakdown["mutation_candidates"]),
            "unique_candidates_after_filter": int(candidate_matrix.shape[0]),
            "reserved_final_oracle_budget": int(final_oracle_reserve),
            "remaining_oracle_budget": int(oracle.remaining_budget),
            "fake_oracle_queries_total": int(fake_oracle_queries),
            "candidate_pool_queries_total": int(candidate_pool_queries),
            "local_search_surrogate_queries_total": int(proposal_surrogate_queries),
            "gflownet_train_calls": int(gflownet_train_calls),
            "gflownet_total_train_steps": int(gflownet_total_train_steps),
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

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
            gfn, stop_reason = _train_upstream_gflownet(
                gflownet_config,
                output_dir=final_gflownet_dir,
                proxy_target="proxies.surrogate_proxy.SurrogateProxy",
                proxy_kwargs={
                    "surrogate_path": str(final_surrogate_path),
                    "prediction_mode": "mean",
                    "exploration_beta": 0.0,
                    "reward_transform": str(
                        gflownet_schedule_cfg.get("reward_transform", "softplus")
                    ),
                    "reward_function": "identity",
                    "reward_min": float(gflownet_schedule_cfg.get("reward_min", 1e-4)),
                    "do_clip_rewards": bool(
                        gflownet_schedule_cfg.get("do_clip_rewards", True)
                    ),
                    "max_length": int(env_cfg["max_length"]),
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

        final_random_pool_size = int(
            hybrid_cfg.get("final_random_pool_size", max(random_pool_size, batch_size))
        )
        final_mutation_pool_size = int(
            hybrid_cfg.get(
                "final_mutation_pool_size",
                max(mutation_pool_size, batch_size * 4),
            )
        )
        final_gflownet_sample_size = int(
            gflownet_schedule_cfg.get(
                "final_gflownet_sample_size",
                hybrid_cfg.get(
                    "final_gflownet_sample_size",
                    max(final_candidate_pool_size, batch_size * 4),
                ),
            )
        )
        ranked_indices = np.argsort(np.asarray(scores, dtype=float))[::-1]
        mutation_anchor_states = [
            states[idx]
            for idx in ranked_indices[: max(mutation_top_k * 2, mutation_top_k)]
        ]
        final_local_search_anchor_states, gflownet_anchor_surrogate_queries = _augment_local_search_anchors_with_gflownet(
            gfn=gfn,
            surrogate=surrogate,
            base_anchor_states=mutation_anchor_states,
            seen_states=states,
            pool_size=max(gflownet_local_anchor_pool_size, batch_size * 8),
            top_k=max(gflownet_local_anchor_top_k, batch_size),
            beta=final_beta,
            diversity_weight=diversity_weight,
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            plausibility_weight=plausibility_bonus_weight,
        )
        proposal_surrogate_queries += int(gflownet_anchor_surrogate_queries)
        final_local_search_pool_size = int(
            hybrid_cfg.get(
                "final_local_search_pool_size",
                max(local_search_pool_size, batch_size * 8),
            )
        )
        local_search_candidates, local_search_stats = propose_local_search_candidates(
            env=env,
            surrogate=surrogate,
            anchor_states=final_local_search_anchor_states,
            proposal_size=final_local_search_pool_size,
            beam_width=max(local_search_beam_width, 16),
            n_steps=max(local_search_steps, 4),
            neighbors_per_step=max(local_search_neighbors_per_step, 64),
            sampling_strategy=local_search_sampling_strategy,
            min_length=min_length,
            candidate_unique=candidate_unique,
            seen_states=states,
            seed=seed + 950_000,
            beta=final_beta,
            mutation_edits=max(local_search_mutation_edits, 1),
            max_length=int(env_cfg["max_length"]),
            gflownet_root=gflownet_root,
            plausibility_weight=plausibility_bonus_weight,
        )
        proposal_surrogate_queries += int(local_search_stats["surrogate_queries"])
        final_candidates, candidate_breakdown = _sample_hybrid_candidates(
            env=env,
            gfn=gfn,
            seen_states=states,
            gflownet_sample_size=final_gflownet_sample_size,
            random_pool_size=final_random_pool_size,
            fallback_random_pool_size=max(fallback_random_pool_size, batch_size),
            min_required=int(min(oracle.remaining_budget, final_candidate_pool_size)),
            min_length=min_length,
            candidate_unique=candidate_unique,
            seed=seed + 900_000,
            sampling_strategy=candidate_sampling_strategy,
            fallback_sampling_strategy=fallback_sampling_strategy,
            local_search_candidates=local_search_candidates,
            mutation_anchor_states=mutation_anchor_states,
            mutation_pool_size=final_mutation_pool_size,
            mutation_edits=max(mutation_edits, 2),
            mutation_sampling_strategy=mutation_sampling_strategy,
            gflownet_root=gflownet_root,
        )
        candidate_matrix = np.asarray(final_candidates, dtype=np.int64)
        if candidate_matrix.size > 0:
            candidate_pool_queries += int(candidate_matrix.shape[0])
            mean, std = surrogate.predict(candidate_matrix, return_std=True)

            # Plausibility prior for final selection too.
            mean = mean + compute_plausibility_bonus(
                candidate_matrix,
                int(env_cfg["max_length"]),
                gflownet_root=gflownet_root,
                weight=plausibility_bonus_weight,
            )

            final_batch = int(min(oracle.remaining_budget, candidate_matrix.shape[0]))
            # Final selection: lower diversity weight to favour pure exploitation.
            selected_idx = select_acquisition_batch(
                mean=mean,
                std=std,
                batch_size=final_batch,
                candidate_states=candidate_matrix,
                beta=final_beta,
                diversity_weight=max(diversity_weight * 0.5, 0.1),
            )
            selected_states = candidate_matrix[selected_idx].tolist()
            queried_scores = (
                oracle(np.asarray(selected_states, dtype=np.int64)).detach().cpu().numpy().tolist()
            )
            states.extend(selected_states)
            scores.extend(float(score) for score in queried_scores)
            final_selection_log = {
                "oracle_queries_before": int(oracle.call_count - len(selected_states)),
                "oracle_queries_after": int(oracle.call_count),
                "selected_candidates": int(len(selected_states)),
                "best_selected_score": float(max(queried_scores)) if queried_scores else 0.0,
                "mean_selected_score": float(np.mean(queried_scores)) if queried_scores else 0.0,
                "gflownet_candidates": int(candidate_breakdown["gflownet_candidates"]),
                "random_candidates": int(candidate_breakdown["random_candidates"]),
                "local_search_candidates": int(candidate_breakdown["local_search_candidates"]),
                "mutation_candidates": int(candidate_breakdown["mutation_candidates"]),
                "candidate_pool_size": int(candidate_matrix.shape[0]),
            }
            if logger is not None:
                logger.log_metrics(
                    step=len(round_logs),
                    metrics={"final_selection_queries": int(len(selected_states))},
                )

    curve = build_query_curve(
        scores,
        optimum_score=optimum_score,
    )
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
        "method": "hybrid_gflownet_active",
        "seed": seed,
        **quality,
        "best_score": float(best_curve[-1]) if best_curve.size > 0 else float(quality["best_score"]),
        "curve": curve,
        "round_logs": round_logs,
        "final_selection_log": final_selection_log,
        "oracle_queries": int(oracle.call_count),
        "real_oracle_queries": int(oracle.call_count),
        "fake_oracle_queries": int(fake_oracle_queries),
        "candidate_pool_queries": int(candidate_pool_queries),
        "proposal_surrogate_queries": int(proposal_surrogate_queries),
        "cheap_model_queries": int(candidate_pool_queries + proposal_surrogate_queries + fake_oracle_queries),
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
        "scores": [float(score) for score in scores],
        "runtime_seconds": float(time.perf_counter() - start_time),
        "optimum_score": optimum_score,
        "stopped_reason": last_gflownet_stop_reason,
        "sampling_strategy": initial_sampling_strategy,
        "candidate_sampling_strategy": candidate_sampling_strategy,
        "mutation_sampling_strategy": mutation_sampling_strategy,
        "local_search_sampling_strategy": local_search_sampling_strategy,
    }
    if optimum_info.get("optimum_words"):
        result["optimum_words"] = list(optimum_info["optimum_words"])
        result["optimum_word_count"] = int(optimum_info["optimum_word_count"])
        result["optimum_source"] = str(optimum_info["optimum_source"])

    if logger is not None:
        logger.dump_summary(result, filename="summary_hybrid.json")

    if gfn is not None and hasattr(gfn, "logger"):
        gfn.logger.end()
    return result
