"""Oracle-access GFlowNet training backed by the upstream gflownet repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.common import flatten_oracle_history
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, search_quality_metrics
from utils.upstream_gflownet import compose_upstream_train_config, ensure_gflownet_on_path


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _build_overrides(
    config: dict[str, Any],
    *,
    proxy_target: str,
    proxy_kwargs: dict[str, Any],
) -> list[str]:
    env_cfg = config["env"]
    gflownet_cfg = config["gflownet"]
    policy_cfg = dict(gflownet_cfg.get("policy", {}))
    evaluator_cfg = dict(gflownet_cfg.get("evaluator", {}))

    overrides = [
        "env=scrabble",
        "gflownet=trajectorybalance",
        "loss=trajectorybalance",
        "policy=mlp_trajectorybalance",
        "proxy=scrabble",
        "logger=base",
        "buffer=base",
        "evaluator=base",
        f"seed={config['seed']}",
        f"gflownet.seed={config['seed']}",
        f"device={config.get('device', 'cpu')}",
        f"float_precision={config.get('float_precision', 32)}",
        f"++env.max_length={int(env_cfg['max_length'])}",
        f"++proxy._target_={proxy_target}",
        "++logger.do.online=false",
        "++logger.lightweight=true",
        "++logger.progressbar.skip=true",
        "++logger.logdir.overwrite=true",
        f"++gflownet.random_action_prob={float(gflownet_cfg.get('random_action_prob', 0.0))}",
        f"++gflownet.optimizer.n_train_steps={int(gflownet_cfg.get('n_train_steps', 2000))}",
        f"++gflownet.optimizer.batch_size.forward={int(gflownet_cfg.get('batch_size_forward', 64))}",
        f"++gflownet.optimizer.lr={float(gflownet_cfg.get('lr', 1e-4))}",
        f"++gflownet.optimizer.z_dim={int(gflownet_cfg.get('z_dim', 16))}",
        f"++gflownet.optimizer.lr_z_mult={float(gflownet_cfg.get('lr_z_mult', 10.0))}",
        "++policy.forward.type=mlp",
        f"++policy.forward.n_hid={int(policy_cfg.get('hidden_dim', 256))}",
        f"++policy.forward.n_layers={int(policy_cfg.get('n_layers', 3))}",
        "++policy.backward.type=mlp",
        f"++policy.backward.n_hid={int(policy_cfg.get('hidden_dim', 256))}",
        f"++policy.backward.n_layers={int(policy_cfg.get('n_layers', 3))}",
        f"++policy.backward.shared_weights={_format_override_value(policy_cfg.get('shared_backward', False))}",
        f"++evaluator.period={int(evaluator_cfg.get('period', 250))}",
        f"++evaluator.n={int(evaluator_cfg.get('n', 128))}",
        f"++evaluator.checkpoints_period={int(evaluator_cfg.get('checkpoints_period', 250))}",
        "++buffer.train.type=null",
        "++buffer.test.type=null",
    ]

    for key, value in proxy_kwargs.items():
        overrides.append(f"++proxy.{key}={_format_override_value(value)}")
    return overrides


def sample_gflownet_terminating_states(gfn, n_samples: int) -> list[list[int]]:
    """Sample terminating states from a trained upstream GFlowNet."""
    if n_samples <= 0:
        return []
    batch, _ = gfn.sample_batch(n_forward=int(n_samples), train=False)
    states = batch.get_terminating_states(sort_by="trajectory")
    if torch.is_tensor(states):
        return states.detach().cpu().to(dtype=torch.long).tolist()
    return [list(state) for state in states]


def _train_upstream_gflownet(
    config: dict[str, Any],
    *,
    output_dir: Path,
    proxy_target: str,
    proxy_kwargs: dict[str, Any],
):
    """Train a GFlowNet using the upstream repo and return the in-memory agent."""
    ensure_gflownet_on_path(config.get("gflownet_root"))
    from gflownet.utils.common import gflownet_from_config

    overrides = _build_overrides(
        config,
        proxy_target=proxy_target,
        proxy_kwargs=proxy_kwargs,
    )
    upstream_cfg = compose_upstream_train_config(
        overrides,
        gflownet_root=config.get("gflownet_root"),
        output_dir=output_dir,
    )
    gfn = gflownet_from_config(upstream_cfg)
    stop_reason = None
    try:
        gfn.train()
    except RuntimeError as exc:
        if "Oracle query budget exceeded" not in str(exc):
            if hasattr(gfn, "logger"):
                gfn.logger.end()
            raise
        stop_reason = str(exc)
    return gfn, stop_reason


def run_oracle_gflownet(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Train a GFlowNet with direct oracle access and summarize its search quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(config["seed"]))

    oracle_cfg = config["oracle"]
    gflownet_cfg = config["gflownet"]

    gfn, stop_reason = _train_upstream_gflownet(
        config,
        output_dir=output_dir,
        proxy_target="proxies.oracle_proxy.OracleProxy",
        proxy_kwargs={
            "oracle_budget": int(oracle_cfg["budget"]),
            "enforce_budget": bool(oracle_cfg.get("enforce_budget", True)),
            "vocabulary_check": bool(oracle_cfg.get("vocabulary_check", False)),
            "reward_function": "identity",
            "reward_min": float(gflownet_cfg.get("reward_min", 0.0)),
            "do_clip_rewards": bool(gflownet_cfg.get("do_clip_rewards", False)),
            "stats_output_path": str(output_dir / "oracle_stats.json"),
        },
    )

    queried_states, queried_scores = flatten_oracle_history(gfn.proxy.call_history)
    curve = build_query_curve(
        queried_scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=queried_scores,
        states=queried_states if queried_states else None,
        oracle_queries=int(gfn.proxy.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    sampled_states = sample_gflownet_terminating_states(
        gfn,
        int(gflownet_cfg.get("evaluation_samples", 256)),
    )
    sampled_scores = []
    if sampled_states and getattr(gfn.proxy, "remaining_budget", float("inf")) > 0:
        sample_budget = int(
            min(len(sampled_states), max(int(gfn.proxy.remaining_budget), 0))
        )
        if sample_budget > 0:
            sampled_subset = np.asarray(sampled_states[:sample_budget], dtype=np.int64)
            sampled_scores = (
                gfn.proxy(sampled_subset).detach().cpu().numpy().astype(float).tolist()
            )

    result = {
        "method": "gflownet_oracle",
        "seed": int(config["seed"]),
        **quality,
        "curve": curve,
        "best_score": float(max(queried_scores)) if queried_scores else float(quality["best_score"]),
        "oracle_queries": int(gfn.proxy.call_count),
        "checkpoint_dir": str(output_dir / "ckpts"),
        "sampled_states": sampled_states,
        "sampled_scores": sampled_scores,
        "stopped_reason": stop_reason,
        "scores": [float(score) for score in queried_scores],
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_gflownet.json")

    if hasattr(gfn, "logger"):
        gfn.logger.end()
    return result
