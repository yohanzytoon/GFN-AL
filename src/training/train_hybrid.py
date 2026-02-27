"""GFlowNet-integrated training loops for pure GFlowNet and hybrid AL+GFN."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
from utils.metrics import build_query_curve, regression_metrics, search_quality_metrics

OBJECTIVE_MAP = {
    "trajectorybalance": {
        "gflownet": "trajectorybalance",
        "loss": "trajectorybalance",
        "policy": "mlp_trajectorybalance",
    },
    "flowmatch": {
        "gflownet": "flowmatch",
        "loss": "flowmatching",
        "policy": "mlp_flowmatch",
    },
    "detailedbalance": {
        "gflownet": "detailedbalance",
        "loss": "detailedbalance",
        "policy": "mlp_detailedbalance",
    },
}


def _bool_str(value: bool) -> str:
    """Hydra-safe lowercase boolean override value."""
    return str(bool(value)).lower()


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
        return select_thompson(mean=mean, std=std, batch_size=batch_size, random_state=rng)
    if acquisition == "uncertainty":
        return np.argsort(std)[::-1][:batch_size]
    raise KeyError(f"Unknown acquisition function: {acquisition}")


def _pythonpath_env(project_src: Path, repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    parts = [str(project_src), str(repo_root)]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _resolve_repo_root(project_root: Path, repo_hint: str | None) -> Path:
    candidates = []
    if repo_hint:
        hint = Path(repo_hint)
        if hint.is_absolute():
            candidates.append(hint)
        else:
            candidates.append((project_root / hint).resolve())
    candidates.extend(
        [
            (project_root / "../gflownet").resolve(),
            (project_root / "gflownet").resolve(),
            (project_root.parent / "gflownet").resolve(),
        ]
    )
    for candidate in candidates:
        if (candidate / "train.py").exists() and (candidate / "gflownet").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not resolve GFlowNet repo root containing train.py. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def run_gflownet_train(
    repo_root: Path,
    project_src: Path,
    run_dir: Path,
    *,
    seed: int,
    device: str,
    max_length: int,
    objective: str,
    n_train_steps: int,
    batch_size_forward: int,
    temperature_logits: float,
    random_action_prob: float,
    proxy_backend: str,
    oracle_budget: int,
    enforce_budget: bool,
    vocabulary_check: bool,
    surrogate_checkpoint: str | None,
    stats_output_path: Path | None,
    python_bin: str = sys.executable,
) -> Path:
    """Launch official gflownet/train.py with Hydra overrides."""
    objective_key = objective.lower()
    if objective_key not in OBJECTIVE_MAP:
        raise KeyError(f"Unknown objective '{objective}'.")

    run_dir.mkdir(parents=True, exist_ok=True)
    objective_cfg = OBJECTIVE_MAP[objective_key]

    cmd = [
        python_bin,
        str(repo_root / "train.py"),
        "env=scrabble_oracle",
        "proxy=oracle_proxy",
        f"gflownet={objective_cfg['gflownet']}",
        f"loss={objective_cfg['loss']}",
        f"policy={objective_cfg['policy']}",
        f"env.max_length={max_length}",
        f"env.oracle_budget={oracle_budget}",
        f"proxy.backend={proxy_backend}",
        f"proxy.oracle_budget={oracle_budget}",
        f"proxy.enforce_budget={_bool_str(enforce_budget)}",
        f"proxy.vocabulary_check={_bool_str(vocabulary_check)}",
        "proxy.reward_function=identity",
        "proxy.reward_function_kwargs={}",
        f"gflownet.optimizer.n_train_steps={n_train_steps}",
        f"gflownet.optimizer.batch_size.forward={batch_size_forward}",
        "gflownet.optimizer.batch_size.backward_dataset=0",
        "gflownet.optimizer.batch_size.backward_replay=0",
        f"gflownet.temperature_logits={temperature_logits}",
        f"gflownet.random_action_prob={random_action_prob}",
        "logger.do.online=False",
        f"hydra.run.dir={run_dir}",
        f"seed={seed}",
        f"device={device}",
        "n_samples=0",
    ]

    if surrogate_checkpoint is not None:
        cmd.append(f"proxy.surrogate_checkpoint={surrogate_checkpoint}")
    if stats_output_path is not None:
        cmd.append(f"proxy.stats_output_path={stats_output_path}")

    subprocess.run(
        cmd,
        check=True,
        cwd=repo_root,
        env=_pythonpath_env(project_src=project_src, repo_root=repo_root),
    )
    return run_dir


def run_gflownet_eval(
    repo_root: Path,
    project_src: Path,
    rundir: Path,
    n_samples: int,
    sampling_batch_size: int,
    device: str,
    samples_only: bool = True,
    python_bin: str = sys.executable,
) -> None:
    """Launch official gflownet/eval.py and keep temporary files by default."""
    cmd = [
        python_bin,
        str(repo_root / "eval.py"),
        f"rundir={rundir}",
        f"n_samples={n_samples}",
        f"sampling_batch_size={sampling_batch_size}",
        f"device={device}",
        f"samples_only={_bool_str(samples_only)}",
        "print_config=False",
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=repo_root,
        env=_pythonpath_env(project_src=project_src, repo_root=repo_root),
        input="n\n",
        text=True,
    )


def run_gflownet_resume(
    repo_root: Path,
    project_src: Path,
    rundir: Path,
    device: str,
    seed: int,
    python_bin: str = sys.executable,
) -> Path:
    """Launch official gflownet/resume.py."""
    cmd = [
        python_bin,
        str(repo_root / "resume.py"),
        f"rundir={rundir}",
        f"device={device}",
        f"seed={seed}",
        "no_wandb=True",
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=repo_root,
        env=_pythonpath_env(project_src=project_src, repo_root=repo_root),
    )
    return rundir


def sample_gflownet_candidates(
    repo_root: Path,
    project_src: Path,
    rundir: Path,
    n_samples: int,
    device: str,
) -> np.ndarray:
    """Sample terminating states from a trained GFlowNet run directory."""
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gflownet.utils.common import load_gflownet_from_rundir  # pylint: disable=import-error

    gfn, _ = load_gflownet_from_rundir(
        rundir=rundir,
        no_wandb=True,
        print_config=False,
        device=device,
        load_last_checkpoint=True,
    )
    batch, _ = gfn.sample_batch(n_forward=n_samples, train=False)
    states_proxy = batch.get_terminating_states(proxy=True)
    states = np.asarray(states_proxy.detach().cpu().numpy(), dtype=np.int64)
    gfn.logger.end()
    return states


def _read_proxy_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pure_gflownet(
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run pure GFlowNet with official train/eval workflow and oracle rewards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    gfn_cfg = config["gflownet"]

    repo_root = _resolve_repo_root(project_root, gfn_cfg.get("repo_root"))
    project_src = project_root / "src"

    batch_size_forward = int(gfn_cfg.get("batch_size_forward", 16))
    budget = int(oracle_cfg["budget"])
    max_steps_by_budget = max(1, budget // max(batch_size_forward, 1))
    train_steps = min(int(gfn_cfg.get("train_steps", 500)), max_steps_by_budget)

    run_dir = output_dir / "pure_gflownet_run"
    stats_path = output_dir / "pure_gflownet_proxy_stats.json"

    run_gflownet_train(
        repo_root=repo_root,
        project_src=project_src,
        run_dir=run_dir,
        seed=seed,
        device=device,
        max_length=int(env_cfg["max_length"]),
        objective=str(gfn_cfg.get("objective", "trajectorybalance")),
        n_train_steps=train_steps,
        batch_size_forward=batch_size_forward,
        temperature_logits=float(gfn_cfg.get("temperature_logits", 1.0)),
        random_action_prob=float(gfn_cfg.get("random_action_prob", 0.0)),
        proxy_backend="oracle",
        oracle_budget=budget,
        enforce_budget=True,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        surrogate_checkpoint=None,
        stats_output_path=stats_path,
        python_bin=str(gfn_cfg.get("python_bin", sys.executable)),
    )

    proxy_stats = _read_proxy_stats(stats_path)
    eval_samples = int(gfn_cfg.get("eval_samples", 0))
    remaining_budget = int(proxy_stats.get("remaining_budget", 0) or 0)
    eval_n = min(eval_samples, max(remaining_budget, 0))
    run_gflownet_eval(
        repo_root=repo_root,
        project_src=project_src,
        rundir=run_dir,
        n_samples=eval_n,
        sampling_batch_size=int(gfn_cfg.get("sampling_batch_size", 128)),
        device=device,
        samples_only=True,
        python_bin=str(gfn_cfg.get("python_bin", sys.executable)),
    )

    score_trace = [
        float(score)
        for batch in proxy_stats.get("call_history", [])
        for score in batch.get("scores", [])
    ]
    sampled_states = sample_gflownet_candidates(
        repo_root=repo_root,
        project_src=project_src,
        rundir=run_dir,
        n_samples=min(int(gfn_cfg.get("diversity_samples", 256)), max(budget, 1)),
        device=device,
    )

    if not score_trace:
        env = ScrabbleOracleEnv(max_length=int(env_cfg["max_length"]), device=device)
        scorer = OracleProxy(
            device=device,
            backend="oracle",
            oracle_budget=None,
            enforce_budget=False,
            vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        )
        scorer.setup(env)
        score_trace = scorer(sampled_states).detach().cpu().numpy().astype(float).tolist()

    # Compute diversity-focused metrics on a fixed sample from the learned policy.
    env = ScrabbleOracleEnv(max_length=int(env_cfg["max_length"]), device=device)
    scorer = OracleProxy(
        device=device,
        backend="oracle",
        oracle_budget=None,
        enforce_budget=False,
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
    )
    scorer.setup(env)
    sampled_scores = scorer(sampled_states).detach().cpu().numpy().astype(float).tolist()

    curve = build_query_curve(
        score_trace,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality_curve = search_quality_metrics(
        scores=score_trace,
        states=None,
        oracle_queries=int(proxy_stats.get("call_count", len(score_trace))),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )
    quality_diversity = search_quality_metrics(
        scores=sampled_scores,
        states=sampled_states.tolist(),
        oracle_queries=int(proxy_stats.get("call_count", len(score_trace))),
        optimum_score=None,
        top_k=10,
        pad_value=0,
    )

    result = {
        "method": "pure_gflownet",
        "seed": seed,
        "run_dir": str(run_dir),
        **quality_curve,
        "diversity_entropy": float(quality_diversity["diversity_entropy"]),
        "unique_fraction": float(quality_diversity["unique_fraction"]),
        "mode_coverage": int(quality_diversity["mode_coverage"]),
        "topk_edit_distance": float(quality_diversity["topk_edit_distance"]),
        "curve": curve,
        "proxy_stats": proxy_stats,
        "scores": [float(s) for s in score_trace],
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_pure_gflownet.json")

    return result


def run_hybrid_learning(
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the hybrid loop: surrogate training + GFlowNet proposal + oracle filtering."""
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
    gfn_cfg = config["gflownet"]
    hybrid_cfg = config["hybrid"]

    repo_root = _resolve_repo_root(project_root, gfn_cfg.get("repo_root"))
    project_src = project_root / "src"

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

    acquisition = str(active_cfg.get("acquisition", "ucb"))
    batch_size = int(active_cfg.get("batch_size", 16))
    candidate_pool_size = int(hybrid_cfg.get("gfn_candidate_pool_size", 256))
    max_rounds = int(hybrid_cfg.get("max_rounds", 100))
    retrain_frequency = int(hybrid_cfg.get("retrain_frequency", 1))

    surrogate_type = str(active_cfg.get("surrogate_type", "gp"))
    surrogate_kwargs = dict(active_cfg.get("surrogate", {}))
    surrogate_kwargs.setdefault("max_length", int(env_cfg["max_length"]))
    surrogate_kwargs.setdefault("device", device)

    initial_size = int(min(active_cfg.get("initial_size", 50), oracle_cfg["budget"]))
    states = env.get_random_terminating_states(
        n_states=initial_size,
        unique=False,
        max_attempts=max(5 * initial_size, 1000),
    )
    scores = oracle(env.states2proxy(states)).detach().cpu().numpy().astype(float).tolist()

    round_logs: list[dict[str, Any]] = []
    gfn_run_dir: Path | None = None

    for round_idx in range(max_rounds):
        if oracle.remaining_budget <= 0:
            break

        surrogate = build_surrogate(surrogate_type=surrogate_type, **surrogate_kwargs)
        train_states = np.asarray(states, dtype=np.int64)
        train_scores = np.asarray(scores, dtype=np.float32)
        surrogate.fit(train_states, train_scores)

        surrogate_ckpt = output_dir / "surrogates" / f"surrogate_round_{round_idx:03d}.pt"
        surrogate.save(surrogate_ckpt)

        if gfn_run_dir is None or round_idx % retrain_frequency == 0:
            gfn_run_dir = output_dir / "gflownet_runs" / f"round_{round_idx:03d}"
            run_gflownet_train(
                repo_root=repo_root,
                project_src=project_src,
                run_dir=gfn_run_dir,
                seed=seed + round_idx,
                device=device,
                max_length=int(env_cfg["max_length"]),
                objective=str(gfn_cfg.get("objective", "trajectorybalance")),
                n_train_steps=int(gfn_cfg.get("train_steps", 200)),
                batch_size_forward=int(gfn_cfg.get("batch_size_forward", 16)),
                temperature_logits=float(gfn_cfg.get("temperature_logits", 1.0)),
                random_action_prob=float(gfn_cfg.get("random_action_prob", 0.0)),
                proxy_backend="surrogate",
                oracle_budget=0,
                enforce_budget=False,
                vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
                surrogate_checkpoint=str(surrogate_ckpt),
                stats_output_path=None,
                python_bin=str(gfn_cfg.get("python_bin", sys.executable)),
            )

        assert gfn_run_dir is not None
        candidates = sample_gflownet_candidates(
            repo_root=repo_root,
            project_src=project_src,
            rundir=gfn_run_dir,
            n_samples=candidate_pool_size,
            device=device,
        )

        mean, std = surrogate.predict(candidates, return_std=True)
        this_batch = int(min(batch_size, oracle.remaining_budget, candidates.shape[0]))
        if this_batch <= 0:
            break

        selected_idx = _select_batch(
            acquisition=acquisition,
            mean=mean,
            std=std,
            batch_size=this_batch,
            best_observed=float(np.max(train_scores)),
            beta=float(active_cfg.get("acquisition_beta", 2.0)),
            xi=float(active_cfg.get("ei_xi", 0.01)),
            rng=rng,
        )

        selected = candidates[selected_idx]
        queried_scores = oracle(selected).detach().cpu().numpy().astype(float)

        states.extend(selected.tolist())
        scores.extend(queried_scores.tolist())

        pred_all = surrogate.predict(np.asarray(states, dtype=np.int64), return_std=False)
        surrogate_fit_metrics = regression_metrics(np.asarray(scores, dtype=float), pred_all)
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
            "surrogate_rmse": float(surrogate_fit_metrics["rmse"]),
            "gfn_run_dir": str(gfn_run_dir),
        }
        round_logs.append(round_payload)
        if logger is not None:
            logger.log_metrics(step=round_idx, metrics=round_payload)

    curve = build_query_curve(
        scores,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=scores,
        states=states,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )
    result = {
        "method": "hybrid",
        "seed": seed,
        **quality,
        "curve": curve,
        "round_logs": round_logs,
        "final_gflownet_run_dir": str(gfn_run_dir) if gfn_run_dir is not None else None,
        "acquisition": acquisition,
        "surrogate_type": surrogate_type,
        "scores": [float(s) for s in scores],
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_hybrid.json")

    return result
