"""Supervised baseline training pipeline (independent of GFlowNet training)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from training.common import filter_new_states
from training.dataset import deduplicate_state_scores, load_dataset
from training.dataset import sample_terminating_states
from utils.device import resolve_device
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import regression_metrics, search_quality_metrics
from utils.scrabble import resolve_scrabble_optimum


class _BaselineNet(nn.Module):
    """Simple shared-trunk model for validity and score prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.validity_head = nn.Linear(dim, 1)
        self.score_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        validity_logits = self.validity_head(hidden)
        score_pred = F.softplus(self.score_head(hidden))
        return validity_logits, score_pred


def _states_to_features(states: np.ndarray, num_tokens: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    onehot = np.eye(num_tokens, dtype=np.float32)[states]
    return onehot.reshape(states.shape[0], -1)


def _train_val_split(
    targets: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stable split that preserves positive examples when possible."""
    positives = np.flatnonzero(targets > 0)
    negatives = np.flatnonzero(targets <= 0)

    if positives.size < 2 or negatives.size < 2:
        perm = rng.permutation(targets.shape[0])
        n_train = int(targets.shape[0] * train_fraction)
        n_train = max(1, min(n_train, targets.shape[0] - 1))
        return perm[:n_train], perm[n_train:]

    pos_perm = rng.permutation(positives)
    neg_perm = rng.permutation(negatives)

    n_train_pos = int(np.ceil(pos_perm.size * train_fraction))
    n_train_neg = int(np.ceil(neg_perm.size * train_fraction))
    n_train_pos = max(1, min(n_train_pos, pos_perm.size - 1))
    n_train_neg = max(1, min(n_train_neg, neg_perm.size - 1))

    train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
    val_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])
    return rng.permutation(train_idx), rng.permutation(val_idx)


def _forward_predictions(
    model: _BaselineNet,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validity probability, score-if-valid, and final score prediction."""
    validity_logits, score_pred = model(x)
    validity_prob = torch.sigmoid(validity_logits)
    final_pred = validity_prob * score_pred
    return validity_prob, score_pred, final_pred


def _batch_loss(
    model: _BaselineNet,
    x: torch.Tensor,
    y: torch.Tensor,
    bce: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute baseline loss and final prediction."""
    validity_logits, score_pred = model(x)
    validity_targets = (y > 0).float()

    loss_validity = bce(validity_logits, validity_targets)
    if validity_targets.sum().item() > 0:
        loss_score = (((score_pred - y) ** 2) * validity_targets).sum() / validity_targets.sum()
    else:
        loss_score = torch.zeros((), device=y.device)

    final_pred = torch.sigmoid(validity_logits) * score_pred
    return loss_validity + loss_score, final_pred


def _validity_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute simple classification metrics for valid-word detection."""
    target_pos = targets > 0
    pred_pos = probs >= 0.5

    tp = int(np.logical_and(target_pos, pred_pos).sum())
    tn = int(np.logical_and(~target_pos, ~pred_pos).sum())
    fp = int(np.logical_and(~target_pos, pred_pos).sum())
    fn = int(np.logical_and(target_pos, ~pred_pos).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(targets.shape[0], 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_supervised_baseline(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the supervised baseline and return experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    seed = int(config["seed"])
    device = resolve_device(config.get("device", "cpu"))
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    dataset_cfg = config["dataset"]
    baseline_cfg = config["baseline"]
    optimum_info = resolve_scrabble_optimum(
        max_length=int(env_cfg["max_length"]),
        vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        configured_optimum_score=config.get("metrics", {}).get("optimum_score"),
        gflownet_root=config.get("gflownet_root"),
    )
    optimum_score = float(optimum_info["optimum_score"])

    dataset_path = dataset_cfg.get("path")
    if not dataset_path:
        raise ValueError(
            "baseline requires dataset.path to point to a saved .npz file. "
            "Run `python experiments/run_dataset.py` first."
        )

    states_np, targets = load_dataset(dataset_path)
    states_np, targets = deduplicate_state_scores(states_np, targets)
    if states_np.shape[0] != targets.shape[0]:
        raise ValueError(
            "dataset states and scores have inconsistent lengths: "
            f"{states_np.shape[0]} vs {targets.shape[0]}"
        )
    if states_np.shape[0] < 2:
        raise ValueError(
            "Supervised baseline requires at least 2 labeled samples in the dataset. "
            f"Received {states_np.shape[0]}."
        )

    initial_quality = search_quality_metrics(
        scores=targets.tolist(),
        states=states_np.tolist(),
        oracle_queries=int(states_np.shape[0]),
        optimum_score=optimum_score,
        top_k=10,
        pad_value=0,
    )
    num_tokens = int(env_cfg.get("num_tokens", 27))
    features = _states_to_features(states_np, num_tokens=num_tokens)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = _train_val_split(
        targets=targets,
        train_fraction=float(baseline_cfg.get("train_fraction", 0.8)),
        rng=rng,
    )

    x_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32, device=device).view(-1, 1)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32, device=device).view(-1, 1)

    model = _BaselineNet(
        input_dim=features.shape[1],
        hidden_dim=int(baseline_cfg.get("hidden_dim", 256)),
        n_layers=int(baseline_cfg.get("n_layers", 2)),
        dropout=float(baseline_cfg.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(baseline_cfg.get("lr", 1e-3)))
    epochs = int(baseline_cfg.get("epochs", 250))
    batch_size = int(baseline_cfg.get("batch_size", 64))
    positive_weight = float(baseline_cfg.get("positive_weight", 5.0))
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start : start + batch_size]
            loss, _ = _batch_loss(model, x_train[idx], y_train[idx], bce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss, _ = _batch_loss(model, x_train, y_train, bce)
            val_loss, val_pred = _batch_loss(model, x_val, y_val, bce)
        train_losses.append(float(train_loss.item()))
        val_losses.append(float(val_loss.item()))
        if logger is not None and epoch % max(1, epochs // 20) == 0:
            logger.log_metrics(
                step=epoch,
                metrics={
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(val_loss.item()),
                },
            )

    with torch.no_grad():
        val_prob, _, val_pred = _forward_predictions(model, x_val)
    val_prob_np = val_prob.squeeze(-1).detach().cpu().numpy()
    val_pred_np = val_pred.squeeze(-1).detach().cpu().numpy()
    val_targets = y_val.squeeze(-1).detach().cpu().numpy()
    reg_metrics = regression_metrics(val_targets, val_pred_np)
    validity = _validity_metrics(val_targets, val_prob_np)

    model_path = output_dir / "supervised_baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    evaluated_states = states_np.tolist()
    evaluated_scores = targets.astype(float).tolist()
    candidate_pool_queries = 0
    additional_oracle_queries = 0
    selected_candidate_states: list[list[int]] = []
    selected_candidate_scores: list[float] = []
    remaining_budget = max(int(oracle_cfg.get("budget", states_np.shape[0])) - int(states_np.shape[0]), 0)

    candidate_pool_size = int(baseline_cfg.get("candidate_pool_size", 0))
    if remaining_budget > 0 and candidate_pool_size > 0:
        env = ScrabbleOracleEnv(
            max_length=int(env_cfg["max_length"]),
            oracle_budget=remaining_budget,
            device=device,
        )
        oracle = OracleProxy(
            device=device,
            float_precision=32,
            oracle_budget=remaining_budget,
            enforce_budget=bool(oracle_cfg.get("enforce_budget", True)),
            vocabulary_check=bool(oracle_cfg.get("vocabulary_check", False)),
        )
        oracle.setup(env)

        candidate_states = sample_terminating_states(
            env,
            candidate_pool_size,
            sampling_strategy=str(baseline_cfg.get("candidate_sampling_strategy", dataset_cfg.get("sampling_strategy", "uniform"))),
            min_length=int(baseline_cfg.get("min_length", dataset_cfg.get("min_length", 3))),
            unique=bool(baseline_cfg.get("candidate_unique", True)),
            seed=seed + 100_000,
            gflownet_root=config.get("gflownet_root"),
        )
        filtered_candidates = filter_new_states(candidate_states, evaluated_states)
        if filtered_candidates:
            candidate_matrix = np.asarray(filtered_candidates, dtype=np.int64)
            candidate_pool_queries = int(candidate_matrix.shape[0])
            candidate_features = torch.tensor(
                _states_to_features(candidate_matrix, num_tokens=num_tokens),
                dtype=torch.float32,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                _, _, candidate_pred = _forward_predictions(model, candidate_features)
            predicted_scores = candidate_pred.squeeze(-1).detach().cpu().numpy()
            selection_size = int(min(remaining_budget, candidate_matrix.shape[0]))
            top_indices = np.argsort(predicted_scores)[::-1][:selection_size]
            selected_candidate_states = candidate_matrix[top_indices].tolist()
            if selected_candidate_states:
                selected_candidate_scores = (
                    oracle(np.asarray(selected_candidate_states, dtype=np.int64))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )
                additional_oracle_queries = len(selected_candidate_states)
                evaluated_states.extend(selected_candidate_states)
                evaluated_scores.extend(float(score) for score in selected_candidate_scores)

    quality = search_quality_metrics(
        scores=evaluated_scores,
        states=evaluated_states,
        oracle_queries=int(states_np.shape[0] + additional_oracle_queries),
        optimum_score=optimum_score,
        top_k=10,
        pad_value=0,
    )

    result = {
        "method": "supervised_baseline",
        "seed": seed,
        **quality,
        "initial_best_score": float(initial_quality["best_score"]),
        "initial_top10_score": float(initial_quality["top10_score"]),
        "dataset_path": str(Path(dataset_path)),
        "num_samples": int(states_np.shape[0]),
        "num_positive_samples": int((targets > 0).sum()),
        "regression": reg_metrics,
        "validity": validity,
        "model_path": str(model_path),
        "train_loss_final": float(train_losses[-1]),
        "val_loss_final": float(val_losses[-1]),
        "scores": [float(score) for score in evaluated_scores],
        "training_scores": targets.astype(float).tolist(),
        "selected_candidate_scores": [float(score) for score in selected_candidate_scores],
        "real_oracle_queries": int(states_np.shape[0] + additional_oracle_queries),
        "fake_oracle_queries": 0,
        "candidate_pool_queries": int(candidate_pool_queries),
        "cheap_model_queries": int(candidate_pool_queries),
        "additional_oracle_queries": int(additional_oracle_queries),
        "remaining_oracle_budget": int(max(remaining_budget - additional_oracle_queries, 0)),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "optimum_score": optimum_score,
    }
    if selected_candidate_states:
        result["selected_candidate_states"] = selected_candidate_states
    if optimum_info.get("optimum_words"):
        result["optimum_words"] = list(optimum_info["optimum_words"])
        result["optimum_word_count"] = int(optimum_info["optimum_word_count"])
        result["optimum_source"] = str(optimum_info["optimum_source"])

    if logger is not None:
        logger.dump_summary(result, filename="summary_baseline.json")

    return result
