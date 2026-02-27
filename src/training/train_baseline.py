"""Supervised baseline training pipeline (independent of GFlowNet training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from environments.scrabble_oracle_env import ScrabbleOracleEnv
from proxies.oracle_proxy import OracleProxy
from utils.logging import ExperimentLogger, set_global_seed
from utils.metrics import build_query_curve, regression_metrics, search_quality_metrics


class _MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _states_to_features(states: np.ndarray, num_tokens: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    onehot = np.eye(num_tokens, dtype=np.float32)[states]
    return onehot.reshape(states.shape[0], -1)


def run_supervised_baseline(
    config: dict[str, Any],
    output_dir: Path,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the supervised baseline and return experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_global_seed(seed)

    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    baseline_cfg = config["baseline"]

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
        backend="oracle",
    )
    oracle.setup(env)

    n_queries = int(min(oracle_cfg["budget"], baseline_cfg.get("num_queries", oracle_cfg["budget"])))
    random_states = env.get_random_terminating_states(
        n_states=n_queries,
        unique=False,
        max_attempts=max(5 * n_queries, 1000),
    )
    proxy_states = env.states2proxy(random_states)
    targets = oracle(proxy_states).detach().cpu().numpy().astype(np.float32)

    curve = build_query_curve(
        targets,
        optimum_score=config.get("metrics", {}).get("optimum_score"),
    )
    quality = search_quality_metrics(
        scores=targets.tolist(),
        states=random_states,
        oracle_queries=int(oracle.call_count),
        optimum_score=config.get("metrics", {}).get("optimum_score"),
        top_k=10,
        pad_value=0,
    )

    states_np = np.asarray(random_states, dtype=np.int64)
    features = _states_to_features(states_np, num_tokens=env.n_letters + 1)

    if features.shape[0] < 2:
        raise ValueError(
            "Supervised baseline requires at least 2 oracle queries to form train/val splits. "
            f"Received {features.shape[0]}."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(features.shape[0])
    n_train = int(features.shape[0] * float(baseline_cfg.get("train_fraction", 0.8)))
    n_train = max(1, min(n_train, features.shape[0] - 1))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.size == 0:
        val_idx = train_idx.copy()

    x_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32, device=device).view(-1, 1)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32, device=device).view(-1, 1)

    model = _MLPRegressor(
        input_dim=features.shape[1],
        hidden_dim=int(baseline_cfg.get("hidden_dim", 256)),
        n_layers=int(baseline_cfg.get("n_layers", 2)),
        dropout=float(baseline_cfg.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(baseline_cfg.get("lr", 1e-3)))
    criterion = nn.MSELoss()
    epochs = int(baseline_cfg.get("epochs", 250))
    batch_size = int(baseline_cfg.get("batch_size", 64))

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_train[idx])
            loss = criterion(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            val_loss = criterion(model(x_val), y_val).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if logger is not None and epoch % max(1, epochs // 20) == 0:
            logger.log_metrics(
                step=epoch,
                metrics={
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                },
            )

    with torch.no_grad():
        val_pred = model(x_val).squeeze(-1).detach().cpu().numpy()
    val_targets = y_val.squeeze(-1).detach().cpu().numpy()
    reg_metrics = regression_metrics(val_targets, val_pred)

    model_path = output_dir / "supervised_baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    result = {
        "method": "supervised_baseline",
        "seed": seed,
        **quality,
        "curve": curve,
        "regression": reg_metrics,
        "model_path": str(model_path),
        "train_loss_final": float(train_losses[-1]),
        "val_loss_final": float(val_losses[-1]),
        "scores": targets.tolist(),
    }

    if logger is not None:
        logger.dump_summary(result, filename="summary_baseline.json")

    return result
