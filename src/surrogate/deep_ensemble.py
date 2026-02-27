"""Deep ensemble surrogate with predictive uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepEnsembleSurrogate:
    """Bootstrap-ensemble MLP surrogate."""

    surrogate_type = "deep_ensemble"

    def __init__(
        self,
        max_length: int = 7,
        num_tokens: int = 27,
        n_models: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.input_dim = self.max_length * self.num_tokens
        self.models: list[_MLPRegressor] = []

    def _featurize(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.shape[1] != self.max_length:
            raise ValueError(
                f"Expected states with length {self.max_length}, got {states.shape[1]}"
            )
        onehot = np.eye(self.num_tokens, dtype=np.float32)[states]
        return onehot.reshape(states.shape[0], -1)

    def _train_single(self, x: torch.Tensor, y: torch.Tensor, seed: int) -> _MLPRegressor:
        model = _MLPRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], generator=generator, device=self.device)
            for start in range(0, x.shape[0], self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = model(x[idx])
                loss = criterion(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Fit bootstrap ensemble."""
        x_np = self._featurize(states)
        y_np = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        n = x.shape[0]
        rng = np.random.default_rng(0)
        self.models = []
        for m in range(self.n_models):
            bootstrap_idx = rng.integers(0, n, size=n)
            x_boot = x[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            self.models.append(self._train_single(x_boot, y_boot, seed=m + 13))

    def predict(
        self,
        states: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict with ensemble moments."""
        if not self.models:
            raise RuntimeError("Deep ensemble is not fitted.")
        x_np = self._featurize(states)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds.append(model(x).squeeze(-1).detach().cpu().numpy())
        pred_matrix = np.stack(preds, axis=0)
        mean = pred_matrix.mean(axis=0)
        std = pred_matrix.std(axis=0) + 1e-6
        if return_std:
            return mean, std
        return mean

    def sample(self, states: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Independent Gaussian draws from ensemble moments."""
        mean, std = self.predict(states, return_std=True)
        rng = np.random.default_rng()
        return rng.normal(
            loc=mean[None, :],
            scale=np.maximum(std[None, :], 1e-9),
            size=(n_samples, mean.shape[0]),
        )

    def save(self, path: str | Path) -> Path:
        """Persist ensemble weights and hyperparameters."""
        if not self.models:
            raise RuntimeError("Deep ensemble is not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "surrogate_type": self.surrogate_type,
            "max_length": self.max_length,
            "num_tokens": self.num_tokens,
            "n_models": self.n_models,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "state_dicts": [m.state_dict() for m in self.models],
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DeepEnsembleSurrogate":
        """Load a persisted deep ensemble."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            max_length=int(payload["max_length"]),
            num_tokens=int(payload["num_tokens"]),
            n_models=int(payload["n_models"]),
            hidden_dim=int(payload["hidden_dim"]),
            n_layers=int(payload["n_layers"]),
            dropout=float(payload["dropout"]),
            lr=float(payload["lr"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            device=device,
        )
        model.models = []
        for state_dict in payload["state_dicts"]:
            net = _MLPRegressor(
                input_dim=model.input_dim,
                hidden_dim=model.hidden_dim,
                n_layers=model.n_layers,
                dropout=model.dropout,
            ).to(model.device)
            net.load_state_dict(state_dict)
            net.eval()
            model.models.append(net)
        return model
