"""Monte-Carlo dropout Bayesian neural surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _DropoutRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
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


class DropoutBNNSurrogate:
    """Single-network surrogate with MC-dropout uncertainty."""

    surrogate_type = "bnn"

    def __init__(
        self,
        max_length: int = 7,
        num_tokens: int = 27,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 160,
        batch_size: int = 64,
        mc_samples: int = 40,
        device: str = "cpu",
    ):
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.device = torch.device(device)
        self.input_dim = self.max_length * self.num_tokens

        self.model = _DropoutRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

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

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Fit dropout network."""
        x_np = self._featurize(states)
        y_np = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], device=self.device)
            for start in range(0, x.shape[0], self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = self.model(x[idx])
                loss = criterion(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()

    def predict(
        self,
        states: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict via MC-dropout moments."""
        x_np = self._featurize(states)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        samples = []
        self.model.train()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                samples.append(self.model(x).squeeze(-1).detach().cpu().numpy())
        pred_matrix = np.stack(samples, axis=0)
        mean = pred_matrix.mean(axis=0)
        std = pred_matrix.std(axis=0) + 1e-6
        self.model.eval()
        if return_std:
            return mean, std
        return mean

    def sample(self, states: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw samples from predictive marginals."""
        mean, std = self.predict(states, return_std=True)
        rng = np.random.default_rng()
        return rng.normal(
            loc=mean[None, :],
            scale=np.maximum(std[None, :], 1e-9),
            size=(n_samples, mean.shape[0]),
        )

    def save(self, path: str | Path) -> Path:
        """Persist BNN checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "surrogate_type": self.surrogate_type,
            "max_length": self.max_length,
            "num_tokens": self.num_tokens,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "mc_samples": self.mc_samples,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DropoutBNNSurrogate":
        """Load persisted BNN checkpoint."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            max_length=int(payload["max_length"]),
            num_tokens=int(payload["num_tokens"]),
            hidden_dim=int(payload["hidden_dim"]),
            n_layers=int(payload["n_layers"]),
            dropout=float(payload["dropout"]),
            lr=float(payload["lr"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            mc_samples=int(payload["mc_samples"]),
            device=device,
        )
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
