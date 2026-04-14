"""Deep-ensemble surrogate for predictive uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.scrabble import SCRABBLE_LETTER_SCORES, domain_features, ngram_plausibility, vocabulary_bigram_model


class _EnsembleMember(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.backbone(x)))


class DeepEnsembleSurrogate:
    """Bootstrapped deep ensemble for mean and uncertainty estimation."""

    surrogate_type = "ensemble"

    def __init__(
        self,
        max_length: int = 7,
        num_tokens: int = 27,
        device: str = "cpu",
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        ensemble_size: int = 5,
        epochs: int = 75,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self._members: list[_EnsembleMember] = []

    def _featurize(self, states: np.ndarray) -> np.ndarray:
        """Convert integer state vectors to feature vectors for the ensemble.

        Mirrors BoTorchGPSurrogate._featurize: one-hot + score_sum + word_len
        + domain features (vowel ratio, ngram plausibility, consonant clusters,
        has-vowel).  See BoTorchGPSurrogate._featurize for detailed documentation.
        """
        states = np.asarray(states, dtype=np.int64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.shape[1] != self.max_length:
            raise ValueError(
                f"Expected states with length {self.max_length}, got {states.shape[1]}"
            )
        onehot = np.eye(self.num_tokens, dtype=np.float32)[states]
        flat = onehot.reshape(states.shape[0], -1)

        letter_scores = SCRABBLE_LETTER_SCORES[states]
        score_sum = letter_scores.sum(axis=1, keepdims=True) / (self.max_length * 10.0)
        word_len = (states != 0).sum(axis=1, keepdims=True).astype(np.float32) / self.max_length

        # Domain features: vowel ratio, ngram plausibility, consonant clusters, has-vowel.
        dom_feats = domain_features(states, self.max_length)
        bigram_model = vocabulary_bigram_model(max_length=self.max_length)
        if bigram_model is not None:
            ngram_scores = ngram_plausibility(states, bigram_model)
            dom_feats[:, 1] = (ngram_scores + 5.0).clip(0.0, 10.0) / 10.0

        return np.concatenate([flat, score_sum, word_len, dom_feats], axis=1)

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Fit each ensemble member on a bootstrap resample."""
        x_np = self._featurize(states)
        y_np = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

        n_samples = x.shape[0]
        input_dim = x.shape[1]
        self._members = []

        for member_idx in range(self.ensemble_size):
            model = _EnsembleMember(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            rng = np.random.default_rng(member_idx)
            boot_idx = rng.integers(0, n_samples, size=n_samples)

            for _ in range(self.epochs):
                order = torch.randperm(n_samples, device=self.device)
                for start in range(0, n_samples, self.batch_size):
                    batch_perm = order[start : start + self.batch_size]
                    idx = torch.as_tensor(
                        boot_idx[batch_perm.detach().cpu().numpy()],
                        dtype=torch.long,
                        device=self.device,
                    )
                    pred = model(x[idx])
                    loss = F.mse_loss(pred, y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            self._members.append(model)

    def predict(
        self,
        states: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict ensemble mean and epistemic uncertainty."""
        if not self._members:
            raise RuntimeError("DeepEnsembleSurrogate must be fit before predict().")
        x_np = self._featurize(states)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = torch.stack([member(x).squeeze(-1) for member in self._members], dim=0)
        mean = preds.mean(dim=0).detach().cpu().numpy()
        std = preds.std(dim=0, unbiased=False).detach().cpu().numpy()
        if return_std:
            return mean, std
        return mean

    def sample(self, states: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw Gaussian samples from the ensemble predictive marginals."""
        mean, std = self.predict(states, return_std=True)
        rng = np.random.default_rng()
        draws = rng.normal(
            loc=mean[None, :],
            scale=np.maximum(std[None, :], 1e-9),
            size=(n_samples, mean.shape[0]),
        )
        return draws

    def save(self, path: str | Path) -> Path:
        """Persist ensemble checkpoint."""
        if not self._members:
            raise RuntimeError("DeepEnsembleSurrogate must be fit before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "surrogate_type": self.surrogate_type,
            "max_length": self.max_length,
            "num_tokens": self.num_tokens,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "ensemble_size": self.ensemble_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "state_dicts": [member.state_dict() for member in self._members],
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DeepEnsembleSurrogate":
        """Load ensemble checkpoint."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            max_length=int(payload["max_length"]),
            num_tokens=int(payload["num_tokens"]),
            device=device,
            hidden_dim=int(payload["hidden_dim"]),
            n_layers=int(payload["n_layers"]),
            dropout=float(payload["dropout"]),
            ensemble_size=int(payload["ensemble_size"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            lr=float(payload["lr"]),
        )
        input_dim = model._featurize(np.zeros((1, model.max_length), dtype=np.int64)).shape[1]
        model._members = []
        for state_dict in payload["state_dicts"]:
            member = _EnsembleMember(
                input_dim=input_dim,
                hidden_dim=model.hidden_dim,
                n_layers=model.n_layers,
                dropout=model.dropout,
            ).to(model.device)
            member.load_state_dict(state_dict)
            member.eval()
            model._members.append(member)
        return model
