"""Gaussian Process surrogate with BoTorch-first implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    BOTORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    BOTORCH_AVAILABLE = False

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class BoTorchGPSurrogate:
    """GP surrogate model with predictive mean and uncertainty."""

    surrogate_type = "gp"

    def __init__(
        self,
        max_length: int = 7,
        num_tokens: int = 27,
        device: str = "cpu",
        fit_maxiter: int = 100,
        prefer_botorch: bool = True,
    ):
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.device = torch.device(device)
        self.fit_maxiter = fit_maxiter
        self.prefer_botorch = prefer_botorch
        self.backend = "botorch" if (BOTORCH_AVAILABLE and prefer_botorch) else "sklearn"

        self._model = None
        self._mll = None
        self._x_mean = None
        self._x_std = None
        self._train_x = None
        self._train_y = None

        self._sk_model: GaussianProcessRegressor | None = None

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

    def _build_botorch(self) -> None:
        assert self._train_x is not None and self._train_y is not None
        assert self._x_mean is not None and self._x_std is not None
        x_norm = (self._train_x - self._x_mean) / self._x_std
        covar = ScaleKernel(RBFKernel(ard_num_dims=x_norm.shape[1]))
        self._model = SingleTaskGP(
            train_X=x_norm,
            train_Y=self._train_y,
            covar_module=covar,
            outcome_transform=Standardize(m=1),
        )
        self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Fit surrogate on state-target pairs."""
        x = self._featurize(states)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

        if self.backend == "botorch":
            train_x = torch.tensor(x, dtype=torch.float64, device=self.device)
            train_y = torch.tensor(y, dtype=torch.float64, device=self.device)
            self._train_x = train_x
            self._train_y = train_y
            self._x_mean = train_x.mean(dim=0, keepdim=True)
            self._x_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
            self._build_botorch()
            assert self._mll is not None
            try:
                fit_gpytorch_mll(
                    self._mll,
                    optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}},
                )
            except TypeError:
                fit_gpytorch_mll(self._mll, optimizer_kwargs={"maxiter": self.fit_maxiter})
            self._model.eval()
        else:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
            self._sk_model = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=0,
            )
            self._sk_model.fit(x, y.ravel())

    def predict(
        self,
        states: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict mean and uncertainty for candidate states."""
        x = self._featurize(states)

        if self.backend == "botorch":
            assert self._model is not None and self._x_mean is not None and self._x_std is not None
            x_tensor = torch.tensor(x, dtype=torch.float64, device=self.device)
            x_norm = (x_tensor - self._x_mean) / self._x_std
            with torch.no_grad():
                posterior = self._model.posterior(x_norm)
                mean = posterior.mean.squeeze(-1)
                std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
            mean_np = mean.detach().cpu().numpy()
            std_np = std.detach().cpu().numpy()
        else:
            assert self._sk_model is not None
            mean_np, std_np = self._sk_model.predict(x, return_std=True)

        if return_std:
            return mean_np, std_np
        return mean_np

    def sample(self, states: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw independent Gaussian samples from predictive marginals."""
        mean, std = self.predict(states, return_std=True)
        rng = np.random.default_rng()
        draws = rng.normal(
            loc=mean[None, :],
            scale=np.maximum(std[None, :], 1e-9),
            size=(n_samples, mean.shape[0]),
        )
        return draws

    def save(self, path: str | Path) -> Path:
        """Persist surrogate checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "surrogate_type": self.surrogate_type,
            "backend": self.backend,
            "max_length": self.max_length,
            "num_tokens": self.num_tokens,
            "fit_maxiter": self.fit_maxiter,
            "prefer_botorch": self.prefer_botorch,
        }
        if self.backend == "botorch":
            assert self._model is not None
            payload.update(
                {
                    "state_dict": self._model.state_dict(),
                    "train_x": self._train_x.detach().cpu(),
                    "train_y": self._train_y.detach().cpu(),
                    "x_mean": self._x_mean.detach().cpu(),
                    "x_std": self._x_std.detach().cpu(),
                }
            )
        else:
            payload["sklearn_model"] = self._sk_model

        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BoTorchGPSurrogate":
        """Load surrogate checkpoint."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            max_length=int(payload["max_length"]),
            num_tokens=int(payload["num_tokens"]),
            device=device,
            fit_maxiter=int(payload.get("fit_maxiter", 100)),
            prefer_botorch=bool(payload.get("prefer_botorch", True)),
        )
        model.backend = str(payload["backend"])
        if model.backend == "botorch":
            model._train_x = payload["train_x"].to(model.device, dtype=torch.float64)
            model._train_y = payload["train_y"].to(model.device, dtype=torch.float64)
            model._x_mean = payload["x_mean"].to(model.device, dtype=torch.float64)
            model._x_std = payload["x_std"].to(model.device, dtype=torch.float64)
            model._build_botorch()
            assert model._model is not None
            model._model.load_state_dict(payload["state_dict"])
            model._model.eval()
        else:
            model._sk_model = payload["sklearn_model"]
        return model
