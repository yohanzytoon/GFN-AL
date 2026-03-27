"""Proxy wrapper that exposes surrogate predictions as GFlowNet rewards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

from surrogate.factory import load_surrogate_checkpoint


class SurrogateProxy(Proxy):
    """Proxy that scores states with a saved surrogate model checkpoint."""

    def __init__(
        self,
        surrogate_path: str,
        prediction_mode: str = "mean",
        exploration_beta: float = 1.0,
        reward_transform: str = "softplus",
        **kwargs,
    ):
        self.surrogate_path = str(Path(surrogate_path))
        self.prediction_mode = str(prediction_mode).lower()
        self.exploration_beta = float(exploration_beta)
        self.reward_transform = str(reward_transform).lower()
        self.surrogate = None
        super().__init__(**kwargs)

    def setup(self, env=None):
        self.surrogate = load_surrogate_checkpoint(
            self.surrogate_path,
            device=str(self.device),
        )
        self._env = env

    def __call__(
        self, states: TensorType | list | npt.NDArray
    ) -> TensorType["batch"]:
        if self.surrogate is None:
            raise RuntimeError("SurrogateProxy.setup(env) must be called before use.")

        states_np = self._states_to_numpy(states)
        mean, std = self.surrogate.predict(states_np, return_std=True)

        if self.prediction_mode == "mean":
            values = mean
        elif self.prediction_mode == "ucb":
            values = mean + self.exploration_beta * std
        else:
            raise ValueError(
                f"Unsupported prediction_mode: {self.prediction_mode}. "
                "Use one of {'mean', 'ucb'}."
            )

        values_tensor = torch.as_tensor(values, dtype=self.float, device=self.device)
        if self.reward_transform == "identity":
            reward_values = values_tensor
        elif self.reward_transform == "softplus":
            reward_values = F.softplus(values_tensor)
        elif self.reward_transform == "clip_positive":
            reward_values = values_tensor.clamp_min(0.0)
        else:
            raise ValueError(
                f"Unsupported reward_transform: {self.reward_transform}. "
                "Use one of {'identity', 'softplus', 'clip_positive'}."
            )
        return tfloat(reward_values, device=self.device, float_type=self.float)

    def _states_to_numpy(
        self, states: TensorType | list | npt.NDArray
    ) -> np.ndarray:
        if torch.is_tensor(states):
            array = states.detach().cpu().numpy()
        else:
            array = np.asarray(states)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array.astype(np.int64)
