"""Proxy wrapper that exposes surrogate predictions as GFlowNet rewards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

from training.common import compute_plausibility_bonus
from surrogate.factory import load_surrogate_checkpoint


class SurrogateProxy(Proxy):
    """Proxy that scores states with a saved surrogate model checkpoint.

    Prediction modes
    ----------------
    mean      : Use the surrogate posterior mean.  Best choice for GFlowNet
                training — the model learns to sample from high-*value* regions,
                not high-uncertainty regions (which is what UCB optimises for).
    ucb       : mean + exploration_beta × std.  Useful as a quick acquisition
                signal but biases the GFlowNet toward uncertain, often-invalid
                candidates.
    Reward transforms
    -----------------
    exp_beta        : exp(beta_scale × clamp(score / score_max, 0, 1))
                      Proper Boltzmann reward: normalises scores to [0,1] then
                      exponentiates.  With beta_scale=5 the top-scoring word
                      gets ≈148× more reward than an invalid word — the GFlowNet
                      receives a sharp, highly discriminative signal.
    softplus        : log(1 + exp(score)).  Nearly identity for large scores;
                      preserved for backward compatibility.
    clip_positive   : clamp(score, 0, ∞).
    identity        : No transform.

    Plausibility modes
    ------------------
    additive        : reward += plausibility_weight × plausibility(x)
                      Shifts all candidates uniformly — the original behaviour.
    multiplicative  : reward *= 1 + plausibility_weight × plausibility(x)
                      Amplifies word-like candidates *proportionally*: a
                      high-scoring word-like state gets a larger absolute boost
                      than a low-scoring one, sharpening the reward landscape.
    """

    def __init__(
        self,
        surrogate_path: str,
        prediction_mode: str = "mean",
        exploration_beta: float = 1.0,
        reward_transform: str = "softplus",
        beta_scale: float = 5.0,
        score_max: float = 45.0,
        plausibility_mode: str = "additive",
        max_length: int | None = None,
        min_length: int = 0,
        gflownet_root: str | None = None,
        plausibility_weight: float = 0.0,
        stats_output_path: str | None = None,
        **kwargs,
    ):
        self.surrogate_path = str(Path(surrogate_path))
        self.prediction_mode = str(prediction_mode).lower()
        self.exploration_beta = float(exploration_beta)
        self.reward_transform = str(reward_transform).lower()
        self.beta_scale = float(beta_scale)
        self.score_max = max(float(score_max), 1e-6)
        self.plausibility_mode = str(plausibility_mode).lower()
        self.max_length = int(max_length) if max_length is not None else None
        self.min_length = max(int(min_length), 0)
        self.gflownet_root = str(gflownet_root) if gflownet_root is not None else None
        self.plausibility_weight = float(plausibility_weight)
        self.stats_output_path = Path(stats_output_path) if stats_output_path else None
        self.surrogate = None
        self.call_count = 0
        self.batch_count = 0
        self.call_history: list[dict[str, float | int | str]] = []
        super().__init__(**kwargs)

    def setup(self, env=None):
        self.surrogate = load_surrogate_checkpoint(
            self.surrogate_path,
            device=str(self.device),
        )
        self._env = env
        self._write_stats()

    def reset_tracking(self) -> None:
        """Reset fake-oracle accounting between GFlowNet training rounds."""
        self.call_count = 0
        self.batch_count = 0
        self.call_history = []
        self._write_stats()

    def __call__(
        self, states: TensorType | list | npt.NDArray
    ) -> TensorType["batch"]:
        if self.surrogate is None:
            raise RuntimeError("SurrogateProxy.setup(env) must be called before use.")

        states_np = self._states_to_numpy(states)
        mean, std = self.surrogate.predict(states_np, return_std=True)

        # --- Step 1: derive a scalar score from the surrogate posterior ---------
        if self.prediction_mode == "mean":
            values = mean
        elif self.prediction_mode == "ucb":
            values = mean + self.exploration_beta * std
        else:
            raise ValueError(
                f"Unsupported prediction_mode: {self.prediction_mode!r}. "
                "Use one of {'mean', 'ucb'}."
            )

        # --- Step 2: apply reward transform to shape the distribution ----------
        values_tensor = torch.as_tensor(
            np.asarray(values, dtype=np.float64), dtype=self.float, device=self.device
        )

        if self.reward_transform == "exp_beta":
            # Boltzmann reward: normalise scores to [0,1] then exponentiate.
            # R(x) = exp(beta_scale × clamp(score / score_max, 0, 1))
            # With beta_scale=5: best word → e^5 ≈ 148, invalid word → e^0 = 1.
            # Creates a 148× discrimination ratio vs ~8× for raw softplus,
            # giving the GFlowNet a much sharper signal about where the peaks are.
            score_norm = (values_tensor / self.score_max).clamp(0.0, 1.0)
            reward_values = torch.exp(self.beta_scale * score_norm)
        elif self.reward_transform == "identity":
            reward_values = values_tensor
        elif self.reward_transform == "softplus":
            reward_values = F.softplus(values_tensor)
        elif self.reward_transform == "clip_positive":
            reward_values = values_tensor.clamp_min(0.0)
        else:
            raise ValueError(
                f"Unsupported reward_transform: {self.reward_transform!r}. "
                "Use one of {'exp_beta', 'identity', 'softplus', 'clip_positive'}."
            )

        # --- Step 3: integrate word-plausibility prior -------------------------
        if self.max_length is not None and self.plausibility_weight != 0.0:
            plaus_bonus = compute_plausibility_bonus(
                states_np,
                int(self.max_length),
                gflownet_root=self.gflownet_root,
                weight=self.plausibility_weight,
            )
            plaus_tensor = torch.as_tensor(
                np.asarray(plaus_bonus, dtype=np.float64),
                dtype=self.float,
                device=self.device,
            )
            if self.plausibility_mode == "multiplicative":
                # R_final = R_base × (1 + plausibility_weight × plausibility(x))
                # Word-like candidates get a *proportional* boost — a high-
                # scoring valid word receives a larger absolute amplification
                # than a low-scoring invalid string.
                reward_values = reward_values * (1.0 + plaus_tensor)
            else:
                # Original additive behaviour.
                reward_values = reward_values + plaus_tensor

        if self.min_length > 0:
            lengths = np.count_nonzero(states_np != 0, axis=1)
            too_short = torch.as_tensor(
                lengths < self.min_length,
                dtype=torch.bool,
                device=self.device,
            )
            if too_short.any():
                reward_values = reward_values.clone()
                reward_values[too_short] = float(max(self.reward_min, 0.0))

        self._record_calls(states_np, reward_values)
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

    def _record_calls(self, states_np: np.ndarray, reward_values: torch.Tensor) -> None:
        n_queries = int(states_np.shape[0])
        self.call_count += n_queries
        self.batch_count += 1
        payload = {
            "batch": int(self.batch_count),
            "n_queries": n_queries,
            "prediction_mode": self.prediction_mode,
            "reward_mean": float(reward_values.mean().item()) if reward_values.numel() > 0 else 0.0,
            "reward_max": float(reward_values.max().item()) if reward_values.numel() > 0 else 0.0,
        }
        self.call_history.append(payload)
        self._write_stats()

    def _write_stats(self) -> None:
        if self.stats_output_path is None:
            return
        self.stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "surrogate",
            "surrogate_path": self.surrogate_path,
            "prediction_mode": self.prediction_mode,
            "exploration_beta": self.exploration_beta,
            "reward_transform": self.reward_transform,
            "beta_scale": self.beta_scale,
            "score_max": self.score_max,
            "plausibility_mode": self.plausibility_mode,
            "plausibility_weight": self.plausibility_weight,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "call_count": int(self.call_count),
            "batch_count": int(self.batch_count),
            "call_history": self.call_history,
        }
        with self.stats_output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
