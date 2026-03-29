"""Budget-aware oracle proxy compatible with the official GFlowNet Proxy API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.proxy.scrabble import ScrabbleScorer
from gflownet.utils.common import tfloat


class OracleProxy(Proxy):
    """
    Proxy wrapper around Scrabble oracle scoring with query-budget enforcement.

    The proxy only exposes the oracle-backed scoring path retained for the
    preliminary milestone.
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        reward_function: str = "identity",
        logreward_function=None,
        reward_function_kwargs: dict[str, Any] | None = None,
        reward_min: float = 0.0,
        do_clip_rewards: bool = False,
        oracle_budget: int | None = None,
        enforce_budget: bool = True,
        vocabulary_check: bool = False,
        stats_output_path: str | None = None,
        **kwargs: Any,
    ):
        self.oracle_budget = oracle_budget
        self.enforce_budget = enforce_budget
        self.vocabulary_check = vocabulary_check
        self.stats_output_path = Path(stats_output_path) if stats_output_path else None

        self.call_count = 0
        self.batch_count = 0
        self.call_history: list[dict[str, Any]] = []
        self._last_serialized_states: list[Any] = []
        self._env = None

        if reward_function_kwargs is None:
            reward_function_kwargs = {}

        super().__init__(
            device=device,
            float_precision=float_precision,
            reward_function=reward_function,
            logreward_function=logreward_function,
            reward_function_kwargs=reward_function_kwargs,
            reward_min=reward_min,
            do_clip_rewards=do_clip_rewards,
            **kwargs,
        )

        self.oracle_scorer = ScrabbleScorer(
            vocabulary_check=vocabulary_check,
            device=device,
            float_precision=float_precision,
            reward_function="identity",
            reward_function_kwargs={},
            reward_min=0.0,
            do_clip_rewards=False,
        )

    @property
    def remaining_budget(self) -> int | float:
        """Remaining oracle budget."""
        if self.oracle_budget is None or self.oracle_budget < 0:
            return float("inf")
        return max(int(self.oracle_budget) - int(self.call_count), 0)

    def setup(self, env=None):
        """Set up proxy with environment information."""
        self._env = env
        self.oracle_scorer.setup(env)

    def reset_tracking(self) -> None:
        """Reset call counters and history."""
        self.call_count = 0
        self.batch_count = 0
        self.call_history = []
        self._write_stats()

    def __call__(
        self, states: TensorType | list | npt.NDArray
    ) -> TensorType["batch"]:
        """Return proxy scores for a batch of states."""
        n_queries = self._batch_size(states)
        self._check_budget(n_queries)
        self._last_serialized_states = self._serialize_states(states)
        scores = self.oracle_scorer(self._prepare_oracle_states(states))
        scores_tensor = tfloat(scores, device=self.device, float_type=self.float)
        self._record_calls(n_queries=n_queries, scores=scores_tensor)
        return scores_tensor

    def _check_budget(self, n_queries: int) -> None:
        if self.oracle_budget is None or self.oracle_budget < 0:
            return
        if self.call_count + n_queries <= int(self.oracle_budget):
            return
        if self.enforce_budget:
            raise RuntimeError(
                "Oracle query budget exceeded: "
                f"used={self.call_count}, requested={n_queries}, budget={self.oracle_budget}"
            )

    def _record_calls(self, n_queries: int, scores: torch.Tensor) -> None:
        self.call_count += int(n_queries)
        self.batch_count += 1
        scores_list = [float(x) for x in scores.detach().cpu().tolist()]
        states_list = list(self._last_serialized_states)
        self.call_history.append(
            {
                "batch": int(self.batch_count),
                "n_queries": int(n_queries),
                "score_mean": float(scores.mean().item()) if scores.numel() > 0 else 0.0,
                "score_max": float(scores.max().item()) if scores.numel() > 0 else 0.0,
                "scores": scores_list,
                "states": states_list,
            }
        )

        if self._env is not None and hasattr(self._env, "record_oracle_query"):
            try:
                self._env.record_oracle_query(
                    states=states_list,
                    rewards=scores_list,
                )
            except Exception:
                pass

        self._write_stats()

    def _write_stats(self) -> None:
        if self.stats_output_path is None:
            return
        self.stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "oracle",
            "call_count": int(self.call_count),
            "batch_count": int(self.batch_count),
            "oracle_budget": self.oracle_budget,
            "remaining_budget": self.remaining_budget,
            "call_history": self.call_history,
        }
        with self.stats_output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _batch_size(self, states: TensorType | list | npt.NDArray) -> int:
        if torch.is_tensor(states):
            if states.ndim == 1:
                return 1
            return int(states.shape[0])
        if isinstance(states, np.ndarray):
            if states.ndim == 1:
                return 1
            return int(states.shape[0])
        if isinstance(states, list):
            return len(states)
        raise TypeError(f"Unsupported state container type: {type(states)}")

    def _prepare_oracle_states(
        self, states: TensorType | list | npt.NDArray
    ) -> TensorType | list:
        """Convert numeric state batches to the tensor format expected by ScrabbleScorer."""
        if torch.is_tensor(states):
            return states.to(device=self.device, dtype=torch.long)

        if isinstance(states, np.ndarray):
            return torch.as_tensor(states, device=self.device, dtype=torch.long)

        if isinstance(states, list):
            if not states:
                return torch.empty((0, 0), device=self.device, dtype=torch.long)
            first = states[0]
            if isinstance(first, str):
                return states
            if isinstance(first, (list, tuple, np.ndarray)):
                if len(first) > 0 and isinstance(first[0], str):
                    return states
            matrix = np.asarray(states, dtype=np.int64)
            return torch.as_tensor(matrix, device=self.device, dtype=torch.long)

        raise TypeError(f"Unsupported state container type: {type(states)}")

    def _serialize_states(
        self, states: TensorType | list | npt.NDArray
    ) -> list[list[int] | str]:
        """Convert states into a JSON-serializable representation."""
        if torch.is_tensor(states):
            tensor = states.detach().cpu()
            if tensor.ndim == 1:
                return [tensor.to(dtype=torch.long).tolist()]
            return tensor.to(dtype=torch.long).tolist()

        if isinstance(states, np.ndarray):
            array = np.asarray(states)
            if array.ndim == 1:
                return [array.astype(np.int64).tolist()]
            return array.astype(np.int64).tolist()

        if isinstance(states, list):
            if not states:
                return []
            first = states[0]
            if isinstance(first, str):
                return [str(state) for state in states]
            if isinstance(first, (list, tuple, np.ndarray)):
                if len(first) > 0 and isinstance(first[0], str):
                    return [" ".join(sample) for sample in states]
                return np.asarray(states, dtype=np.int64).tolist()
            return [states]  # pragma: no cover - defensive

        raise TypeError(f"Unsupported state container type: {type(states)}")

