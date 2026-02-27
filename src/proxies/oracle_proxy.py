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

from surrogate import load_surrogate


class OracleProxy(Proxy):
    """
    Proxy wrapper around Scrabble oracle scoring with query-budget enforcement.

    The proxy supports two backends:
    - ``oracle``: calls :class:`gflownet.proxy.scrabble.ScrabbleScorer`
    - ``surrogate``: loads and evaluates a persisted surrogate checkpoint
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
        backend: str = "oracle",
        surrogate_checkpoint: str | None = None,
        stats_output_path: str | None = None,
        **kwargs: Any,
    ):
        self.oracle_budget = oracle_budget
        self.enforce_budget = enforce_budget
        self.vocabulary_check = vocabulary_check
        self.backend = backend
        self.surrogate_checkpoint = surrogate_checkpoint
        self.stats_output_path = Path(stats_output_path) if stats_output_path else None

        self.call_count = 0
        self.batch_count = 0
        self.call_history: list[dict[str, Any]] = []
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
        self.surrogate = None

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
        if self.backend == "surrogate":
            self._load_surrogate()

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
        if self.backend == "oracle":
            self._check_budget(n_queries)
            scores = self.oracle_scorer(states)
            scores_tensor = tfloat(scores, device=self.device, float_type=self.float)
            self._record_calls(n_queries=n_queries, scores=scores_tensor)
            return scores_tensor

        if self.backend == "surrogate":
            self._load_surrogate()
            state_matrix = self._states_to_matrix(states)
            mean, _ = self.surrogate.predict(state_matrix, return_std=True)
            return tfloat(mean, device=self.device, float_type=self.float)

        raise ValueError(
            f"Unsupported backend '{self.backend}'. Expected 'oracle' or 'surrogate'."
        )

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
        self.call_history.append(
            {
                "batch": int(self.batch_count),
                "n_queries": int(n_queries),
                "score_mean": float(scores.mean().item()) if scores.numel() > 0 else 0.0,
                "score_max": float(scores.max().item()) if scores.numel() > 0 else 0.0,
                "scores": scores_list,
            }
        )

        if self._env is not None and hasattr(self._env, "record_oracle_query"):
            try:
                self._env.record_oracle_query(
                    states=[None] * n_queries,
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
            "backend": self.backend,
            "call_count": int(self.call_count),
            "batch_count": int(self.batch_count),
            "oracle_budget": self.oracle_budget,
            "remaining_budget": self.remaining_budget,
            "call_history": self.call_history,
        }
        with self.stats_output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_surrogate(self) -> None:
        if self.surrogate is not None:
            return
        if not self.surrogate_checkpoint:
            raise ValueError(
                "backend='surrogate' requires surrogate_checkpoint to be provided."
            )
        self.surrogate = load_surrogate(self.surrogate_checkpoint, device=str(self.device))

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

    def _states_to_matrix(self, states: TensorType | list | npt.NDArray) -> np.ndarray:
        if torch.is_tensor(states):
            matrix = states.detach().cpu().numpy()
        elif isinstance(states, np.ndarray):
            matrix = states
        elif isinstance(states, list):
            if len(states) == 0:
                if self._env is None:
                    return np.zeros((0, 0), dtype=np.int64)
                return np.zeros((0, self._env.max_length), dtype=np.int64)
            first = states[0]
            if isinstance(first, str):
                if self._env is None:
                    raise RuntimeError("String state inputs require proxy.setup(env).")
                matrix = np.asarray(
                    [self._env.readable2state(self._to_readable(item)) for item in states],
                    dtype=np.int64,
                )
            elif isinstance(first, (list, tuple, np.ndarray)):
                if len(first) > 0 and isinstance(first[0], str):
                    if self._env is None:
                        raise RuntimeError("Token list inputs require proxy.setup(env).")
                    matrix = np.asarray(
                        [self._tokens_to_indices(item) for item in states], dtype=np.int64
                    )
                else:
                    matrix = np.asarray(states, dtype=np.int64)
            else:
                raise TypeError(f"Unsupported list state element type: {type(first)}")
        else:
            raise TypeError(f"Unsupported state container type: {type(states)}")

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    def _to_readable(self, item: str) -> str:
        text = item.strip()
        if " " in text:
            return text
        return " ".join(list(text.upper()))

    def _tokens_to_indices(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        assert self._env is not None
        values = [self._env.token2idx[token.upper()] for token in tokens]
        if len(values) < self._env.max_length:
            values = values + [self._env.pad_idx] * (self._env.max_length - len(values))
        return values[: self._env.max_length]
