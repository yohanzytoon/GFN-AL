"""Scrabble environment extension with oracle-budget bookkeeping."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from gflownet.envs.scrabble import Scrabble


class ScrabbleOracleEnv(Scrabble):
    """
    Extend the official Scrabble environment with oracle-query accounting.

    The state/action dynamics are unchanged; this class only tracks oracle reward calls
    and enforces query budgets when used with budget-aware proxies.
    """

    def __init__(
        self,
        oracle_budget: int | None = None,
        track_oracle_history: bool = True,
        **kwargs: Any,
    ):
        self.oracle_budget = oracle_budget
        self.track_oracle_history = track_oracle_history
        self.oracle_calls = 0
        self.oracle_history: list[dict[str, Any]] = []
        super().__init__(**kwargs)

    @property
    def remaining_oracle_budget(self) -> int | float:
        """Remaining number of oracle calls."""
        if self.oracle_budget is None or self.oracle_budget < 0:
            return float("inf")
        return max(int(self.oracle_budget) - int(self.oracle_calls), 0)

    def can_query_oracle(self, n_queries: int = 1) -> bool:
        """Whether `n_queries` additional oracle calls are allowed."""
        if self.oracle_budget is None or self.oracle_budget < 0:
            return True
        return self.oracle_calls + n_queries <= int(self.oracle_budget)

    def record_oracle_query(
        self,
        states: Sequence[Any],
        rewards: Sequence[float],
    ) -> None:
        """Register a batch oracle query."""
        n_queries = len(states)
        if not self.can_query_oracle(n_queries):
            raise RuntimeError(
                f"Oracle budget exceeded in env: used={self.oracle_calls}, "
                f"requested={n_queries}, budget={self.oracle_budget}"
            )
        self.oracle_calls += n_queries
        if self.track_oracle_history:
            self.oracle_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "batch_size": n_queries,
                    "rewards": [float(r) for r in rewards],
                }
            )

    def reset_oracle_tracking(self) -> None:
        """Reset oracle call counters and call history."""
        self.oracle_calls = 0
        self.oracle_history = []
