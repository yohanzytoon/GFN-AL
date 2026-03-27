"""Shared helpers for experiment training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from surrogate.factory import build_surrogate


def build_surrogate_from_config(
    surrogate_cfg: dict[str, Any],
    *,
    max_length: int,
    num_tokens: int,
    device: str,
):
    """Instantiate a surrogate backend from config."""
    return build_surrogate(
        surrogate_cfg,
        max_length=max_length,
        num_tokens=num_tokens,
        device=device,
    )


def filter_new_states(
    candidate_states: Iterable[Iterable[int]],
    seen_states: Iterable[Iterable[int]],
) -> list[list[int]]:
    """Remove candidate states already present in the seen set while preserving order."""
    seen = {tuple(int(x) for x in state) for state in seen_states}
    filtered: list[list[int]] = []
    added: set[tuple[int, ...]] = set()
    for state in candidate_states:
        key = tuple(int(x) for x in state)
        if key in seen or key in added:
            continue
        filtered.append(list(key))
        added.add(key)
    return filtered


def flatten_oracle_history(
    call_history: list[dict[str, Any]],
) -> tuple[list[list[int] | str], list[float]]:
    """Flatten oracle call history into per-query states and scores."""
    states: list[list[int] | str] = []
    scores: list[float] = []
    for record in call_history:
        record_states = record.get("states", [])
        record_scores = record.get("scores", [])
        if record_states and len(record_states) == len(record_scores):
            states.extend(record_states)
            scores.extend(float(score) for score in record_scores)
        else:
            scores.extend(float(score) for score in record_scores)
    return states, scores


def ensure_path(path: str | Path | None) -> Path | None:
    """Convert optional path-like values to Path."""
    if path is None:
        return None
    return Path(path)
