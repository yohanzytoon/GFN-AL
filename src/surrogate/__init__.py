"""Surrogate model registry and factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from surrogate.bnn import DropoutBNNSurrogate
from surrogate.deep_ensemble import DeepEnsembleSurrogate
from surrogate.gp_model import BoTorchGPSurrogate

SURROGATE_REGISTRY = {
    "gp": BoTorchGPSurrogate,
    "deep_ensemble": DeepEnsembleSurrogate,
    "bnn": DropoutBNNSurrogate,
}


def build_surrogate(surrogate_type: str, **kwargs: Any):
    """Instantiate surrogate by name."""
    if surrogate_type not in SURROGATE_REGISTRY:
        raise KeyError(
            f"Unknown surrogate '{surrogate_type}'. Available: {list(SURROGATE_REGISTRY)}"
        )
    return SURROGATE_REGISTRY[surrogate_type](**kwargs)


def load_surrogate(path: str | Path, device: str = "cpu"):
    """Load surrogate checkpoint from disk."""
    payload = torch.load(path, map_location=device, weights_only=False)
    surrogate_type = payload.get("surrogate_type")
    if surrogate_type not in SURROGATE_REGISTRY:
        raise KeyError(
            f"Unknown surrogate type in checkpoint '{surrogate_type}'. "
            f"Available: {list(SURROGATE_REGISTRY)}"
        )
    return SURROGATE_REGISTRY[surrogate_type].load(path=path, device=device)
