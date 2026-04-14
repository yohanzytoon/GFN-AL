"""Factory helpers for surrogate construction and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from surrogate.ensemble_model import DeepEnsembleSurrogate
from surrogate.gp_model import BoTorchGPSurrogate


def build_surrogate(
    config: dict[str, Any],
    *,
    max_length: int,
    num_tokens: int,
    device: str,
):
    """Instantiate the configured surrogate model."""
    surrogate_type = str(config.get("type", "gp")).lower()
    common = {
        "max_length": int(max_length),
        "num_tokens": int(num_tokens),
        "device": device,
    }

    if surrogate_type == "gp":
        return BoTorchGPSurrogate(
            fit_maxiter=int(config.get("fit_maxiter", 80)),
            prefer_botorch=bool(config.get("prefer_botorch", True)),
            max_train_points=int(config.get("max_train_points", 500)),
            **common,
        )
    if surrogate_type in {"ensemble", "deep_ensemble"}:
        return DeepEnsembleSurrogate(
            hidden_dim=int(config.get("hidden_dim", 256)),
            n_layers=int(config.get("n_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            ensemble_size=int(config.get("ensemble_size", 5)),
            epochs=int(config.get("epochs", 75)),
            batch_size=int(config.get("batch_size", 64)),
            lr=float(config.get("lr", 1e-3)),
            **common,
        )
    raise ValueError(
        f"Unsupported surrogate type: {surrogate_type}. "
        "Use one of {'gp', 'ensemble'}."
    )


def load_surrogate_checkpoint(path: str | Path, device: str = "cpu"):
    """Load a surrogate checkpoint without knowing its backend in advance."""
    payload = torch.load(path, map_location=device, weights_only=False)
    surrogate_type = str(payload.get("surrogate_type", "gp")).lower()
    if surrogate_type == "gp":
        return BoTorchGPSurrogate.load(path, device=device)
    if surrogate_type in {"ensemble", "deep_ensemble"}:
        return DeepEnsembleSurrogate.load(path, device=device)
    raise ValueError(f"Unsupported surrogate checkpoint type: {surrogate_type}")
