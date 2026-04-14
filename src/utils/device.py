"""Helpers for selecting a usable torch device."""

from __future__ import annotations

import torch


def mps_available() -> bool:
    """Whether Apple Metal Performance Shaders is available."""
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(device: str | torch.device) -> str:
    """Return a usable torch device string for this machine."""
    if isinstance(device, torch.device):
        requested = device.type
    else:
        requested = str(device).lower()

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if mps_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return requested

