"""Helpers for working with the upstream gflownet repository."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict


def resolve_gflownet_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve the upstream gflownet repository path."""
    candidates = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[3] / "gflownet",
            here.parents[2] / "../gflownet",
            Path.cwd().resolve().parent / "gflownet",
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "gflownet").exists() and (candidate / "config").exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve the upstream gflownet repository. "
        "Expected a checkout at ../gflownet or an explicit gflownet_root."
    )


def ensure_gflownet_on_path(gflownet_root: str | Path | None = None) -> Path:
    """Ensure the upstream repo is importable and return its root path."""
    root = resolve_gflownet_root(gflownet_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def compose_upstream_train_config(
    overrides: Iterable[str],
    *,
    gflownet_root: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    """Compose an upstream train config with local overrides."""
    root = ensure_gflownet_on_path(gflownet_root)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(root / "config"),
        version_base="1.1",
        job_name="gfn_active_learning",
    ):
        cfg = compose(config_name="train", overrides=list(overrides))

    if output_dir is not None:
        with open_dict(cfg):
            cfg.logger.logdir.path = str(Path(output_dir).resolve())
    return cfg


