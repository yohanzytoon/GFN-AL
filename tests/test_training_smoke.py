from __future__ import annotations

from pathlib import Path

from training.train_active import run_active_learning
from training.train_baseline import run_supervised_baseline


def _base_config(seed: int = 0):
    return {
        "seed": seed,
        "device": "cpu",
        "env": {"max_length": 7},
        "oracle": {"budget": 40, "vocabulary_check": False},
        "baseline": {
            "num_queries": 40,
            "train_fraction": 0.8,
            "hidden_dim": 64,
            "n_layers": 1,
            "dropout": 0.0,
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 16,
        },
        "active": {
            "initial_size": 10,
            "batch_size": 5,
            "candidate_pool_size": 32,
            "max_rounds": 4,
            "surrogate_type": "gp",
            "acquisition": "ucb",
            "acquisition_beta": 1.5,
            "ei_xi": 0.01,
            "surrogate": {"fit_maxiter": 10},
        },
    }


def test_supervised_baseline_smoke(tmp_path: Path):
    cfg = _base_config(seed=0)
    result = run_supervised_baseline(cfg, output_dir=tmp_path / "baseline", logger=None)
    assert result["method"] == "supervised_baseline"
    assert result["oracle_queries"] == 40
    assert len(result["curve"]["queries"]) == 40


def test_active_learning_smoke(tmp_path: Path):
    cfg = _base_config(seed=1)
    result = run_active_learning(cfg, output_dir=tmp_path / "active", logger=None)
    assert result["method"] == "active_learning"
    assert result["oracle_queries"] <= 40
    assert result["best_score"] >= 0.0
