from __future__ import annotations

from pathlib import Path

from training.dataset import generate_random_dataset
from training.train_active import run_active_learning
from training.train_baseline import run_supervised_baseline


def _base_config(seed: int = 0):
    return {
        "seed": seed,
        "device": "cpu",
        "env": {"max_length": 7},
        "oracle": {"budget": 40, "vocabulary_check": False},
        "dataset": {"path": None, "num_queries": 40, "unique": False},
        "baseline": {
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
            "acquisition_beta": 1.5,
            "surrogate": {"fit_maxiter": 10},
        },
    }


def test_dataset_generation_smoke(tmp_path: Path):
    cfg = _base_config(seed=0)
    result = generate_random_dataset(cfg, output_dir=tmp_path / "dataset", logger=None)
    assert result["method"] == "dataset_generation"
    assert result["num_samples"] == 40
    assert Path(result["dataset_path"]).exists()


def test_supervised_baseline_smoke(tmp_path: Path):
    cfg = _base_config(seed=0)
    dataset_result = generate_random_dataset(cfg, output_dir=tmp_path / "dataset", logger=None)
    cfg["dataset"]["path"] = dataset_result["dataset_path"]
    result = run_supervised_baseline(cfg, output_dir=tmp_path / "baseline", logger=None)
    assert result["method"] == "supervised_baseline"
    assert result["oracle_queries"] == 40
    assert result["dataset_path"] == dataset_result["dataset_path"]


def test_active_learning_smoke(tmp_path: Path):
    cfg = _base_config(seed=1)
    result = run_active_learning(cfg, output_dir=tmp_path / "active", logger=None)
    assert result["method"] == "active_learning"
    assert result["oracle_queries"] <= 40
    assert result["best_score"] >= 0.0
