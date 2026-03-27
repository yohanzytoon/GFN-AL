from __future__ import annotations

from pathlib import Path

from training.dataset import generate_random_dataset
from training.train_active import run_active_learning
from training.train_baseline import run_supervised_baseline
from training.train_gflownet import run_oracle_gflownet
from training.train_hybrid import run_hybrid_gflownet_active


def _base_config(seed: int = 0):
    return {
        "seed": seed,
        "device": "cpu",
        "float_precision": 32,
        "gflownet_root": "../gflownet",
        "env": {"max_length": 7, "num_tokens": 27},
        "oracle": {"budget": 40, "vocabulary_check": False, "enforce_budget": True},
        "dataset": {
            "path": None,
            "num_queries": 40,
            "unique": True,
            "sampling_strategy": "uniform",
            "min_length": 3,
        },
        "baseline": {
            "train_fraction": 0.8,
            "hidden_dim": 64,
            "n_layers": 1,
            "dropout": 0.0,
            "positive_weight": 3.0,
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 16,
        },
        "active": {
            "initial_size": 10,
            "batch_size": 5,
            "candidate_pool_size": 32,
            "max_rounds": 4,
            "sampling_strategy": "uniform",
            "candidate_unique": True,
            "min_length": 3,
            "acquisition": {"name": "ei", "beta": 1.5, "xi": 0.0, "thompson_samples": 1},
            "surrogate": {"type": "ensemble", "hidden_dim": 32, "n_layers": 1, "ensemble_size": 2, "epochs": 3, "batch_size": 16, "lr": 1e-3},
        },
        "gflownet": {
            "n_train_steps": 10,
            "batch_size_forward": 4,
            "lr": 1e-4,
            "z_dim": 4,
            "lr_z_mult": 10.0,
            "random_action_prob": 0.0,
            "reward_min": 0.0,
            "do_clip_rewards": False,
            "evaluation_samples": 4,
            "policy": {"hidden_dim": 32, "n_layers": 1, "shared_backward": False},
            "evaluator": {"period": 10, "n": 4, "checkpoints_period": 10},
        },
        "hybrid": {
            "initial_size": 8,
            "batch_size": 4,
            "candidate_pool_size": 8,
            "fallback_random_pool_size": 8,
            "max_rounds": 2,
            "sampling_strategy": "uniform",
            "candidate_unique": True,
            "min_length": 3,
            "acquisition": {"name": "ucb", "beta": 1.0, "xi": 0.0, "thompson_samples": 1},
            "surrogate": {"type": "ensemble", "hidden_dim": 32, "n_layers": 1, "ensemble_size": 2, "epochs": 3, "batch_size": 16, "lr": 1e-3},
            "gflownet": {
                "n_train_steps": 10,
                "batch_size_forward": 4,
                "lr": 1e-4,
                "z_dim": 4,
                "lr_z_mult": 10.0,
                "random_action_prob": 0.0,
                "reward_min": 1e-4,
                "do_clip_rewards": True,
                "retrain_every": 1,
                "sample_size": 8,
                "prediction_mode": "mean",
                "exploration_beta": 1.0,
                "reward_transform": "softplus",
                "policy": {"hidden_dim": 32, "n_layers": 1, "shared_backward": False},
                "evaluator": {"period": 10, "n": 4, "checkpoints_period": 10},
            },
        },
    }


def test_dataset_generation_smoke(tmp_path: Path):
    cfg = _base_config(seed=0)
    result = generate_random_dataset(cfg, output_dir=tmp_path / "dataset", logger=None)
    assert result["method"] == "dataset_generation"
    assert result["num_samples"] > 0
    assert Path(result["dataset_path"]).exists()


def test_supervised_baseline_smoke(tmp_path: Path):
    cfg = _base_config(seed=0)
    dataset_result = generate_random_dataset(cfg, output_dir=tmp_path / "dataset", logger=None)
    cfg["dataset"]["path"] = dataset_result["dataset_path"]
    result = run_supervised_baseline(cfg, output_dir=tmp_path / "baseline", logger=None)
    assert result["method"] == "supervised_baseline"
    assert result["oracle_queries"] == result["num_samples"]
    assert result["dataset_path"] == dataset_result["dataset_path"]


def test_active_learning_smoke(tmp_path: Path):
    cfg = _base_config(seed=1)
    result = run_active_learning(cfg, output_dir=tmp_path / "active", logger=None)
    assert result["method"] == "active_learning"
    assert result["oracle_queries"] <= 40
    assert result["best_score"] >= 0.0


class _FakeLogger:
    def end(self):
        return None


class _FakeProxy:
    def __init__(self):
        self.call_count = 4
        self.remaining_budget = 4
        self.call_history = [
            {"states": [[1, 2, 0, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0]], "scores": [1.0, 2.0]},
            {"states": [[4, 5, 0, 0, 0, 0, 0], [6, 7, 0, 0, 0, 0, 0]], "scores": [3.0, 4.0]},
        ]

    def __call__(self, states):
        import numpy as np
        import torch

        arr = np.asarray(states)
        self.call_count += arr.shape[0]
        self.remaining_budget = max(self.remaining_budget - arr.shape[0], 0)
        return torch.tensor([1.0] * arr.shape[0], dtype=torch.float32)


class _FakeGFN:
    def __init__(self):
        self.proxy = _FakeProxy()
        self.logger = _FakeLogger()


def test_oracle_gflownet_summary_with_fake_backend(tmp_path: Path, monkeypatch):
    cfg = _base_config(seed=2)

    def _fake_train(*args, **kwargs):
        return _FakeGFN(), None

    monkeypatch.setattr("training.train_gflownet._train_upstream_gflownet", _fake_train)
    monkeypatch.setattr(
        "training.train_gflownet.sample_gflownet_terminating_states",
        lambda gfn, n_samples: [[1, 2, 0, 0, 0, 0, 0]] * n_samples,
    )
    result = run_oracle_gflownet(cfg, output_dir=tmp_path / "gflownet", logger=None)
    assert result["method"] == "gflownet_oracle"
    assert result["best_score"] == 4.0


def test_hybrid_smoke_with_fake_gflownet_backend(tmp_path: Path, monkeypatch):
    cfg = _base_config(seed=3)

    def _fake_train(*args, **kwargs):
        return _FakeGFN(), None

    monkeypatch.setattr("training.train_hybrid._train_upstream_gflownet", _fake_train)
    monkeypatch.setattr(
        "training.train_hybrid.sample_gflownet_terminating_states",
        lambda gfn, n_samples: [[1, 2, 0, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]],
    )
    result = run_hybrid_gflownet_active(cfg, output_dir=tmp_path / "hybrid", logger=None)
    assert result["method"] == "hybrid_gflownet_active"
    assert result["oracle_queries"] <= 40
