from __future__ import annotations

from pathlib import Path

from training.train_active import run_active_learning
from training.train_gflownet import run_oracle_gflownet
from training.train_hybrid_gfn_only import (
    _bootstrap_oracle_dataset,
    _sample_gfn_only_candidates,
    run_hybrid_gfn_only,
)


def _base_config(seed: int = 0):
    return {
        "seed": seed,
        "device": "cpu",
        "float_precision": 32,
        "gflownet_root": "../gflownet",
        "env": {"max_length": 7, "num_tokens": 27},
        "oracle": {"budget": 40, "vocabulary_check": False, "enforce_budget": True},
        "active": {
            "initial_size": 10,
            "batch_size": 5,
            "candidate_pool_size": 32,
            "max_rounds": 4,
            "sampling_strategy": "uniform",
            "candidate_unique": True,
            "min_length": 3,
            "diversity_weight": 0.0,
            "acquisition": {"name": "ucb", "beta": 1.5},
            "surrogate": {"type": "gp", "fit_maxiter": 10, "prefer_botorch": False},
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
            "max_rounds": 2,
            "sampling_strategy": "uniform",
            "initial_sampling_strategy": "uniform",
            "fallback_sampling_strategy": "uniform",
            "candidate_unique": True,
            "min_length": 3,
            "diversity_weight": 0.0,
            "plausibility_bonus_weight": 0.0,
            "acquisition": {"name": "ucb", "beta": 1.0, "beta_min": 0.0, "final_beta": 0.0},
            "surrogate": {"type": "gp", "fit_maxiter": 10, "prefer_botorch": False},
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
                "start_round": 0,
                "min_positive_count": 0,
                "sample_size": 8,
                "sample_batches": 1,
                "max_sample_batches": 2,
                "prediction_mode": "mean",
                "exploration_beta": 1.0,
                "reward_transform": "softplus",
                "policy": {"hidden_dim": 32, "n_layers": 1, "shared_backward": False},
                "evaluator": {"period": 10, "n": 4, "checkpoints_period": 10},
            },
        },
    }


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


def test_hybrid_gfn_only_smoke_with_fake_gflownet_backend(tmp_path: Path, monkeypatch):
    cfg = _base_config(seed=4)
    cfg["hybrid"]["gflownet"]["start_round"] = 0
    cfg["hybrid"]["gflownet"]["min_positive_count"] = 0
    cfg["hybrid"]["gflownet"]["sample_batches"] = 2
    cfg["hybrid"]["gflownet"]["max_sample_batches"] = 4

    def _fake_train(*args, **kwargs):
        return _FakeGFN(), None

    monkeypatch.setattr("training.train_hybrid_gfn_only._train_upstream_gflownet", _fake_train)
    monkeypatch.setattr(
        "training.train_hybrid_gfn_only.sample_gflownet_terminating_states",
        lambda gfn, n_samples: [[1, 2, 0, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]],
    )
    result = run_hybrid_gfn_only(cfg, output_dir=tmp_path / "hybrid_gfn_only", logger=None)
    assert result["method"] == "hybrid_gfn_only"
    assert result["candidate_sampling_strategy"] == "gflownet_only"
    assert result["oracle_queries"] <= 40


def test_hybrid_gfn_only_resamples_gfn_candidates_until_pool_fills(monkeypatch):
    batches = iter(
        [
            [[1, 2, 0, 0, 0, 0, 0]],
            [[2, 3, 0, 0, 0, 0, 0]],
            [[4, 5, 0, 0, 0, 0, 0]],
        ]
    )

    monkeypatch.setattr(
        "training.train_hybrid_gfn_only.sample_gflownet_terminating_states",
        lambda gfn, n_samples: next(batches),
    )

    candidates, breakdown = _sample_gfn_only_candidates(
        gfn=object(),
        seen_states=[],
        gflownet_sample_size=1,
        gflownet_sample_batches=1,
        gflownet_max_sample_batches=4,
        min_required=3,
    )

    assert breakdown["gflownet_sample_batches_used"] == 3
    assert candidates == [
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [4, 5, 0, 0, 0, 0, 0],
    ]


def test_hybrid_gfn_only_bootstraps_before_first_gfn(monkeypatch):
    class _BootstrapOracle:
        def __init__(self):
            self.remaining_budget = 6

        def __call__(self, states):
            import numpy as np
            import torch

            arr = np.asarray(states)
            self.remaining_budget -= int(arr.shape[0])
            scores = [1.0 if int(row[0]) == 3 else 0.0 for row in arr]
            return torch.tensor(scores, dtype=torch.float32)

    monkeypatch.setattr(
        "training.train_hybrid_gfn_only.sample_terminating_states",
        lambda *args, **kwargs: [
            [3, 4, 0, 0, 0, 0, 0],
            [5, 6, 0, 0, 0, 0, 0],
            [7, 8, 0, 0, 0, 0, 0],
        ],
    )

    states, scores, logs, bootstrap_rounds = _bootstrap_oracle_dataset(
        env=None,
        oracle=_BootstrapOracle(),
        initial_states=[[1, 2, 0, 0, 0, 0, 0]],
        initial_scores=[0.0],
        bootstrap_pool_size=8,
        bootstrap_batch_size=2,
        bootstrap_sampling_strategy="frequency",
        min_length=3,
        candidate_unique=True,
        final_oracle_reserve=0,
        target_min_rounds=1,
        target_positive_count=1,
        seed=0,
        gflownet_root=None,
        max_bootstrap_queries=6,
    )

    assert bootstrap_rounds == 1
    assert len(logs) == 1
    assert len(states) == 3
    assert scores[-2:] == [1.0, 0.0]
