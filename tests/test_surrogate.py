from __future__ import annotations

import numpy as np

from surrogate.ensemble_model import DeepEnsembleSurrogate
from surrogate.factory import build_surrogate


def test_deep_ensemble_fit_predict_shapes():
    states = np.array(
        [
            [1, 2, 0, 0],
            [1, 3, 0, 0],
            [2, 3, 0, 0],
            [4, 5, 0, 0],
        ],
        dtype=np.int64,
    )
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    model = DeepEnsembleSurrogate(
        max_length=4,
        num_tokens=27,
        device="cpu",
        hidden_dim=32,
        n_layers=1,
        dropout=0.0,
        ensemble_size=2,
        epochs=5,
        batch_size=2,
        lr=1e-2,
    )
    model.fit(states, scores)
    mean, std = model.predict(states, return_std=True)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert np.all(std >= 0.0)


def test_surrogate_factory_builds_ensemble():
    model = build_surrogate(
        {
            "type": "ensemble",
            "hidden_dim": 16,
            "n_layers": 1,
            "ensemble_size": 2,
            "epochs": 3,
            "batch_size": 2,
        },
        max_length=4,
        num_tokens=27,
        device="cpu",
    )
    assert isinstance(model, DeepEnsembleSurrogate)
