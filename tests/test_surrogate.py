from __future__ import annotations

import numpy as np
import pytest

from surrogate.gp_model import BoTorchGPSurrogate
from surrogate.factory import build_surrogate


def test_gp_fit_predict_shapes():
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
    model = BoTorchGPSurrogate(
        max_length=4,
        num_tokens=27,
        device="cpu",
        fit_maxiter=5,
        prefer_botorch=False,
        max_train_points=10,
    )
    model.fit(states, scores)
    mean, std = model.predict(states, return_std=True)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert np.all(std >= 0.0)


def test_surrogate_factory_builds_gp():
    model = build_surrogate(
        {
            "type": "gp",
            "fit_maxiter": 5,
            "prefer_botorch": False,
            "max_train_points": 10,
        },
        max_length=4,
        num_tokens=27,
        device="cpu",
    )
    assert isinstance(model, BoTorchGPSurrogate)


def test_surrogate_factory_rejects_removed_ensemble():
    with pytest.raises(ValueError, match="Use 'gp'"):
        build_surrogate(
            {"type": "ensemble"},
            max_length=4,
            num_tokens=27,
            device="cpu",
        )
