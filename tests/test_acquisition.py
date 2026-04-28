from __future__ import annotations

import numpy as np

from acquisition.ucb import select_acquisition_batch, select_ucb


def test_ucb_selects_highest_scores():
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    idx = select_ucb(mean=mean, std=std, batch_size=2, beta=1.0)
    assert idx.tolist() == [3, 2]


def test_select_acquisition_batch_ucb():
    mean = np.array([0.0, 1.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    idx = select_acquisition_batch(
        mean=mean,
        std=std,
        batch_size=1,
        beta=1.0,
    )
    assert idx.tolist() == [2]


def test_select_acquisition_batch_uses_diversity_when_states_available():
    mean = np.array([10.0, 9.9, 9.8])
    std = np.zeros(3)
    states = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 2, 0],
            [9, 9, 9, 0],
        ],
        dtype=np.int64,
    )
    idx = select_acquisition_batch(
        mean=mean,
        std=std,
        batch_size=2,
        candidate_states=states,
        beta=1.0,
        diversity_weight=2.0,
    )
    assert idx.tolist() == [0, 2]
