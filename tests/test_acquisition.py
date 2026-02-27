from __future__ import annotations

import numpy as np

from acquisition.ei import select_ei
from acquisition.thompson import select_thompson
from acquisition.ucb import select_ucb


def test_ucb_selects_highest_scores():
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    idx = select_ucb(mean=mean, std=std, batch_size=2, beta=1.0)
    assert idx.tolist() == [3, 2]


def test_ei_prefers_large_improvement():
    mean = np.array([1.0, 1.2, 2.5])
    std = np.array([0.2, 0.2, 0.2])
    idx = select_ei(mean=mean, std=std, best_observed=1.1, batch_size=1, xi=0.01)
    assert idx.tolist() == [2]


def test_thompson_returns_requested_batch_size():
    mean = np.array([0.0, 0.0, 0.0, 0.0])
    std = np.array([1.0, 1.0, 1.0, 1.0])
    idx = select_thompson(mean=mean, std=std, batch_size=3)
    assert len(idx) == 3
