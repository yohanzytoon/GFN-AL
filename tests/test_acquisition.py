from __future__ import annotations

import numpy as np

from acquisition.ucb import select_ucb


def test_ucb_selects_highest_scores():
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    idx = select_ucb(mean=mean, std=std, batch_size=2, beta=1.0)
    assert idx.tolist() == [3, 2]
