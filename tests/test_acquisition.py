from __future__ import annotations

import numpy as np

from acquisition.ei import expected_improvement, select_ei
from acquisition.factory import select_acquisition_batch
from acquisition.thompson import select_thompson
from acquisition.ucb import select_ucb
from acquisition.uncertainty import select_uncertainty


def test_ucb_selects_highest_scores():
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.1, 0.1, 0.1, 0.1])
    idx = select_ucb(mean=mean, std=std, batch_size=2, beta=1.0)
    assert idx.tolist() == [3, 2]


def test_expected_improvement_positive_for_promising_points():
    mean = np.array([0.5, 1.5, 2.5])
    std = np.array([0.2, 0.3, 0.4])
    ei = expected_improvement(mean=mean, std=std, best_f=1.0, xi=0.0)
    assert ei.shape[0] == 3
    assert ei[-1] > ei[0]


def test_select_ei_prefers_best_candidate():
    mean = np.array([0.0, 1.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    idx = select_ei(mean=mean, std=std, batch_size=1, best_f=1.0, xi=0.0)
    assert idx.tolist() == [2]


def test_uncertainty_selects_most_uncertain():
    std = np.array([0.1, 0.5, 0.2])
    idx = select_uncertainty(std=std, batch_size=2)
    assert idx.tolist() == [1, 2]


def test_thompson_selects_highest_draws():
    draws = np.array([0.2, 1.1, -0.5, 3.0])
    idx = select_thompson(draws=draws, batch_size=2)
    assert idx.tolist() == [3, 1]


class _DummySurrogate:
    def sample(self, states, n_samples=1):
        assert states.shape[0] == 3
        return np.array([[0.1, 5.0, 1.0]])


def test_factory_dispatches_thompson():
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([1.0, 1.0, 1.0])
    states = np.zeros((3, 4), dtype=np.int64)
    idx = select_acquisition_batch(
        "thompson",
        mean=mean,
        std=std,
        batch_size=1,
        best_f=0.0,
        surrogate=_DummySurrogate(),
        candidate_states=states,
    )
    assert idx.tolist() == [1]
