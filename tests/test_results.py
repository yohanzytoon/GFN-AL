from __future__ import annotations

from utils.results import (
    aggregate_method_curves,
    aggregate_method_summaries,
    pairwise_metric_tests,
)


def test_aggregate_method_summaries():
    summaries = {
        "a": [{"best_score": 1.0}, {"best_score": 3.0}],
        "b": [{"best_score": 2.0}, {"best_score": 4.0}],
    }
    aggregated = aggregate_method_summaries(summaries, metrics=["best_score"])
    assert aggregated["a"]["best_score"]["mean"] == 2.0
    assert aggregated["b"]["best_score"]["max"] == 4.0


def test_aggregate_method_curves():
    summaries = {
        "a": [{"curve": {"queries": [1, 2], "best": [1.0, 2.0]}}],
        "b": [{"curve": {"queries": [1, 2], "best": [0.5, 3.0]}}],
    }
    curves = aggregate_method_curves(summaries, curve_key="best", budget=2)
    assert curves["a"]["mean"][-1] == 2.0
    assert curves["b"]["mean"][-1] == 3.0


def test_pairwise_metric_tests():
    summaries = {
        "ref": [{"best_score": 1.0}, {"best_score": 2.0}],
        "other": [{"best_score": 0.5}, {"best_score": 1.5}],
    }
    tests = pairwise_metric_tests(
        summaries,
        metric="best_score",
        reference_method="ref",
    )
    assert "ref_vs_other" in tests
