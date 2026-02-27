from __future__ import annotations

from utils.metrics import (
    aggregate_curves_with_ci,
    build_query_curve,
    diversity_entropy,
    paired_statistical_tests,
    running_best,
    running_topk_mean,
    search_quality_metrics,
    topk_edit_distance,
)


def test_running_best_and_topk():
    scores = [1.0, 3.0, 2.0, 5.0]
    assert running_best(scores).tolist() == [1.0, 3.0, 3.0, 5.0]
    topk = running_topk_mean(scores, k=2)
    assert topk[-1] == 4.0


def test_diversity_entropy_positive():
    states = [[1, 2, 0, 0], [2, 3, 0, 0], [3, 4, 0, 0]]
    assert diversity_entropy(states, pad_value=0) > 0


def test_build_query_curve_lengths_match():
    curve = build_query_curve([1.0, 2.0, 3.0])
    assert len(curve["queries"]) == len(curve["best"]) == len(curve["top10"]) == 3
    curve_regret = build_query_curve([1.0, 2.0, 3.0], optimum_score=4.0)
    assert "regret" in curve_regret


def test_paired_stats_contains_effect_size():
    stats = paired_statistical_tests([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
    assert "cohen_d_paired" in stats
    assert stats["n"] == 3


def test_topk_edit_distance_positive_for_distinct_sequences():
    states = [[1, 2, 3, 0], [1, 2, 4, 0], [5, 6, 7, 0]]
    scores = [1.0, 2.0, 3.0]
    assert topk_edit_distance(states, scores, k=3, pad_value=0) > 0.0


def test_search_quality_metrics_contains_expected_fields():
    states = [[1, 2, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0]]
    scores = [0.0, 2.0, 3.0]
    metrics = search_quality_metrics(scores=scores, states=states, oracle_queries=3, pad_value=0)
    assert metrics["best_score"] == 3.0
    assert "valid_word_ratio" in metrics
    assert "mode_coverage" in metrics


def test_curve_aggregation_shapes():
    curves = [
        {"queries": [1, 2, 3], "best": [1.0, 2.0, 3.0]},
        {"queries": [1, 2, 3], "best": [0.5, 2.5, 3.5]},
    ]
    stats = aggregate_curves_with_ci(curves, budget=3, key="best")
    assert stats["mean"].shape[0] == 3
    assert stats["lower"].shape[0] == 3
