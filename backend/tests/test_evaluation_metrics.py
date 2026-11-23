"""Tests for ranking metrics."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_hits() -> None:
    relevance = {"a": 2, "b": 1, "c": 0}

    assert recall_at_k(["a", "c"], relevance, 5) == 0.5
    assert recall_at_k(["c"], relevance, 5) == 0.0
    assert recall_at_k(["a", "b"], relevance, 5) == 1.0


def test_recall_respects_k() -> None:
    relevance = {"a": 1}

    assert recall_at_k(["x", "a"], relevance, 1) == 0.0
    assert recall_at_k(["x", "a"], relevance, 2) == 1.0


def test_recall_is_undefined_without_relevant_judgments() -> None:
    assert recall_at_k(["a"], {"a": 0}, 5) is None
    assert recall_at_k(["a"], {}, 5) is None


def test_precision_at_k() -> None:
    relevance = {"a": 1, "b": 2}

    assert precision_at_k(["a", "x", "b"], relevance, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], relevance, 5) == pytest.approx(0.2)
    assert precision_at_k([], relevance, 5) == 0.0
    assert precision_at_k(["a"], {}, 5) is None


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "a"], {"a": 1}) == 0.5
    assert reciprocal_rank(["x"], {"a": 1}) == 0.0
    assert reciprocal_rank([], {"a": 1}) == 0.0
    assert reciprocal_rank(["x"], {}) is None


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 5) == pytest.approx(1.0)


def test_ndcg_penalizes_worse_ordering() -> None:
    relevance = {"a": 2, "b": 1}

    assert ndcg_at_k(["b", "a"], relevance, 5) < ndcg_at_k(["a", "b"], relevance, 5)


def test_ndcg_is_undefined_without_graded_relevance() -> None:
    assert ndcg_at_k(["a"], {"a": 0}, 5) is None


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 50) == 2.5
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile([], 50) == 0.0


def test_percentile_rejects_invalid_p() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_mean_ignores_undefined_values() -> None:
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None
