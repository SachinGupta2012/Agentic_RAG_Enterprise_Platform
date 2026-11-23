"""Tests for evaluation report aggregation."""

from __future__ import annotations

import pytest

from app.evaluation.qrels import QrelsRecord
from app.evaluation.report import build_report, evaluate_mode
from app.evaluation.runner import ModeRun


def _record(query_id: str, doc_id: str, grade: int) -> QrelsRecord:
    return QrelsRecord(
        query_id=query_id,
        doc_id=doc_id,
        relevance=grade,
        judge_provider="groq",
        judge_model="test-model",
        judgment_source="llm",
        judgment_timestamp="2026-01-01T00:00:00+00:00",
        reason="reason",
        prompt_version="judge-v1",
    )


def _run(
    query_id: str,
    mode: str,
    docs: tuple[str, ...],
    wall_ms: float = 100.0,
    error: str | None = None,
) -> ModeRun:
    return ModeRun(
        query_id=query_id,
        mode=mode,
        ranked_doc_ids=docs,
        latency_ms=wall_ms,
        wall_ms=wall_ms,
        empty=not docs,
        error=error,
    )


def test_evaluate_mode_computes_expected_metrics() -> None:
    records = [
        _record("dev-q0001", "a", 2),
        _record("dev-q0001", "b", 0),
        _record("dev-q0003", "c", 1),
    ]
    relevance = {
        "dev-q0001": {"a": 2, "b": 0},
        "dev-q0003": {"c": 1},
    }
    runs = [
        _run("dev-q0001", "dense", ("a", "b")),
        _run("dev-q0002", "dense", ("x",)),
        _run("dev-q0003", "dense", ()),
    ]

    metrics = evaluate_mode("dense", runs, relevance, records)

    assert metrics["queries_evaluated"] == 2
    assert metrics["queries_without_relevant_judgments"] == 1
    assert metrics["empty_result_queries"] == 1
    assert metrics["recall_at_5"] == pytest.approx(0.5)
    assert metrics["precision_at_5"] == pytest.approx(0.1)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg_at_5"] == pytest.approx(0.5)
    assert metrics["judgment_coverage"] == pytest.approx(2 / 3, abs=1e-4)


def test_failed_runs_are_counted_and_excluded_from_latency() -> None:
    records = [_record("dev-q0001", "a", 2)]
    relevance = {"dev-q0001": {"a": 2}}
    runs = [
        _run("dev-q0001", "dense", ("a",), wall_ms=10.0),
        _run("dev-q0002", "dense", (), wall_ms=9999.0, error="RetrievalUnavailable"),
    ]

    metrics = evaluate_mode("dense", runs, relevance, records)

    assert metrics["failed_queries"] == 1
    assert metrics["p95_latency_ms"] == pytest.approx(10.0)


def test_evaluate_mode_requires_runs() -> None:
    with pytest.raises(ValueError):
        evaluate_mode("dense", [], {}, [])


def test_all_irrelevant_query_is_counted_as_excluded() -> None:
    records = [_record("q", "a", 0)]
    metrics = evaluate_mode("dense", [_run("q", "dense", ("a",))], {"q": {"a": 0}}, records)
    assert metrics["queries_evaluated"] == 0
    assert metrics["queries_without_relevant_judgments"] == 1
    assert metrics["recall_at_5"] is None


def test_build_report_includes_policy_and_caveat() -> None:
    records = [_record("dev-q0001", "a", 2)]
    relevance = {"dev-q0001": {"a": 2}}
    runs = [_run("dev-q0001", "hybrid", ("a",))]

    report = build_report(
        modes=["hybrid"],
        runs=runs,
        records=records,
        relevance_by_query=relevance,
        metadata={"collection": "test"},
    )

    assert report["metrics_label"] == "llm-judgment-based pooled estimate"
    assert "not absolute corpus recall" in report["caveat"]
    assert report["policy"]["relevant_grade_threshold"] == 1
    assert report["modes"][0]["mode"] == "hybrid"
    assert report["metadata"]["collection"] == "test"
