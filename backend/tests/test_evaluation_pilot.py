"""Saved ranks must be reconstructed by rank, and partial queries excluded."""

import pytest

from app.evaluation.pilot import evaluate_pilot
from app.evaluation.qrels import QrelsRecord


def _inputs() -> tuple[list[dict], list[QrelsRecord]]:
    candidates = [
        {
            "doc_id": f"d{i}",
            "retrieved_by": [
                {"mode": mode, "rank": 10 - i if mode == "sparse" else i + 1}
                for mode in ("dense", "sparse", "hybrid")
            ],
        }
        for i in range(10)
    ]
    records = [
        QrelsRecord(
            query_id="q1",
            doc_id=f"d{i}",
            relevance=2 if i == 9 else 0,
            judge_provider="groq",
            judge_model="test-model",
            prompt_version="judge-v1",
            judgment_source="llm",
            judgment_timestamp="2026-11-23T00:00:00+00:00",
        )
        for i in range(10)
    ]
    return [
        {"query_id": "q1", "candidates": list(reversed(candidates))},
        {"query_id": "q2", "candidates": candidates},
    ], records


def test_pilot_excludes_partial_queries_and_uses_saved_mode_ranks() -> None:
    pool, records = _inputs()
    report = evaluate_pilot(pool, records, expected_queries=1)
    assert report["selected_query_ids"] == ["q1"]
    assert report["excluded_queries"] == 1
    dense, sparse, _ = report["modes"]
    assert dense["recall_at_5"] == 0
    assert dense["recall_at_10"] == 1
    assert dense["mrr"] == 0.1
    assert sparse["mrr"] == 1
    assert sparse["ndcg_at_5"] == 1
    assert sparse["precision_at_5"] == 0.2
    assert all(row["judgment_coverage"] == 1 for row in report["modes"])
    assert not any("latency" in key for row in report["modes"] for key in row)


def test_pilot_rejects_unexpected_scope_or_missing_rank() -> None:
    pool, records = _inputs()
    with pytest.raises(ValueError, match="Expected 23"):
        evaluate_pilot(pool, records)
    pool[0]["candidates"][0]["retrieved_by"][0]["rank"] = 1
    with pytest.raises(ValueError, match="unique contiguous ranks"):
        evaluate_pilot(pool, records, expected_queries=1)


def test_all_irrelevant_pilot_query_is_excluded_from_macro_average() -> None:
    pool, records = _inputs()
    for record in records:
        record.relevance = 0
    report = evaluate_pilot(pool, records, expected_queries=1)
    for row in report["modes"]:
        assert row["queries_evaluated"] == 0
        assert row["queries_without_relevant_judgments"] == 1
        assert row["recall_at_10"] is None
