"""Tests for manual verification sampling and agreement statistics."""

from __future__ import annotations

import pytest

from app.evaluation.qrels import QrelsRecord
from app.evaluation.verification import (
    agreement_report,
    cohen_kappa,
    extract_verification_pairs,
    stratified_sample,
)


def _record(index: int, grade: int) -> QrelsRecord:
    return QrelsRecord(
        query_id=f"dev-q{index:04d}",
        doc_id=f"doc-{index}",
        relevance=grade,
        judge_provider="groq",
        judge_model="test-model",
        judgment_source="llm",
        judgment_timestamp="2026-01-01T00:00:00+00:00",
        reason="reason",
        prompt_version="judge-v1",
    )


def test_stratified_sample_covers_all_grades() -> None:
    records = [_record(i, grade) for i, grade in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2])]

    sample = stratified_sample(records, size=6, seed=1)

    assert len(sample) == 6
    assert {record.relevance for record in sample} == {0, 1, 2}


def test_stratified_sample_is_deterministic() -> None:
    records = [_record(i, i % 3) for i in range(30)]

    first = [record.doc_id for record in stratified_sample(records, 10, seed=5)]
    second = [record.doc_id for record in stratified_sample(records, 10, seed=5)]

    assert first == second


def test_stratified_sample_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        stratified_sample([], 0)


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa([(0, 0), (1, 1), (2, 2)]) == pytest.approx(1.0)


def test_cohen_kappa_is_none_without_pairs() -> None:
    assert cohen_kappa([]) is None


def test_agreement_report_counts_exact_and_within_one() -> None:
    report = agreement_report([(0, 0), (1, 2), (2, 2), (0, 1)])

    assert report["verified_pairs"] == 4
    assert report["exact_agreement"] == pytest.approx(0.5)
    assert report["within_one_agreement"] == pytest.approx(1.0)
    assert report["confusion"]["human_1_llm_2"] == 1


def test_agreement_report_is_empty_when_pending() -> None:
    report = agreement_report([])

    assert report["verified_pairs"] == 0
    assert report["exact_agreement"] is None


def test_extract_pairs_skips_unreviewed_rows() -> None:
    rows = [
        {"human_relevance": 2, "llm_relevance": 2},
        {"human_relevance": None, "llm_relevance": 1},
        {"human_relevance": 0, "llm_relevance": 1},
    ]

    assert extract_verification_pairs(rows) == [(2, 2), (0, 1)]


def test_extract_pairs_rejects_invalid_human_grade() -> None:
    with pytest.raises(ValueError):
        extract_verification_pairs([{"human_relevance": 5, "llm_relevance": 1}])
