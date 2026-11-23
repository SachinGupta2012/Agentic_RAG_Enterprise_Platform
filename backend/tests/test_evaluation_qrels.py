"""Tests for qrels parsing, validation and coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.jsonl import write_jsonl
from app.evaluation.qrels import (
    QrelsRecord,
    QrelsValidationError,
    judgment_coverage,
    load_qrels,
    qrels_by_query,
    qrels_fingerprint,
    write_qrels,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query_id": "dev-q0001",
        "doc_id": "doc-a",
        "relevance": 2,
        "judge_provider": "groq",
        "judge_model": "test-model",
        "judgment_source": "llm",
        "judgment_timestamp": "2026-01-01T00:00:00+00:00",
        "reason": "same issue",
        "prompt_version": "judge-v1",
    }
    payload.update(overrides)
    return payload


def _record(**overrides: object) -> QrelsRecord:
    return QrelsRecord.model_validate(_payload(**overrides))


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    records = [_record(), _record(doc_id="doc-b", relevance=0)]

    assert write_qrels(path, records) == 2

    loaded = load_qrels(path)
    assert [record.doc_id for record in loaded] == ["doc-a", "doc-b"]
    assert loaded[1].relevance == 0


def test_duplicate_pair_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    write_jsonl(
        path,
        [
            _record().model_dump(),
            _record(relevance=0).model_dump(),
        ],
    )

    with pytest.raises(QrelsValidationError, match="duplicate"):
        load_qrels(path)


def test_invalid_grade_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    write_jsonl(path, [_payload(relevance=3)])

    with pytest.raises(QrelsValidationError, match="invalid qrels record"):
        load_qrels(path)


def test_invalid_timestamp_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    write_jsonl(path, [_payload(judgment_timestamp="yesterday")])

    with pytest.raises(QrelsValidationError):
        load_qrels(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QrelsValidationError, match="not found"):
        load_qrels(tmp_path / "missing.jsonl")


def test_qrels_by_query_groups_by_query() -> None:
    grouped = qrels_by_query(
        [_record(), _record(doc_id="doc-b", relevance=1), _record(query_id="dev-q0002")]
    )

    assert grouped["dev-q0001"] == {"doc-a": 2, "doc-b": 1}
    assert grouped["dev-q0002"] == {"doc-a": 2}


def test_judgment_coverage_counts_retrieved_documents() -> None:
    ranked = {"dev-q0001": ["doc-a", "doc-b"], "dev-q0002": ["doc-c"]}
    records = [_record(doc_id="doc-a"), _record(doc_id="doc-b")]

    assert judgment_coverage(ranked, records) == pytest.approx(2 / 3)


def test_judgment_coverage_is_none_without_retrievals() -> None:
    assert judgment_coverage({}, [_record()]) is None


def test_fingerprint_summarizes_qrels() -> None:
    fingerprint = qrels_fingerprint([_record(), _record(doc_id="doc-b", judgment_source="human")])

    assert fingerprint["judgments"] == 2
    assert fingerprint["queries"] == 1
    assert fingerprint["documents"] == 2
    assert fingerprint["sources"] == ["human", "llm"]
