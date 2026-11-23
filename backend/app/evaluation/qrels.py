"""Versioned relevance judgments (qrels) for offline retrieval evaluation.

A qrel is independent of retrieval mode: one ``(query_id, doc_id)`` pair carries
one shared judgment that every mode is scored against.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.evaluation.jsonl import read_jsonl, write_jsonl

JudgmentSource = Literal["llm", "human"]


class QrelsValidationError(ValueError):
    """Raised when a qrels file is malformed or internally inconsistent."""


class QrelsRecord(BaseModel):
    """A single graded relevance judgment."""

    query_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=2)
    judge_provider: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judgment_source: JudgmentSource
    judgment_timestamp: str = Field(min_length=1)
    reason: str = ""
    prompt_version: str = Field(min_length=1)

    @field_validator("judgment_timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("judgment_timestamp must be ISO-8601") from exc
        return value


def utc_now_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""

    return datetime.now(UTC).isoformat()


def load_qrels(path: Path) -> list[QrelsRecord]:
    """Load and validate a qrels JSONL file."""

    if not path.is_file():
        raise QrelsValidationError(f"Qrels file not found: {path}")

    records: list[QrelsRecord] = []
    seen: dict[tuple[str, str], QrelsRecord] = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        try:
            record = QrelsRecord.model_validate(row)
        except Exception as exc:  # pydantic ValidationError
            raise QrelsValidationError(f"{path.name}:{index}: invalid qrels record: {exc}") from exc
        key = (record.query_id, record.doc_id)
        if key in seen:
            raise QrelsValidationError(
                f"{path.name}:{index}: duplicate judgment for {key[0]}/{key[1]}"
            )
        seen[key] = record
        records.append(record)
    return records


def write_qrels(path: Path, records: Iterable[QrelsRecord], *, append: bool = False) -> int:
    """Write qrels records as JSONL."""

    return write_jsonl(path, (record.model_dump() for record in records), append=append)


def qrels_by_query(records: Sequence[QrelsRecord]) -> dict[str, dict[str, int]]:
    """Group judgments into ``{query_id: {doc_id: relevance}}``."""

    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        grouped.setdefault(record.query_id, {})[record.doc_id] = record.relevance
    return grouped


def judgment_coverage(
    ranked_by_query: Mapping[str, Sequence[str]],
    records: Sequence[QrelsRecord],
) -> float | None:
    """Return the share of retrieved documents that carry a judgment.

    A coverage below ``1.0`` means some retrieved documents were never judged, so
    ranking metrics treat them as non-relevant. This value makes that assumption
    explicit in every report.
    """

    judged = {(record.query_id, record.doc_id) for record in records}
    retrieved = 0
    covered = 0
    for query_id, ranked in ranked_by_query.items():
        for doc_id in ranked:
            retrieved += 1
            if (query_id, doc_id) in judged:
                covered += 1
    if retrieved == 0:
        return None
    return covered / retrieved


def qrels_fingerprint(records: Sequence[QrelsRecord]) -> dict[str, object]:
    """Return a compact provenance summary for reports."""

    return {
        "judgments": len(records),
        "queries": len({record.query_id for record in records}),
        "documents": len({record.doc_id for record in records}),
        "sources": sorted({record.judgment_source for record in records}),
        "prompt_versions": sorted({record.prompt_version for record in records}),
        "providers": sorted({record.judge_provider for record in records}),
        "models": sorted({record.judge_model for record in records}),
    }
