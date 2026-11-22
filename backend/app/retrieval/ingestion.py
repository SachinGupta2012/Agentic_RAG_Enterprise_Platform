"""Validation and preparation of the retrieval split for Qdrant ingestion.

The evaluation split is never loaded for indexing. The ingestion script reads
only `tickets_retrieval.parquet` and uses the dev/eval files solely to prove
that no identifier overlaps exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.retrieval.pii import RedactionCounts, redact_pii
from app.retrieval.store import TicketDocument

REQUIRED_COLUMNS = (
    "ticket_id",
    "source_ticket_id",
    "subject",
    "customer_query",
    "agent_response",
    "category",
    "department",
    "priority",
    "language",
    "tags",
    "resolution_status",
    "source_dataset",
    "metadata",
)
DENSE_TEXT_LIMIT = 2000
SPARSE_TEXT_LIMIT = 8000


class IngestionValidationError(ValueError):
    """The retrieval data failed a safety or schema check."""


@dataclass(frozen=True)
class IngestionReport:
    """Auditable summary of one ingestion run."""

    generated_at: str
    collection: str
    source_file: str
    source_sha256: str
    source_rows: int
    documents_prepared: int
    documents_upserted: int
    collection_count: int
    dense_model: str
    sparse_model: str
    batch_size: int
    pii_redactions: dict[str, int]
    duration_seconds: float
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "generated_at": self.generated_at,
            "collection": self.collection,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_rows": self.source_rows,
            "documents_prepared": self.documents_prepared,
            "documents_upserted": self.documents_upserted,
            "collection_count": self.collection_count,
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "batch_size": self.batch_size,
            "pii_redactions": self.pii_redactions,
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the split manifest."""

    if not path.is_file():
        raise IngestionValidationError(f"Split manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise IngestionValidationError("Split manifest must contain a JSON object.")
    return data


def validate_checksum(path: Path, manifest: Mapping[str, Any], split: str) -> str:
    """Verify a split file against the manifest checksum."""

    files = manifest.get("files")
    if not isinstance(files, Mapping) or split not in files:
        raise IngestionValidationError(f"Manifest has no entry for split {split!r}.")
    entry = files[split]
    if not isinstance(entry, Mapping):
        raise IngestionValidationError(f"Manifest entry for {split!r} is invalid.")

    expected = str(entry.get("sha256") or "")
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise IngestionValidationError(
            f"{path.name} checksum mismatch. Re-run the Phase 1 notebook and "
            "re-download the processed files."
        )
    return actual


def load_parquet(path: Path, columns: Sequence[str] | None = None) -> Any:
    """Load a Parquet file, importing pandas only when ingestion is run."""

    if not path.is_file():
        raise IngestionValidationError(f"Data file not found: {path}")
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise IngestionValidationError(
            "pandas is required for ingestion. Run with `uv sync --extra retrieval`."
        ) from exc

    return pd.read_parquet(path, columns=list(columns) if columns else None)


def validate_retrieval_frame(frame: Any, expected_rows: int | None = None) -> None:
    """Validate the retrieval split before any text is sent to Qdrant."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise IngestionValidationError(f"Missing required columns: {', '.join(missing)}")

    if expected_rows is not None and len(frame) != expected_rows:
        raise IngestionValidationError(
            f"Expected {expected_rows} retrieval rows, found {len(frame)}."
        )

    if frame.empty:
        raise IngestionValidationError("The retrieval split is empty.")

    for column in (
        "ticket_id",
        "source_ticket_id",
        "subject",
        "customer_query",
        "agent_response",
        "category",
        "department",
        "priority",
        "language",
        "resolution_status",
        "source_dataset",
    ):
        if frame[column].isna().any():
            raise IngestionValidationError(f"Column {column!r} contains null values.")

    if frame["ticket_id"].duplicated().any():
        raise IngestionValidationError("ticket_id values must be unique.")

    for value in frame["ticket_id"]:
        try:
            uuid.UUID(str(value))
        except (ValueError, TypeError) as exc:
            raise IngestionValidationError(f"Invalid ticket_id: {value!r}") from exc

    if not (frame["resolution_status"] == "resolved").all():
        raise IngestionValidationError(
            "Only resolved historical tickets may be ingested into the knowledge base."
        )

    if not (frame["language"] == "en").all():
        raise IngestionValidationError("The retrieval corpus must be English-only for this phase.")

    if frame["source_dataset"].isna().any():
        raise IngestionValidationError("source_dataset must be present on every row.")


def load_split_ids(path: Path) -> set[str]:
    """Load only the ticket identifiers from a split file."""

    frame = load_parquet(path, columns=["ticket_id"])
    return {str(value) for value in frame["ticket_id"].tolist()}


def validate_split_isolation(
    retrieval_ids: set[str],
    dev_path: Path,
    eval_path: Path,
) -> None:
    """Fail if retrieval shares any identifier with dev or eval."""

    for label, path in (("dev", dev_path), ("eval", eval_path)):
        if not path.is_file():
            continue
        overlap = retrieval_ids & load_split_ids(path)
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise IngestionValidationError(
                f"Retrieval split overlaps the {label} split ({len(overlap)} ids): {examples}"
            )


def parse_tags(value: Any) -> tuple[str, ...]:
    """Parse the JSON tags column into a tuple of strings."""

    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        raise IngestionValidationError("tags must be a JSON list.")
    return tuple(str(tag).strip() for tag in parsed if str(tag).strip())


def parse_metadata(value: Any) -> dict[str, Any]:
    """Parse the JSON metadata column."""

    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise IngestionValidationError("metadata must be a JSON object.")
    return parsed


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{cut}..."


def _source_row_index(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("source_row_index")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_to_document(row: Mapping[str, Any]) -> TicketDocument:
    """Redact one row and build the text used for dense and sparse vectors."""

    subject = str(row["subject"]).strip()
    customer_query = str(row["customer_query"]).strip()
    agent_response = str(row["agent_response"]).strip()

    subject, subject_counts = redact_pii(subject)
    customer_query, query_counts = redact_pii(customer_query)
    agent_response, answer_counts = redact_pii(agent_response)
    redactions: RedactionCounts = subject_counts.merge(query_counts).merge(answer_counts)

    metadata = parse_metadata(row.get("metadata"))
    source_version_value = metadata.get("source_version")
    source_version = str(source_version_value).strip() if source_version_value is not None else None
    if not source_version:
        source_version = None

    dense_text = _truncate(
        f"Subject: {subject}\n\nCustomer issue:\n{customer_query}",
        DENSE_TEXT_LIMIT,
    )
    sparse_text = _truncate(
        f"Subject: {subject}\n\nCustomer issue:\n{customer_query}\n\nResolution:\n{agent_response}",
        SPARSE_TEXT_LIMIT,
    )

    return TicketDocument(
        ticket_id=str(row["ticket_id"]).strip(),
        source_ticket_id=str(row["source_ticket_id"]).strip(),
        subject=subject,
        customer_query=customer_query,
        agent_response=agent_response,
        category=str(row["category"]).strip(),
        department=str(row["department"]).strip(),
        priority=str(row["priority"]).strip(),
        language=str(row["language"]).strip().lower(),
        tags=parse_tags(row.get("tags")),
        resolution_status=str(row["resolution_status"]).strip(),
        source_dataset=str(row["source_dataset"]).strip(),
        source_row_index=_source_row_index(metadata),
        source_version=source_version,
        dense_text=dense_text,
        sparse_text=sparse_text,
        display_text=sparse_text,
        pii_redactions=redactions,
    )


def prepare_documents(frame: Any, limit: int | None = None) -> list[TicketDocument]:
    """Convert validated rows into redacted ingestion documents."""

    records = (
        frame.head(limit).to_dict(orient="records") if limit else frame.to_dict(orient="records")
    )
    return [record_to_document(record) for record in records]


def aggregate_redactions(documents: Sequence[TicketDocument]) -> dict[str, int]:
    """Sum PII redactions across all prepared documents."""

    total = RedactionCounts()
    for document in documents:
        total = total.merge(document.pii_redactions)
    return total.as_dict()


def build_report(
    *,
    collection: str,
    source_path: Path,
    source_sha256: str,
    source_rows: int,
    documents: Sequence[TicketDocument],
    documents_upserted: int,
    collection_count: int,
    dense_model: str,
    sparse_model: str,
    batch_size: int,
    duration_seconds: float,
    warnings: Sequence[str],
) -> IngestionReport:
    """Build an ingestion report without credentials or secrets."""

    return IngestionReport(
        generated_at=datetime.now(UTC).isoformat(),
        collection=collection,
        source_file=str(source_path),
        source_sha256=source_sha256,
        source_rows=source_rows,
        documents_prepared=len(documents),
        documents_upserted=documents_upserted,
        collection_count=collection_count,
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=batch_size,
        pii_redactions=aggregate_redactions(documents),
        duration_seconds=round(duration_seconds, 2),
        warnings=list(warnings),
    )


def write_report(path: Path, report: IngestionReport) -> None:
    """Write an ingestion report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
