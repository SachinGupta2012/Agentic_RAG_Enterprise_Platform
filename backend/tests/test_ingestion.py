"""Tests for ingestion validation and document preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")

from app.retrieval.ingestion import (  # noqa: E402
    IngestionValidationError,
    aggregate_redactions,
    parse_metadata,
    parse_tags,
    prepare_documents,
    record_to_document,
    validate_checksum,
    validate_retrieval_frame,
)
from app.retrieval.pii import RedactionCounts  # noqa: E402


def _frame() -> object:
    return pandas.DataFrame(
        [
            {
                "ticket_id": "7294fd0a-a460-58e0-b3e2-07058d7fe95d",
                "source_ticket_id": "dataset:1",
                "subject": "Account Disruption",
                "customer_query": "The portal is offline since morning.",
                "agent_response": "Our team is investigating the outage.",
                "category": "Incident",
                "department": "Technical Support",
                "priority": "high",
                "language": "en",
                "tags": '["Account", "Outage"]',
                "resolution_status": "resolved",
                "source_dataset": "dataset",
                "metadata": (
                    '{"source_row_index": 1, "source_version": "51.0", "split": "retrieval"}'
                ),
            }
        ]
    )


def test_validate_retrieval_frame_accepts_valid_data() -> None:
    validate_retrieval_frame(_frame())


def test_validate_rejects_unresolved_ticket() -> None:
    frame = _frame()
    frame.loc[0, "resolution_status"] = "open"

    with pytest.raises(IngestionValidationError, match="resolved"):
        validate_retrieval_frame(frame)


def test_validate_rejects_duplicate_ticket_id() -> None:
    frame = pandas.concat([_frame(), _frame()], ignore_index=True)

    with pytest.raises(IngestionValidationError, match="unique"):
        validate_retrieval_frame(frame)


def test_validate_rejects_non_english_ticket() -> None:
    frame = _frame()
    frame.loc[0, "language"] = "de"

    with pytest.raises(IngestionValidationError, match="English"):
        validate_retrieval_frame(frame)


def test_validate_checksum(tmp_path: Path) -> None:
    path = tmp_path / "split.parquet"
    path.write_bytes(b"ticket-data")
    manifest = {
        "files": {
            "retrieval": {
                "sha256": hashlib.sha256(b"ticket-data").hexdigest(),
                "rows": 1,
            }
        }
    }

    assert (
        validate_checksum(path, manifest, "retrieval") == hashlib.sha256(b"ticket-data").hexdigest()
    )


def test_parse_tags_and_metadata() -> None:
    assert parse_tags('["Account", "Outage"]') == ("Account", "Outage")
    assert parse_tags("[]") == ()
    assert parse_metadata('{"source_row_index": 4}')["source_row_index"] == 4


def test_parse_tags_rejects_non_list() -> None:
    with pytest.raises(IngestionValidationError):
        parse_tags('{"tag": "Account"}')


def test_record_to_document_redacts_and_builds_text() -> None:
    record = _frame().iloc[0].to_dict()
    record["customer_query"] = "Email me at a.customer@example.com about the outage."

    document = record_to_document(record)

    assert "a.customer@example.com" not in document.customer_query
    assert document.pii_redactions.email == 1
    assert document.ticket_id == "7294fd0a-a460-58e0-b3e2-07058d7fe95d"
    assert document.source_row_index == 1
    assert document.source_version == "51.0"
    assert "Customer issue:" in document.dense_text
    assert "Resolution:" in document.sparse_text


def test_prepare_documents_respects_limit() -> None:
    frame = pandas.concat([_frame(), _frame()], ignore_index=True)

    documents = prepare_documents(frame, limit=1)

    assert len(documents) == 1


def test_aggregate_redactions_sums_counts() -> None:
    documents = [record_to_document({**_frame().iloc[0].to_dict(), "customer_query": "a@b.com"})]

    counts = aggregate_redactions(documents)

    assert counts["email"] == 1
    assert counts["phone"] == 0
    assert counts["url"] == 0


def test_default_redaction_counts_are_zero() -> None:
    assert RedactionCounts().total == 0
