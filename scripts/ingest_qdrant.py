"""Ingest the retrieval split into Qdrant Cloud.

Usage:
    uv run --extra retrieval python scripts/ingest_qdrant.py --recreate

The script refuses to ingest if the retrieval file does not match the manifest,
if any row is unresolved or non-English, or if retrieval identifiers overlap the
dev/eval splits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.retrieval.ingestion import (
    IngestionValidationError,
    aggregate_redactions,
    build_report,
    load_manifest,
    load_parquet,
    prepare_documents,
    validate_checksum,
    validate_retrieval_frame,
    validate_split_isolation,
    write_report,
)
from app.retrieval.store import QdrantTicketStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tickets_retrieval.parquet"),
        help="Retrieval split Parquet file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/split_manifest.json"),
        help="Split manifest with SHA-256 checksums.",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("data/processed/tickets_dev.parquet"),
        help="Dev split, used only for an overlap check.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/processed/tickets_eval.parquet"),
        help="Eval split, used only for an overlap check.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override the Qdrant collection name.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Documents per upsert batch (default: settings value).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N documents. Use for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and prepare documents without connecting to Qdrant.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/qdrant_ingestion_report.json"),
        help="Where to write the ingestion report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion workflow."""

    args = parse_args(argv)
    settings = get_settings()

    try:
        manifest = load_manifest(args.manifest)
        source_sha256 = validate_checksum(args.input, manifest, "retrieval")
        retrieval_entry = manifest["files"]["retrieval"]
        expected_rows = int(retrieval_entry["rows"])

        frame = load_parquet(args.input)
        validate_retrieval_frame(frame, expected_rows)

        retrieval_ids = {str(value) for value in frame["ticket_id"].tolist()}
        validate_split_isolation(retrieval_ids, args.dev, args.eval)

        documents = prepare_documents(frame, limit=args.limit)
    except (IngestionValidationError, KeyError, ValueError) as exc:
        print(f"Ingestion validation failed: {exc}", file=sys.stderr)
        return 1

    redactions = aggregate_redactions(documents)
    print(f"Prepared {len(documents)} documents.")
    print(f"PII redactions: {json.dumps(redactions)}")
    if args.limit:
        print(
            f"WARNING: --limit is active; only {len(documents)} of {expected_rows} "
            "documents will be written."
        )

    if args.dry_run:
        print("Dry run complete; no Qdrant writes were made.")
        return 0

    store = QdrantTicketStore(settings=settings, collection_name=args.collection)
    started = time.perf_counter()
    try:
        store.ensure_collection(recreate=args.recreate)
        written = store.upsert_documents(documents, batch_size=args.batch_size)
        collection_count = store.count()
    except Exception as exc:
        print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    duration_seconds = time.perf_counter() - started
    warnings: list[str] = []
    if args.limit:
        warnings.append("Ingestion used --limit; the collection is partial.")
    if collection_count != len(documents):
        warnings.append(
            f"Collection count {collection_count} does not match prepared documents "
            f"{len(documents)}."
        )

    report = build_report(
        collection=store.collection_name,
        source_path=args.input,
        source_sha256=source_sha256,
        source_rows=len(frame),
        documents=documents,
        documents_upserted=written,
        collection_count=collection_count,
        dense_model=settings.qdrant_dense_model,
        sparse_model=settings.qdrant_sparse_model,
        batch_size=args.batch_size or settings.retrieval_batch_size,
        duration_seconds=duration_seconds,
        warnings=warnings,
    )
    write_report(args.report, report)
    print(json.dumps(report.as_dict(), indent=2))
    print(f"Report written to {args.report}")

    return 0 if collection_count == len(documents) else 2


if __name__ == "__main__":
    raise SystemExit(main())
