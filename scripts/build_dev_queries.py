"""Select the deterministic dev-split query set used for offline evaluation.

Only ``tickets_dev.parquet`` is read. The eval split is never touched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.jsonl import sha256_file, write_json
from app.evaluation.queries import QuerySelectionError, select_dev_queries, write_dev_queries
from app.retrieval.ingestion import (
    IngestionValidationError,
    load_manifest,
    load_parquet,
    validate_checksum,
)

DEFAULT_COLUMNS = [
    "ticket_id",
    "customer_query",
    "category",
    "department",
    "priority",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("data/processed/tickets_dev.parquet"),
        help="Dev split Parquet file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/split_manifest.json"),
        help="Split manifest with SHA-256 checksums.",
    )
    parser.add_argument("--limit", type=int, default=150, help="Number of queries to select.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
        help="Query set output file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/eval/dev_queries_summary.json"),
        help="Summary output file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and write the dev query set."""

    args = parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        checksum = validate_checksum(args.dev, manifest, "dev")
        frame = load_parquet(args.dev, columns=DEFAULT_COLUMNS)
    except (IngestionValidationError, KeyError, ValueError) as exc:
        print(f"Query selection failed: {exc}", file=sys.stderr)
        return 1

    try:
        queries = select_dev_queries(frame, limit=args.limit, seed=args.seed)
    except QuerySelectionError as exc:
        print(f"Query selection failed: {exc}", file=sys.stderr)
        return 1

    write_dev_queries(args.output, queries)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file": str(args.dev),
        "source_sha256": checksum,
        "source_rows": len(frame),
        "queries": len(queries),
        "seed": args.seed,
        "limit": args.limit,
        "selection_method": (
            "dev customer_query only; normalized; deduplicated by normalized text; "
            "deterministic sample; sorted by ticket_id; stable dev-qNNNN ids"
        ),
        "eval_split_used": False,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    write_json(args.summary, summary)

    print(f"Wrote {len(queries)} queries to {args.output}")
    print(f"Summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
