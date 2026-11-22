"""Proxy check for retrieval quality using the dev split.

This is not an evaluation benchmark. It measures metadata agreement and prints
examples for manual inspection. The eval split is never used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.retrieval.ingestion import (
    IngestionValidationError,
    load_manifest,
    load_parquet,
    validate_checksum,
)
from app.retrieval.schemas import RetrievalSearchRequest
from app.retrieval.store import QdrantTicketStore


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
    parser.add_argument("--limit", type=int, default=20, help="Number of dev queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Hits per query.")
    parser.add_argument("--collection", default=None, help="Override the collection.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/retrieval_dev_proxy_report.json"),
        help="Where to write the proxy report.",
    )
    return parser.parse_args(argv)


def _rate(matches: int, total: int) -> float:
    return round(matches / total, 4) if total else 0.0


def main(argv: list[str] | None = None) -> int:
    """Run the dev proxy check."""

    args = parse_args(argv)
    settings = get_settings()

    try:
        manifest = load_manifest(args.manifest)
        validate_checksum(args.dev, manifest, "dev")
        frame = load_parquet(args.dev)
    except (IngestionValidationError, KeyError, ValueError) as exc:
        print(f"Dev check failed: {exc}", file=sys.stderr)
        return 1

    sample = frame.sample(n=min(args.limit, len(frame)), random_state=42)
    store = QdrantTicketStore(settings=settings, collection_name=args.collection)

    results: list[dict[str, Any]] = []
    category_matches = 0
    department_matches = 0
    priority_matches = 0
    non_empty = 0
    total_latency = 0.0

    try:
        for _, row in sample.iterrows():
            request = RetrievalSearchRequest(
                query=str(row["customer_query"]),
                top_k=args.top_k,
            )
            response = store.search(request)
            hits = response.hits
            if hits:
                non_empty += 1
            category_matches += int(any(hit.category == str(row["category"]) for hit in hits))
            department_matches += int(any(hit.department == str(row["department"]) for hit in hits))
            priority_matches += int(any(hit.priority == str(row["priority"]) for hit in hits))
            total_latency += response.latency_ms
            results.append(
                {
                    "dev_ticket_id": str(row["ticket_id"]),
                    "query": str(row["customer_query"])[:300],
                    "expected": {
                        "category": str(row["category"]),
                        "department": str(row["department"]),
                        "priority": str(row["priority"]),
                    },
                    "latency_ms": response.latency_ms,
                    "hits": [
                        {
                            "ticket_id": hit.ticket_id,
                            "score": hit.score,
                            "subject": hit.subject,
                            "category": hit.category,
                            "department": hit.department,
                            "priority": hit.priority,
                            "snippet": hit.snippet,
                        }
                        for hit in hits
                    ],
                }
            )
    except Exception as exc:
        print(f"Dev check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    count = len(results)
    report = {
        "caveat": (
            "Proxy check only: metadata agreement is not relevance ground truth. "
            "Do not report these values as retrieval quality metrics."
        ),
        "collection": store.collection_name,
        "sample_size": count,
        "top_k": args.top_k,
        "proxy_metrics": {
            "non_empty_result_rate": _rate(non_empty, count),
            "category_match_at_k": _rate(category_matches, count),
            "department_match_at_k": _rate(department_matches, count),
            "priority_match_at_k": _rate(priority_matches, count),
            "mean_latency_ms": round(total_latency / count, 2) if count else 0.0,
        },
        "examples": results[:5],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Proxy check only; metadata agreement is not relevance ground truth.")
    print(json.dumps(report["proxy_metrics"], indent=2))
    print(f"Full report written to {args.report}")
    for example in report["examples"][:3]:
        print("\nQuery:", example["query"][:200])
        for hit in example["hits"][:3]:
            print(
                f"  - {hit['subject']} | {hit['category']} / {hit['department']} "
                f"/ {hit['priority']} | score={hit['score']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
