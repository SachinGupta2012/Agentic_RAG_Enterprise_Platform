"""Evaluate dense, sparse and hybrid retrieval against shared LLM qrels.

All modes run on the same query set and are scored against the same judgments.
Metrics are pooled-relevance estimates from LLM judgments, not human ground
truth and not absolute corpus recall.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.evaluation.jsonl import sha256_file, write_json
from app.evaluation.qrels import load_qrels, qrels_by_query, qrels_fingerprint
from app.evaluation.queries import load_dev_queries
from app.evaluation.report import build_report
from app.evaluation.runner import run_queries
from app.retrieval.store import QdrantTicketStore

DEFAULT_MODES = "dense,sparse,hybrid"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
        help="Query set file.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/eval/qrels_llm_v1.jsonl"),
        help="Qrels file shared by every mode.",
    )
    parser.add_argument("--modes", default=DEFAULT_MODES, help="Comma-separated modes.")
    parser.add_argument("--top-k", type=int, default=10, help="Results per query.")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0 = all).")
    parser.add_argument("--collection", default=None, help="Override the collection.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/retrieval_evaluation.json"),
        help="Report output file.",
    )
    return parser.parse_args(argv)


def _print_table(modes: list[dict[str, Any]]) -> None:
    headers = [
        "mode",
        "R@5",
        "R@10",
        "P@5",
        "MRR",
        "nDCG@5",
        "p50 ms",
        "p95 ms",
        "cov",
    ]
    print(" | ".join(headers))
    for row in modes:
        values = [
            str(row["mode"]),
            _fmt(row["recall_at_5"]),
            _fmt(row["recall_at_10"]),
            _fmt(row["precision_at_5"]),
            _fmt(row["mrr"]),
            _fmt(row["ndcg_at_5"]),
            _fmt(row["p50_latency_ms"]),
            _fmt(row["p95_latency_ms"]),
            _fmt(row["judgment_coverage"]),
        ]
        print(" | ".join(values))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write a report."""

    args = parse_args(argv)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        print("At least one mode is required.", file=sys.stderr)
        return 1
    if args.top_k < 10:
        print(
            "Warning: --top-k is below 10, so Recall@10 is truncated.",
            file=sys.stderr,
        )

    try:
        queries = load_dev_queries(args.queries)
        records = load_qrels(args.qrels)
    except Exception as exc:
        print(f"Could not load evaluation inputs: {exc}", file=sys.stderr)
        return 1

    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("No queries to evaluate.", file=sys.stderr)
        return 1

    settings = get_settings()
    store = QdrantTicketStore(settings=settings, collection_name=args.collection)
    try:
        log = run_queries(store, queries, modes, args.top_k)
        collection = store.collection_name
    finally:
        store.close()

    relevance_by_query = qrels_by_query(records)
    fingerprint = qrels_fingerprint(records)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "collection": collection,
        "dense_model": settings.qdrant_dense_model,
        "sparse_model": settings.qdrant_sparse_model,
        "rrf_k": settings.retrieval_rrf_k,
        "top_k": args.top_k,
        "modes": modes,
        "query_file": str(args.queries),
        "query_file_sha256": sha256_file(args.queries),
        "qrels_file": str(args.qrels),
        "qrels_file_sha256": sha256_file(args.qrels),
        "queries_requested": len(queries),
        "qrels": fingerprint,
    }

    report = build_report(
        modes=modes,
        runs=log.runs,
        records=records,
        relevance_by_query=relevance_by_query,
        metadata=metadata,
        failures=log.failures,
    )
    write_json(args.output, report)

    print(report["caveat"])
    print(
        f"Queries: {len(queries)} | Judged documents: {fingerprint['documents']} | "
        f"Failures: {len(log.failures)}"
    )
    _print_table(report["modes"])
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
