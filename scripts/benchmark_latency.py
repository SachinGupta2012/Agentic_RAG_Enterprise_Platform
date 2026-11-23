"""Benchmark retrieval latency per mode.

Methodology:
- the same fixed query set is used for every mode;
- ``--warmup`` requests per mode are discarded before measuring;
- each measurement is a full client request, so it includes Qdrant Cloud
  Inference (query embedding) and network time, which cannot be separated
  client-side;
- P50/P95 use linear interpolation over the measured samples;
- LLM judging time is never part of this benchmark.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.evaluation.jsonl import sha256_file, write_json
from app.evaluation.metrics import percentile
from app.evaluation.queries import DevQuery, load_dev_queries
from app.evaluation.runner import run_query
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
    parser.add_argument("--modes", default=DEFAULT_MODES, help="Comma-separated modes.")
    parser.add_argument("--top-k", type=int, default=10, help="Results per query.")
    parser.add_argument("--warmup", type=int, default=5, help="Discarded warm-up requests.")
    parser.add_argument("--repeats", type=int, default=30, help="Measured requests per mode.")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0 = all).")
    parser.add_argument("--collection", default=None, help="Override the collection.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/latency_benchmark.json"),
        help="Benchmark report output.",
    )
    return parser.parse_args(argv)


def _benchmark_mode(
    store: QdrantTicketStore,
    queries: list[DevQuery],
    mode: str,
    top_k: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    failures = 0
    for index in range(warmup):
        run = run_query(
            store,
            queries[index % len(queries)].query_id,
            queries[index % len(queries)].text,
            mode,
            top_k,
        )
        if run.error:
            failures += 1

    wall_samples: list[float] = []
    reported_samples: list[float] = []
    empty = 0
    for index in range(repeats):
        query = queries[index % len(queries)]
        run = run_query(store, query.query_id, query.text, mode, top_k)
        if run.error:
            failures += 1
            continue
        if run.empty:
            empty += 1
        wall_samples.append(run.wall_ms)
        reported_samples.append(run.latency_ms)

    return {
        "mode": mode,
        "requests_measured": len(wall_samples),
        "warmup_discarded": warmup,
        "failures": failures,
        "empty_results": empty,
        "p50_ms": round(percentile(wall_samples, 50), 2),
        "p95_ms": round(percentile(wall_samples, 95), 2),
        "mean_ms": round(sum(wall_samples) / len(wall_samples), 2) if wall_samples else None,
        "min_ms": round(min(wall_samples), 2) if wall_samples else None,
        "max_ms": round(max(wall_samples), 2) if wall_samples else None,
        "p50_qdrant_reported_ms": round(percentile(reported_samples, 50), 2),
        "p95_qdrant_reported_ms": round(percentile(reported_samples, 95), 2),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the latency benchmark."""

    args = parse_args(argv)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        print("At least one mode is required.", file=sys.stderr)
        return 1
    if args.repeats < 1:
        print("--repeats must be at least 1", file=sys.stderr)
        return 1

    try:
        queries = load_dev_queries(args.queries)
    except Exception as exc:
        print(f"Could not load queries: {exc}", file=sys.stderr)
        return 1
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("No queries to benchmark.", file=sys.stderr)
        return 1

    settings = get_settings()
    store = QdrantTicketStore(settings=settings, collection_name=args.collection)
    try:
        results = [
            _benchmark_mode(store, queries, mode, args.top_k, args.warmup, args.repeats)
            for mode in modes
        ]
        collection = store.collection_name
    finally:
        store.close()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "warmup_requests_discarded": args.warmup,
            "requests_per_mode": args.repeats,
            "same_query_set_for_all_modes": True,
            "latency_scope": (
                "client-observed full request, including Qdrant Cloud Inference for query "
                "embedding and network time; server-side inference cannot be separated "
                "client-side"
            ),
            "percentile_method": "linear interpolation",
            "llm_judging_excluded": True,
        },
        "collection": collection,
        "dense_model": settings.qdrant_dense_model,
        "sparse_model": settings.qdrant_sparse_model,
        "rrf_k": settings.retrieval_rrf_k,
        "top_k": args.top_k,
        "query_file": str(args.queries),
        "query_file_sha256": sha256_file(args.queries),
        "queries_available": len(queries),
        "modes": results,
    }
    write_json(args.output, report)

    print("mode | p50 ms | p95 ms | measured | failures")
    for row in results:
        print(
            f"{row['mode']} | {row['p50_ms']} | {row['p95_ms']} | "
            f"{row['requests_measured']} | {row['failures']}"
        )
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
