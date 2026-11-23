"""Build the pooled candidate set used for LLM relevance judging.

Runs every query through dense, sparse and hybrid retrieval, merges the top-k
results per mode, deduplicates by document ID, and records which mode surfaced
each document. Read-only: no writes to Qdrant.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.evaluation.candidates import (
    MODES,
    PoolDocument,
    pool_candidates,
    pool_entry,
    write_pool_documents,
)
from app.evaluation.jsonl import sha256_file, write_json, write_jsonl
from app.evaluation.queries import DevQuery, load_dev_queries
from app.retrieval.schemas import RetrievalSearchRequest
from app.retrieval.store import QdrantTicketStore, RetrievalError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/dev_queries.jsonl"),
        help="Query set produced by build_dev_queries.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/candidate_pool.jsonl"),
        help="Candidate pool output file.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/eval/pool_documents.jsonl"),
        help="Pooled document text output file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/eval/candidate_pool_summary.json"),
        help="Pool summary output file.",
    )
    parser.add_argument("--depth", type=int, default=10, help="Candidates per mode.")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0 = all).")
    parser.add_argument("--collection", default=None, help="Override the collection.")
    return parser.parse_args(argv)


def _document_from_payload(doc_id: str, payload: dict[str, Any]) -> PoolDocument:
    tags_value = payload.get("tags") or []
    tags = tuple(str(tag) for tag in tags_value) if isinstance(tags_value, list) else ()
    return PoolDocument(
        doc_id=doc_id,
        subject=str(payload.get("subject") or ""),
        customer_query=str(payload.get("customer_query") or payload.get("text") or ""),
        agent_response=str(payload.get("agent_response") or ""),
        category=str(payload.get("category") or ""),
        department=str(payload.get("department") or ""),
        priority=str(payload.get("priority") or ""),
        language=str(payload.get("language") or ""),
        tags=tags,
    )


def main(argv: list[str] | None = None) -> int:
    """Build the candidate pool."""

    args = parse_args(argv)
    if args.depth < 1:
        print("--depth must be at least 1", file=sys.stderr)
        return 1

    try:
        queries: list[DevQuery] = load_dev_queries(args.queries)
    except Exception as exc:
        print(f"Could not load queries: {exc}", file=sys.stderr)
        return 1

    if args.limit:
        queries = queries[: args.limit]

    settings = get_settings()
    store = QdrantTicketStore(settings=settings, collection_name=args.collection)
    pool_rows: list[dict[str, Any]] = []
    all_doc_ids: set[str] = set()
    per_mode_counts: dict[str, int] = dict.fromkeys(MODES, 0)

    try:
        for index, query in enumerate(queries, start=1):
            ranked_by_mode: dict[str, list[str]] = {}
            for mode in MODES:
                response = store.search(
                    RetrievalSearchRequest(query=query.text, top_k=args.depth, mode=mode)  # type: ignore[arg-type]
                )
                ranked_by_mode[mode] = [hit.ticket_id for hit in response.hits]
                per_mode_counts[mode] += len(response.hits)

            candidates = pool_candidates(ranked_by_mode)
            pool_rows.append(pool_entry(query.query_id, query.text, candidates))
            all_doc_ids.update(candidate.doc_id for candidate in candidates)
            if index % 10 == 0:
                print(f"Pooled {index}/{len(queries)} queries", flush=True)

        payloads = store.fetch_payloads(sorted(all_doc_ids))
    except RetrievalError as exc:
        print(f"Candidate pooling failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    missing = sorted(all_doc_ids - set(payloads))
    if missing:
        print(
            f"Candidate pooling failed: {len(missing)} pooled ids missing from Qdrant "
            f"(first: {missing[0]})",
            file=sys.stderr,
        )
        return 1

    documents = [_document_from_payload(doc_id, payloads[doc_id]) for doc_id in sorted(all_doc_ids)]

    write_jsonl(args.output, pool_rows)
    write_pool_documents(args.documents, documents)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "collection": store.collection_name,
        "dense_model": settings.qdrant_dense_model,
        "sparse_model": settings.qdrant_sparse_model,
        "depth_per_mode": args.depth,
        "modes": list(MODES),
        "queries": len(queries),
        "retrieved_per_mode": per_mode_counts,
        "pooled_candidates": sum(len(row["candidates"]) for row in pool_rows),
        "unique_documents": len(documents),
        "query_file": str(args.queries),
        "query_file_sha256": sha256_file(args.queries),
        "pool_file": str(args.output),
        "pool_file_sha256": sha256_file(args.output),
        "documents_file": str(args.documents),
        "documents_file_sha256": sha256_file(args.documents),
    }
    write_json(args.summary, summary)

    print(f"Pooled {len(queries)} queries into {len(documents)} unique documents")
    print(f"Pool written to {args.output}")
    print(f"Documents written to {args.documents}")
    print(f"Summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
