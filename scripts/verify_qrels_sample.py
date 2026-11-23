"""Build the manual verification worksheet and report human/LLM agreement.

The worksheet is generated with blank ``human_relevance`` and ``human_comment``
fields. LLM grades and reasons are preserved exactly. Re-running the script keeps
any human grades already recorded for the same ``(query_id, doc_id)`` pair.

Human grades must be entered by a reviewer. This script never invents them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.candidates import load_pool_documents
from app.evaluation.jsonl import read_jsonl, write_json, write_jsonl
from app.evaluation.qrels import load_qrels
from app.evaluation.queries import load_dev_queries
from app.evaluation.verification import (
    agreement_report,
    extract_verification_pairs,
    stratified_sample,
)


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
        help="LLM qrels file.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/eval/pool_documents.jsonl"),
        help="Pooled document text file.",
    )
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=Path("data/eval/manual_verification.jsonl"),
        help="Human verification worksheet.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/eval/manual_verification_report.json"),
        help="Agreement report output.",
    )
    parser.add_argument("--size", type=int, default=20, help="Pairs in the sample.")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed.")
    parser.add_argument(
        "--max-chars", type=int, default=1200, help="Maximum characters per document field."
    )
    return parser.parse_args(argv)


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _load_existing(worksheet: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not worksheet.is_file():
        return {}
    preserved: dict[tuple[str, str], dict[str, object]] = {}
    for row in read_jsonl(worksheet):
        key = (str(row.get("query_id") or ""), str(row.get("doc_id") or ""))
        preserved[key] = {
            "human_relevance": row.get("human_relevance"),
            "human_comment": row.get("human_comment", ""),
        }
    return preserved


def main(argv: list[str] | None = None) -> int:
    """Generate the worksheet and agreement report."""

    args = parse_args(argv)

    try:
        queries = {query.query_id: query for query in load_dev_queries(args.queries)}
        records = load_qrels(args.qrels)
        documents = load_pool_documents(args.documents)
    except Exception as exc:
        print(f"Could not load evaluation inputs: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("Qrels are empty; nothing to verify.", file=sys.stderr)
        return 1

    sample = stratified_sample(records, size=args.size, seed=args.seed)
    preserved = _load_existing(args.worksheet)

    rows: list[dict[str, object]] = []
    for record in sample:
        document = documents.get(record.doc_id)
        if document is None:
            print(
                f"Warning: no document text for {record.doc_id}; skipping.",
                file=sys.stderr,
            )
            continue
        query = queries.get(record.query_id)
        previous = preserved.get((record.query_id, record.doc_id), {})
        rows.append(
            {
                "query_id": record.query_id,
                "query": query.text if query else "",
                "doc_id": record.doc_id,
                "subject": document.subject,
                "customer_query": _clip(document.customer_query, args.max_chars),
                "agent_response": _clip(document.agent_response, args.max_chars),
                "llm_relevance": record.relevance,
                "llm_reason": record.reason,
                "llm_provider": record.judge_provider,
                "llm_model": record.judge_model,
                "prompt_version": record.prompt_version,
                "human_relevance": previous.get("human_relevance"),
                "human_comment": previous.get("human_comment", ""),
            }
        )

    write_jsonl(args.worksheet, rows)

    pairs = extract_verification_pairs(rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "verified" if pairs else "pending_human_review",
        "worksheet": str(args.worksheet),
        "sample_size": len(rows),
        "grade_mixture": {
            str(grade): sum(1 for row in rows if row["llm_relevance"] == grade)
            for grade in (0, 1, 2)
        },
        "agreement": agreement_report(pairs),
        "note": (
            "Human grades are assigned by a reviewer. Agreement statistics are only "
            "computed for rows that already contain a human grade."
        ),
    }
    write_json(args.report, report)

    print(f"Worksheet written to {args.worksheet} ({len(rows)} pairs)")
    print(f"Report written to {args.report}")
    if pairs:
        print(f"Agreement: {report['agreement']}")
    else:
        print("Status: pending human review (no human grades recorded yet).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
