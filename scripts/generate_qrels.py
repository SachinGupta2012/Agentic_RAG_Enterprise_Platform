"""Generate LLM relevance judgments for the pooled candidates.

Offline evaluation only. Resumable: existing ``(query_id, doc_id)`` pairs are
skipped, and every new judgment is appended immediately. Provider failures are
recorded as failures and are never converted into a relevance grade.

No ticket content or API key is logged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evaluation.candidates import load_pool, load_pool_documents
from app.evaluation.config import get_evaluation_settings
from app.evaluation.jsonl import write_json, write_jsonl
from app.evaluation.judge import (
    PROMPT_VERSION,
    JudgeConfigurationError,
    JudgeError,
    JudgePermanentError,
    JudgeUncertainError,
    LLMJudge,
    build_chat_client,
)
from app.evaluation.qrels import (
    QrelsValidationError,
    load_qrels,
    utc_now_iso,
    write_qrels,
)
from app.evaluation.queries import DevQuery, load_dev_queries
from app.evaluation.recovery import durable_json, recovery_guard, validate_resume


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
        "--pool",
        type=Path,
        default=Path("data/eval/candidate_pool.jsonl"),
        help="Candidate pool file.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/eval/pool_documents.jsonl"),
        help="Pooled document text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/qrels_llm_v1.jsonl"),
        help="Qrels output file (appended, resumable).",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("data/eval/judgment_failures.jsonl"),
        help="Failed judgment log.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/eval/judgment_summary.json"),
        help="Run summary output file.",
    )
    parser.add_argument("--provider", choices=("groq", "gemini"), default=None)
    parser.add_argument("--model", default=None, help="Override the judge model.")
    parser.add_argument(
        "--queries-limit",
        type=int,
        default=0,
        help="Process only the first N queries (0 = all). Use 10 for the pilot.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Stop after N new judgments in this run (0 = no cap).",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=10,
        help="Abort after this many consecutive judgment failures.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Restrict to a query id (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit.")
    parser.add_argument("--requests-per-minute", type=int, default=8)
    return parser.parse_args(argv)


def _select_queries(queries: list[DevQuery], args: argparse.Namespace) -> list[DevQuery]:
    selected = queries
    if args.query_id:
        wanted = set(args.query_id)
        selected = [query for query in selected if query.query_id in wanted]
    if args.queries_limit:
        selected = selected[: args.queries_limit]
    return selected


def _run(argv: list[str] | None = None) -> int:
    """Generate judgments for the pooled candidate set."""

    args = parse_args(argv)
    settings = get_evaluation_settings()
    updates: dict[str, object] = {}
    if args.provider:
        updates["judge_provider"] = args.provider
    if args.model:
        updates["judge_model"] = args.model
    if updates:
        settings = settings.model_copy(update=updates)
    if args.requests_per_minute <= 0:
        raise ValueError("Recovery requires positive request pacing")
    settings = settings.model_copy(update={"judge_requests_per_minute": args.requests_per_minute})

    print(f"Provider: {settings.judge_provider}")
    print(f"Model: {settings.resolved_model}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Requests/minute limit: {settings.judge_requests_per_minute or 'unlimited'}")

    if args.dry_run:
        print("Dry run: no requests were made.")
        return 0

    api_key = settings.api_key_for(settings.judge_provider)
    if not api_key:
        print(
            f"Judging blocked: no API key configured for {settings.judge_provider}. "
            "Set the key in .env. No judgments were written.",
            file=sys.stderr,
        )
        return 2

    try:
        queries = load_dev_queries(args.queries)
        pool = load_pool(args.pool)
        documents = load_pool_documents(args.documents)
    except Exception as exc:
        print(f"Could not load evaluation inputs: {exc}", file=sys.stderr)
        return 1

    selected = _select_queries(queries, args)
    if not selected:
        print("No queries selected.", file=sys.stderr)
        return 1

    existing: set[tuple[str, str]] = set()
    if args.output.is_file():
        try:
            for record in load_qrels(args.output):
                existing.add((record.query_id, record.doc_id))
        except QrelsValidationError as exc:
            print(f"Existing qrels are invalid: {exc}", file=sys.stderr)
            return 1

    validate_resume(
        args.output,
        [args.queries, args.pool, args.documents],
        settings.judge_provider,
        settings.resolved_model,
        PROMPT_VERSION,
        {
            key: getattr(settings, key)
            for key in (
                "judge_temperature",
                "judge_max_output_tokens",
                "judge_reasoning_effort",
                "judge_max_document_chars",
            )
        },
    )
    client = build_chat_client(settings)
    if settings.judge_provider == "groq":
        from app.evaluation.judge import GroqChatClient

        assert isinstance(client, GroqChatClient)
        client.telemetry = lambda event: write_jsonl(
            args.output.with_suffix(".requests.jsonl"), [event], append=True
        )
    judge = LLMJudge(client, settings)
    pending = args.output.with_suffix(args.output.suffix + ".pending.json")

    started_at = utc_now_iso()
    created = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    def record_failure(failure: dict[str, str]) -> None:
        failures.append(failure)
        write_jsonl(args.failures, [failure], append=True)

    grades = {0: 0, 1: 0, 2: 0}
    aborted = False
    consecutive_failures = 0

    try:
        for query in selected:
            doc_ids = pool.get(query.query_id, [])
            if not doc_ids:
                print(f"Warning: no pooled candidates for {query.query_id}", file=sys.stderr)
            for doc_id in doc_ids:
                if (query.query_id, doc_id) in existing:
                    skipped += 1
                    continue
                if args.max_pairs and created >= args.max_pairs:
                    aborted = True
                    break

                document = documents.get(doc_id)
                if document is None:
                    record_failure(
                        {
                            "query_id": query.query_id,
                            "doc_id": doc_id,
                            "error_type": "MissingDocument",
                            "error": "pooled document text not found",
                            "timestamp": utc_now_iso(),
                        }
                    )
                    continue

                durable_json(
                    pending,
                    {"query_id": query.query_id, "doc_id": doc_id, "started_at": utc_now_iso()},
                )
                try:
                    result = judge.judge(query.text, document)
                except JudgeConfigurationError as exc:
                    record_failure(
                        {
                            "query_id": query.query_id,
                            "doc_id": doc_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "timestamp": utc_now_iso(),
                        }
                    )
                    print(f"Judge configuration failure: {exc}", file=sys.stderr)
                    if not isinstance(exc, JudgeUncertainError):
                        pending.unlink(missing_ok=True)
                    aborted = True
                    break
                except JudgeError as exc:
                    record_failure(
                        {
                            "query_id": query.query_id,
                            "doc_id": doc_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "timestamp": utc_now_iso(),
                        }
                    )
                    consecutive_failures += 1
                    pending.unlink(missing_ok=True)
                    print(
                        f"Judge failure ({type(exc).__name__}); {consecutive_failures} consecutive",
                        file=sys.stderr,
                    )
                    if (
                        isinstance(exc, JudgePermanentError)
                        or consecutive_failures >= args.max_consecutive_failures
                    ):
                        print(
                            "Too many consecutive failures; stopping. Re-run to resume.",
                            file=sys.stderr,
                        )
                        aborted = True
                        break
                    continue

                consecutive_failures = 0
                record = result.to_record(query.query_id, doc_id)
                write_qrels(args.output, [record], append=True)
                pending.unlink(missing_ok=True)
                existing.add((query.query_id, doc_id))
                grades[result.relevance] += 1
                created += 1
                if created % 25 == 0:
                    print(f"Judged {created} new pairs", flush=True)

            if aborted:
                break
    finally:
        close = getattr(getattr(client, "_client", None), "close", None)
        if close:
            close()

    summary = {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "provider": settings.judge_provider,
        "model": settings.resolved_model,
        "prompt_version": PROMPT_VERSION,
        "queries_selected": len(selected),
        "judgments_created": created,
        "judgments_skipped_existing": skipped,
        "judgments_failed": len(failures),
        "grade_distribution": {str(grade): count for grade, count in grades.items()},
        "aborted": aborted,
        "output": str(args.output),
        "failures": str(args.failures),
    }
    summary_path = args.summary.with_name(
        args.summary.stem + "_" + started_at.replace(":", "").replace("+", "_") + ".json"
    )
    write_json(summary_path, summary)

    print(f"Created {created} judgments, skipped {skipped} existing, {len(failures)} failures.")
    print(f"Grades: {summary['grade_distribution']}")
    print(f"Qrels written to {args.output}")
    print(f"Summary written to {summary_path}")
    if aborted:
        print("Run stopped early; re-run to resume.", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return _run(argv)
    try:
        with recovery_guard(args.output):
            return _run(argv)
    except (ValueError, FileExistsError) as exc:
        print(f"Recovery blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
