"""Continue the pinned Groq run across quota windows, then finalize Phase 2.

Stops on uncertain requests, provenance errors or non-quota failures. Never
switches provider/model, reads reserved evaluation tickets, or writes Qdrant.
Run from the repository root. Keep the computer awake while this worker runs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.evaluation.candidates import load_pool
from app.evaluation.jsonl import read_jsonl, sha256_file
from app.evaluation.qrels import load_qrels, qrels_fingerprint
from app.evaluation.recovery import durable_json, recovery_guard

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/eval"
QRELS = ARTIFACTS / "qrels_llm_v1.jsonl"
STATE = ARTIFACTS / "recovery_status.json"
STOP = ARTIFACTS / "recovery.stop"


def status(phase: str, **details: object) -> None:
    durable_json(STATE, {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **details})


def remaining() -> int:
    pool = load_pool(ARTIFACTS / "candidate_pool.jsonl")
    pairs = {(q, d) for q, docs in pool.items() for d in docs}
    saved = {(r.query_id, r.doc_id) for r in load_qrels(QRELS)}
    return len(pairs - saved)


def pause(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if STOP.exists():
            status("stopped", reason="recovery.stop exists")
            return False
        time.sleep(min(30, max(0, end - time.monotonic())))
    return True


def command(args: list[str], log: Path) -> int:
    with log.open("a", encoding="utf-8") as handle:
        environment = dict(os.environ)
        if "pytest" in args:
            environment["RUN_QDRANT_INTEGRATION"] = "1"
        return subprocess.call(
            args, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )


def finalize() -> int:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outputs = {
        "evaluation": ARTIFACTS / f"retrieval_evaluation_{stamp}.json",
        "worksheet": ARTIFACTS / f"manual_verification_{stamp}.jsonl",
        "verification": ARTIFACTS / f"manual_verification_report_{stamp}.json",
        "latency": ARTIFACTS / f"latency_benchmark_{stamp}.json",
    }
    log = ARTIFACTS / f"finalization_{stamp}.log"
    python = [sys.executable, "-B"]
    operations = [
        [
            *python,
            "scripts/verify_qrels_sample.py",
            "--size",
            "20",
            "--worksheet",
            str(outputs["worksheet"]),
            "--report",
            str(outputs["verification"]),
        ],
        [*python, "scripts/evaluate_retrieval.py", "--output", str(outputs["evaluation"])],
        [*python, "scripts/benchmark_latency.py", "--output", str(outputs["latency"])],
        [*python, "-m", "ruff", "check", "--no-cache", "."],
        [*python, "-m", "ruff", "format", "--check", "--no-cache", "."],
        [*python, "-m", "mypy", "--no-incremental", "--cache-dir", "nul"],
        [*python, "scripts/check_notebooks.py"],
        [*python, "-m", "pytest", "-p", "no:cacheprovider", "-q"],
    ]
    status("finalizing", judgments=len(load_qrels(QRELS)), log=str(log))
    for args in operations:
        if command(args, log):
            status("needs_attention", reason="Finalization check failed", log=str(log))
            return 1
    evaluation = json.loads(outputs["evaluation"].read_text(encoding="utf-8"))
    worksheet = read_jsonl(outputs["worksheet"])
    complete = all(
        m["judgment_coverage"] == 1 and m["failed_queries"] == 0 for m in evaluation["modes"]
    )
    report = {
        "status": "complete" if complete else "needs_attention",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics_label": "LLM-judgment-based pooled estimates",
        "judgment_coverage": 1.0,
        "remaining_pairs": remaining(),
        "qrels": qrels_fingerprint(load_qrels(QRELS)),
        "qrels_sha256": sha256_file(QRELS),
        "metrics": evaluation["modes"],
        "latency": json.loads(outputs["latency"].read_text(encoding="utf-8")),
        "worksheet_pairs": len(worksheet),
        "human_labels_blank": all(row.get("human_relevance") in (None, "") for row in worksheet),
        "quality_gates": "passed; see finalization log",
        "artifacts": {key: str(path) for key, path in outputs.items()},
        "limitations": [
            "Pooled estimates are not human ground truth or absolute corpus recall.",
            "Human verification labels are blank; agreement cannot yet be measured.",
            "Historical judgments predate request telemetry and generation-settings pinning.",
        ],
    }
    report_path = ARTIFACTS / f"phase2_completion_{stamp}.json"
    durable_json(report_path, report)
    status(str(report["status"]), report=str(report_path))
    return 0 if complete else 1


def main() -> int:
    with recovery_guard(ARTIFACTS / "phase2_worker"):
        # An already authorized foreground resume may still be running.
        while QRELS.with_suffix(QRELS.suffix + ".lock").exists():
            status("waiting_for_active_judge", remaining_pairs=remaining())
            if not pause(30):
                return 1
        for _ in range(30):
            if STOP.exists():
                status("stopped", reason="recovery.stop exists")
                return 1
            if remaining() == 0:
                return finalize()
            previous = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
            until = previous.get("resume_after")
            if until:
                delay = (datetime.fromisoformat(until) - datetime.now(UTC)).total_seconds()
                if delay > 0 and not pause(delay):
                    return 1
            before = datetime.now(UTC)
            status(
                "judging",
                remaining_pairs=remaining(),
                provider="groq",
                model="openai/gpt-oss-120b",
                requests_per_minute=8,
            )
            code = command(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "scripts/generate_qrels.py",
                    "--provider",
                    "groq",
                    "--model",
                    "openai/gpt-oss-120b",
                ],
                ARTIFACTS / "recovery_judging.log",
            )
            if remaining() == 0:
                return finalize()
            failures_path = ARTIFACTS / "judgment_failures.jsonl"
            failures = read_jsonl(failures_path) if failures_path.exists() else []
            latest = failures[-1] if failures else {}
            if (
                latest.get("error_type") != "JudgeQuotaError"
                or datetime.fromisoformat(latest["timestamp"]) < before
            ):
                status(
                    "needs_attention",
                    exit_code=code,
                    remaining_pairs=remaining(),
                    reason="Non-quota interruption; no automatic resend",
                )
                return 1
            match = re.search(r"cooldown ([\d.]+)s", str(latest.get("error", "")))
            delay = float(match.group(1)) + 60 if match and float(match.group(1)) > 60 else 86400
            resume_after = datetime.now(UTC) + timedelta(seconds=delay)
            status(
                "quota_cooldown", remaining_pairs=remaining(), resume_after=resume_after.isoformat()
            )
        status("needs_attention", reason="30 quota windows reached")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
