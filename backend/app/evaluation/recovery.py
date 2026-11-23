"""Durable single-writer guard and input/provenance checks for judge recovery."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.evaluation.jsonl import sha256_file
from app.evaluation.qrels import load_qrels


def durable_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def recovery_guard(output: Path) -> Iterator[None]:
    """Fail closed on concurrent runs; stale locks require explicit inspection."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def validate_resume(
    output: Path,
    inputs: list[Path],
    provider: str,
    model: str,
    prompt: str,
    generation: dict[str, Any],
) -> None:
    """Pin inputs and generation settings; never silently combine judge variants."""
    if output.is_file():
        for record in load_qrels(output):
            if (record.judge_provider, record.judge_model, record.prompt_version) != (
                provider,
                model,
                prompt,
            ):
                raise ValueError("Existing qrels provenance differs from requested judge")
    manifest = output.with_suffix(output.suffix + ".resume.json")
    expected = {
        "inputs": {str(path.resolve()): sha256_file(path) for path in inputs},
        "provider": provider,
        "model": model,
        "prompt_version": prompt,
        "generation": generation,
    }
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Resume inputs or generation settings changed")
    else:
        durable_json(manifest, expected)
    pending = output.with_suffix(output.suffix + ".pending.json")
    if pending.exists():
        pair = json.loads(pending.read_text(encoding="utf-8"))
        saved = {(r.query_id, r.doc_id) for r in load_qrels(output)} if output.exists() else set()
        if (pair["query_id"], pair["doc_id"]) not in saved:
            raise ValueError(
                "Uncertain in-flight pair requires reconciliation; refusing automatic resend"
            )
        pending.unlink()
