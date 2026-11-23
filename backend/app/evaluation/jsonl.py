"""Small JSON/JSONL helpers shared by the offline evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines and rejecting non-object rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            rows.append(parsed)
    return rows


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
) -> int:
    """Write rows as JSONL and return the number of rows written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    written = 0
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written


def read_json(path: Path) -> Any:
    """Read a JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON document with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
