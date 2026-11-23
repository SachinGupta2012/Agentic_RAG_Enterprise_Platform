"""Development-split query selection for offline evaluation.

Selection method
----------------
Queries are drawn only from ``tickets_dev.parquet``. The ``customer_query`` text
is normalized (whitespace collapsed), rows shorter than three characters are
dropped, and duplicate normalized query texts are collapsed so the same question
is not judged twice. A deterministic sample of ``limit`` rows is taken with a
fixed random seed and then sorted by ``ticket_id`` before stable identifiers
(``dev-q0001``...) are assigned.

The dev ticket itself is never treated as a gold answer. Phase 1 assigned whole
near-duplicate clusters to a single split, so a dev ticket has no near-duplicate
inside the 16,000-record retrieval corpus. Relevance therefore has to come from
judged retrieval candidates rather than from the dev row.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.jsonl import read_jsonl, write_jsonl

REQUIRED_COLUMNS = ("ticket_id", "customer_query", "category", "department", "priority")


class QuerySelectionError(ValueError):
    """Raised when the dev split cannot produce a valid query set."""


@dataclass(frozen=True)
class DevQuery:
    """A stable evaluation query derived from the dev split."""

    query_id: str
    text: str
    source_ticket_id: str
    category: str
    department: str
    priority: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return {
            "query_id": self.query_id,
            "text": self.text,
            "source_ticket_id": self.source_ticket_id,
            "category": self.category,
            "department": self.department,
            "priority": self.priority,
        }


def _normalize(text: Any) -> str:
    return " ".join(str(text).split())


def select_dev_queries(frame: Any, limit: int = 150, seed: int = 42) -> list[DevQuery]:
    """Select a deterministic, de-duplicated query set from the dev split."""

    if limit < 1:
        raise QuerySelectionError("limit must be at least 1")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise QuerySelectionError(f"Dev split is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise QuerySelectionError("Dev split is empty")

    prepared = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    prepared["text"] = prepared["customer_query"].map(_normalize)
    prepared = prepared[prepared["text"].str.len() >= 3]
    prepared = prepared.drop_duplicates(subset=["text"], keep="first")
    prepared = prepared.dropna(subset=["text", "ticket_id"])

    if prepared.empty:
        raise QuerySelectionError("No usable queries remain after normalization")

    if len(prepared) > limit:
        prepared = prepared.sample(n=limit, random_state=seed)

    prepared = prepared.sort_values("ticket_id", kind="mergesort").reset_index(drop=True)

    queries: list[DevQuery] = []
    for position, row in enumerate(prepared.itertuples(index=False), start=1):
        queries.append(
            DevQuery(
                query_id=f"dev-q{position:04d}",
                text=str(row.text),
                source_ticket_id=str(row.ticket_id),
                category=str(row.category),
                department=str(row.department),
                priority=str(row.priority),
            )
        )
    return queries


def write_dev_queries(path: Path, queries: Iterable[DevQuery]) -> int:
    """Write dev queries as JSONL."""

    return write_jsonl(path, (query.as_dict() for query in queries))


def load_dev_queries(path: Path) -> list[DevQuery]:
    """Load dev queries from JSONL."""

    if not path.is_file():
        raise QuerySelectionError(f"Query file not found: {path}")

    queries: list[DevQuery] = []
    seen: set[str] = set()
    for index, row in enumerate(read_jsonl(path), start=1):
        try:
            query = DevQuery(
                query_id=str(row["query_id"]),
                text=str(row["text"]),
                source_ticket_id=str(row.get("source_ticket_id", "")),
                category=str(row.get("category", "")),
                department=str(row.get("department", "")),
                priority=str(row.get("priority", "")),
            )
        except KeyError as exc:
            raise QuerySelectionError(f"{path.name}:{index}: missing field {exc}") from exc
        if query.query_id in seen:
            raise QuerySelectionError(f"{path.name}:{index}: duplicate query_id")
        seen.add(query.query_id)
        queries.append(query)
    return queries


def query_ids(queries: Sequence[DevQuery]) -> list[str]:
    """Return the query identifiers in file order."""

    return [query.query_id for query in queries]
