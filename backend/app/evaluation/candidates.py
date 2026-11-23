"""Candidate pooling for LLM relevance judging.

Every query is run through dense, sparse and hybrid retrieval. The top-k results
of each mode are merged into one candidate set, deduplicated by stable document
ID, while recording which mode surfaced each document and at which rank. The same
pool is then judged once and reused by all three modes, so no mode gets a
different judgment set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.jsonl import read_jsonl, write_jsonl

MODES: tuple[str, ...] = ("dense", "sparse", "hybrid")


class PoolError(ValueError):
    """Raised when candidate pooling input is invalid."""


@dataclass(frozen=True)
class RetrievalRef:
    """Where a document was found during pooling."""

    mode: str
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "rank": self.rank}


@dataclass(frozen=True)
class PooledCandidate:
    """A deduplicated candidate document and its retrieval provenance."""

    doc_id: str
    retrieved_by: tuple[RetrievalRef, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "retrieved_by": [ref.as_dict() for ref in self.retrieved_by],
        }


@dataclass(frozen=True)
class PoolDocument:
    """The redacted document text sent to the relevance judge."""

    doc_id: str
    subject: str
    customer_query: str
    agent_response: str
    category: str
    department: str
    priority: str
    language: str
    tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "subject": self.subject,
            "customer_query": self.customer_query,
            "agent_response": self.agent_response,
            "category": self.category,
            "department": self.department,
            "priority": self.priority,
            "language": self.language,
            "tags": list(self.tags),
        }


def pool_candidates(ranked_by_mode: Mapping[str, Sequence[str]]) -> list[PooledCandidate]:
    """Merge per-mode rankings into one deduplicated candidate list.

    Order is deterministic: modes are visited in :data:`MODES` order when present,
    then any extra mode in mapping order, and documents keep first-seen order.
    """

    if not ranked_by_mode:
        raise PoolError("at least one retrieval mode is required")

    ordered_modes = [mode for mode in MODES if mode in ranked_by_mode]
    ordered_modes.extend(mode for mode in ranked_by_mode if mode not in ordered_modes)

    refs: dict[str, list[RetrievalRef]] = {}
    order: list[str] = []
    for mode in ordered_modes:
        for rank, doc_id in enumerate(ranked_by_mode[mode], start=1):
            if not doc_id:
                continue
            if doc_id not in refs:
                refs[doc_id] = []
                order.append(doc_id)
            refs[doc_id].append(RetrievalRef(mode=mode, rank=rank))

    return [PooledCandidate(doc_id=doc_id, retrieved_by=tuple(refs[doc_id])) for doc_id in order]


def pool_entry(
    query_id: str,
    query_text: str,
    candidates: Iterable[PooledCandidate],
) -> dict[str, Any]:
    """Build the JSONL row for one query's candidate pool."""

    return {
        "query_id": query_id,
        "query": query_text,
        "candidates": [candidate.as_dict() for candidate in candidates],
    }


def load_pool(path: Path) -> dict[str, list[str]]:
    """Load a candidate pool file as ``{query_id: [doc_id, ...]}``."""

    if not path.is_file():
        raise PoolError(f"Candidate pool not found: {path}")

    pool: dict[str, list[str]] = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise PoolError(f"{path.name}:{index}: missing query_id")
        if query_id in pool:
            raise PoolError(f"{path.name}:{index}: duplicate query_id {query_id}")
        doc_ids: list[str] = []
        for candidate in row.get("candidates") or []:
            doc_id = str(candidate.get("doc_id") or "")
            if not doc_id:
                raise PoolError(f"{path.name}:{index}: candidate without doc_id")
            if doc_id in doc_ids:
                raise PoolError(f"{path.name}:{index}: duplicate doc_id {doc_id}")
            doc_ids.append(doc_id)
        pool[query_id] = doc_ids
    return pool


def load_pool_documents(path: Path) -> dict[str, PoolDocument]:
    """Load pooled document text keyed by document ID."""

    if not path.is_file():
        raise PoolError(f"Pool document file not found: {path}")

    documents: dict[str, PoolDocument] = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            raise PoolError(f"{path.name}:{index}: missing doc_id")
        if doc_id in documents:
            raise PoolError(f"{path.name}:{index}: duplicate doc_id {doc_id}")
        documents[doc_id] = PoolDocument(
            doc_id=doc_id,
            subject=str(row.get("subject") or ""),
            customer_query=str(row.get("customer_query") or ""),
            agent_response=str(row.get("agent_response") or ""),
            category=str(row.get("category") or ""),
            department=str(row.get("department") or ""),
            priority=str(row.get("priority") or ""),
            language=str(row.get("language") or ""),
            tags=tuple(str(tag) for tag in (row.get("tags") or [])),
        )
    return documents


def write_pool_documents(path: Path, documents: Iterable[PoolDocument]) -> int:
    """Write pooled document text as JSONL."""

    return write_jsonl(path, (document.as_dict() for document in documents))
