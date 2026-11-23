"""Execution of retrieval modes for offline evaluation and benchmarking."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.evaluation.queries import DevQuery
from app.retrieval.schemas import RetrievalSearchRequest
from app.retrieval.store import QdrantTicketStore, RetrievalError


@dataclass(frozen=True)
class ModeRun:
    """One query executed in one retrieval mode."""

    query_id: str
    mode: str
    ranked_doc_ids: tuple[str, ...]
    latency_ms: float
    wall_ms: float
    empty: bool
    error: str | None = None


@dataclass
class RunLog:
    """Collected runs plus failures for a batch."""

    runs: list[ModeRun] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)


def run_query(
    store: QdrantTicketStore,
    query_id: str,
    query_text: str,
    mode: str,
    top_k: int,
) -> ModeRun:
    """Run one query in one mode and measure client-observed latency."""

    request = RetrievalSearchRequest(query=query_text, top_k=top_k, mode=mode)
    started = time.perf_counter()
    try:
        response = store.search(request)
    except RetrievalError as exc:
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        return ModeRun(
            query_id=query_id,
            mode=mode,
            ranked_doc_ids=(),
            latency_ms=wall_ms,
            wall_ms=wall_ms,
            empty=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    wall_ms = round((time.perf_counter() - started) * 1000, 2)
    ranked = tuple(hit.ticket_id for hit in response.hits)
    return ModeRun(
        query_id=query_id,
        mode=mode,
        ranked_doc_ids=ranked,
        latency_ms=response.latency_ms,
        wall_ms=wall_ms,
        empty=not ranked,
    )


def run_queries(
    store: QdrantTicketStore,
    queries: Sequence[DevQuery],
    modes: Iterable[str],
    top_k: int,
) -> RunLog:
    """Run every query in every mode, recording failures instead of aborting."""

    log = RunLog()
    for query in queries:
        for mode in modes:
            run = run_query(store, query.query_id, query.text, mode, top_k)
            log.runs.append(run)
            if run.error:
                log.failures.append({"query_id": query.query_id, "mode": mode, "error": run.error})
    return log


def ranked_by_query(runs: Iterable[ModeRun], mode: str) -> dict[str, list[str]]:
    """Return ``{query_id: [doc_id, ...]}`` for one mode."""

    return {run.query_id: list(run.ranked_doc_ids) for run in runs if run.mode == mode}


def latency_samples(runs: Iterable[ModeRun], mode: str) -> list[float]:
    """Return client-observed latency samples for one mode."""

    return [run.wall_ms for run in runs if run.mode == mode]
