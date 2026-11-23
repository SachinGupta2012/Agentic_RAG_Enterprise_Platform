"""Tests for retrieval-mode execution and latency capture."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.evaluation.queries import DevQuery
from app.evaluation.runner import (
    latency_samples,
    ranked_by_query,
    run_queries,
    run_query,
)
from app.retrieval.store import RetrievalUnavailable


class FakeStore:
    """Store stub that returns scripted hits or raises."""

    def __init__(self, doc_ids: list[str] | None = None, error: Exception | None = None) -> None:
        self._doc_ids = doc_ids or []
        self._error = error
        self.requests: list[Any] = []

    def search(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        hits = [SimpleNamespace(ticket_id=doc_id) for doc_id in self._doc_ids]
        return SimpleNamespace(hits=hits, latency_ms=42.0)


def _query(query_id: str = "dev-q0001") -> DevQuery:
    return DevQuery(
        query_id=query_id,
        text="portal offline",
        source_ticket_id="ticket-1",
        category="Incident",
        department="Support",
        priority="high",
    )


def test_run_query_collects_ranked_ids() -> None:
    store = FakeStore(["a", "b"])

    run = run_query(store, "dev-q0001", "portal offline", "dense", 5)  # type: ignore[arg-type]

    assert run.ranked_doc_ids == ("a", "b")
    assert run.empty is False
    assert run.error is None
    assert run.latency_ms == 42.0
    assert store.requests[0].mode == "dense"
    assert store.requests[0].top_k == 5


def test_run_query_handles_empty_results() -> None:
    store = FakeStore([])

    run = run_query(store, "dev-q0001", "portal offline", "hybrid", 5)  # type: ignore[arg-type]

    assert run.ranked_doc_ids == ()
    assert run.empty is True


def test_run_query_records_retrieval_error() -> None:
    store = FakeStore(error=RetrievalUnavailable("down"))

    run = run_query(store, "dev-q0001", "portal offline", "sparse", 5)  # type: ignore[arg-type]

    assert run.error is not None
    assert "RetrievalUnavailable" in run.error
    assert run.empty is True


def test_run_queries_records_failures_and_groups_by_mode() -> None:
    store = FakeStore(["a"])

    log = run_queries(store, [_query("dev-q0001"), _query("dev-q0002")], ["dense", "sparse"], 3)  # type: ignore[arg-type]

    assert len(log.runs) == 4
    assert log.failures == []
    assert ranked_by_query(log.runs, "dense") == {
        "dev-q0001": ["a"],
        "dev-q0002": ["a"],
    }
    assert len(latency_samples(log.runs, "sparse")) == 2
    assert all(sample >= 0.0 for sample in latency_samples(log.runs, "sparse"))


def test_run_queries_captures_failure_details() -> None:
    store = FakeStore(error=RetrievalUnavailable("down"))

    log = run_queries(store, [_query()], ["hybrid"], 3)  # type: ignore[arg-type]

    assert len(log.failures) == 1
    assert log.failures[0]["mode"] == "hybrid"
    assert "down" in log.failures[0]["error"]
