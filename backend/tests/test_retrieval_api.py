"""API tests for retrieval endpoints using a fake store."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.dependencies import get_ticket_store
from app.retrieval.schemas import (
    RetrievalHealthResponse,
    RetrievalHit,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from app.retrieval.store import RetrievalUnavailable


class FakeStore:
    """Fake store that records the last request."""

    def __init__(self) -> None:
        self.last_request: RetrievalSearchRequest | None = None
        self.error: Exception | None = None

    def search(self, request: RetrievalSearchRequest) -> RetrievalSearchResponse:
        self.last_request = request
        if self.error is not None:
            raise self.error
        return RetrievalSearchResponse(
            query=request.query,
            hits=[
                RetrievalHit(
                    ticket_id="7294fd0a-a460-58e0-b3e2-07058d7fe95d",
                    score=0.5,
                    subject="Account Disruption",
                    snippet="The account portal is offline.",
                    category="Incident",
                    department="Technical Support",
                    priority="high",
                    language="en",
                )
            ],
            count=1,
            dense_model="sentence-transformers/all-minilm-l6-v2",
            sparse_model="qdrant/bm25",
            collection="support_tickets_v1",
            latency_ms=12.5,
        )

    def health(self) -> RetrievalHealthResponse:
        if self.error is not None:
            raise self.error
        return RetrievalHealthResponse(
            status="ok",
            collection="support_tickets_v1",
            document_count=16000,
            dense_model="sentence-transformers/all-minilm-l6-v2",
            sparse_model="qdrant/bm25",
        )


@pytest.fixture
def fake_store() -> Iterator[FakeStore]:
    store = FakeStore()
    app.dependency_overrides[get_ticket_store] = lambda: store
    try:
        yield store
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_search_returns_hits(client: TestClient, fake_store: FakeStore) -> None:
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "account portal offline", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["hits"][0]["subject"] == "Account Disruption"
    assert fake_store.last_request is not None
    assert fake_store.last_request.top_k == 3


def test_search_normalizes_filters(client: TestClient, fake_store: FakeStore) -> None:
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "portal offline",
            "filters": {"category": "incident", "priority": "HIGH"},
        },
    )

    assert response.status_code == 200
    assert fake_store.last_request is not None
    assert fake_store.last_request.filters.category == "Incident"
    assert fake_store.last_request.filters.priority == "high"


def test_search_rejects_short_query(client: TestClient) -> None:
    response = client.post("/api/v1/retrieval/search", json={"query": "ab"})

    assert response.status_code == 422


def test_search_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "valid query", "top_k": 0},
    )

    assert response.status_code == 422


def test_search_returns_503_when_unavailable(
    client: TestClient,
    fake_store: FakeStore,
) -> None:
    fake_store.error = RetrievalUnavailable("down")

    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "portal offline"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval service is unavailable."


def test_health_returns_count(client: TestClient, fake_store: FakeStore) -> None:
    response = client.get("/api/v1/retrieval/health")

    assert response.status_code == 200
    assert response.json()["document_count"] == 16000


def test_health_returns_503_when_unavailable(
    client: TestClient,
    fake_store: FakeStore,
) -> None:
    fake_store.error = RetrievalUnavailable("down")

    response = client.get("/api/v1/retrieval/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval service is unavailable."
