"""Tests for the Qdrant store without network access."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.retrieval.pii import RedactionCounts
from app.retrieval.schemas import RetrievalFilters, RetrievalSearchRequest
from app.retrieval.store import (
    QdrantTicketStore,
    RetrievalCollectionMissing,
    RetrievalDimensionMismatch,
    RetrievalNotConfigured,
    TicketDocument,
    build_filter,
    build_point,
)


class FakeClient:
    """Minimal Qdrant client surface used by the store tests."""

    def __init__(self) -> None:
        self.query_kwargs: dict[str, Any] | None = None
        self.exists = True
        self.count_value = 16000
        self.vector_size = 384

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        vector = SimpleNamespace(size=self.vector_size, distance="Cosine")
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors={"dense": vector}))
        )

    def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        return SimpleNamespace(count=self.count_value)

    def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query_kwargs = kwargs
        point = SimpleNamespace(
            id="7294fd0a-a460-58e0-b3e2-07058d7fe95d",
            score=0.75,
            payload={
                "ticket_id": "7294fd0a-a460-58e0-b3e2-07058d7fe95d",
                "subject": "Account Disruption",
                "customer_query": "The account portal is offline.",
                "agent_response": "Our team is investigating the outage.",
                "category": "Incident",
                "department": "Technical Support",
                "priority": "high",
                "language": "en",
                "tags": ["Account", "Outage"],
                "source_ticket_id": "dataset:1",
                "source_dataset": "dataset",
                "source_row_index": 1,
                "source_version": "51.0",
                "pii_redactions": {"email": 0, "phone": 0, "url": 0},
            },
        )
        return SimpleNamespace(points=[point])


def _settings() -> Settings:
    return Settings(
        qdrant_url="https://example.cloud.qdrant.io",
        qdrant_api_key="test-key",
    )


def test_build_filter_maps_and_normalizes_values() -> None:
    filters = RetrievalFilters(
        category="incident",
        department="Technical Support",
        priority="HIGH",
        language="EN",
    )

    result = build_filter(filters)

    assert result is not None
    assert result.must is not None
    values = {
        condition.key: condition.match.value  # type: ignore[union-attr]
        for condition in result.must
    }
    assert values == {
        "category": "Incident",
        "department": "Technical Support",
        "priority": "high",
        "language": "en",
    }


def test_build_filter_returns_none_for_empty_filters() -> None:
    assert build_filter(RetrievalFilters()) is None


def test_build_point_uses_named_vectors_and_provenance() -> None:
    document = TicketDocument(
        ticket_id="7294fd0a-a460-58e0-b3e2-07058d7fe95d",
        source_ticket_id="dataset:1",
        subject="Account Disruption",
        customer_query="The account portal is offline.",
        agent_response="Our team is investigating.",
        category="Incident",
        department="Technical Support",
        priority="high",
        language="en",
        tags=("Account", "Outage"),
        resolution_status="resolved",
        source_dataset="dataset",
        source_row_index=1,
        source_version="51.0",
        dense_text="Subject: Account Disruption",
        sparse_text="Subject: Account Disruption Resolution: investigating",
        display_text="Subject: Account Disruption",
        pii_redactions=RedactionCounts(),
    )

    point = build_point(
        document,
        dense_model="sentence-transformers/all-minilm-l6-v2",
        sparse_model="qdrant/bm25",
        dense_vector="dense",
        sparse_vector="bm25",
    )

    assert point.id == document.ticket_id
    assert point.payload is not None
    assert point.payload["source_has_native_id"] is False
    assert point.payload["source_ticket_id_kind"] == "derived_locator"
    assert point.payload["split"] == "retrieval"
    assert point.vector["dense"].model == "sentence-transformers/all-minilm-l6-v2"
    assert point.vector["bm25"].model == "qdrant/bm25"


def test_search_builds_hybrid_prefetch_and_maps_hits() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    response = store.search(RetrievalSearchRequest(query="account portal offline", top_k=3))

    assert client.query_kwargs is not None
    assert client.query_kwargs["limit"] == 3
    assert len(client.query_kwargs["prefetch"]) == 2
    assert client.query_kwargs["query"].rrf.k == 60
    assert response.count == 1
    assert response.hits[0].subject == "Account Disruption"
    assert response.hits[0].source_row_index == 1
    assert response.hits[0].score_type == "rrf"


def test_search_redacts_query_before_embedding() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    response = store.search(
        RetrievalSearchRequest(query="Please email a.customer@example.com", top_k=3)
    )

    assert client.query_kwargs is not None
    prefetch = client.query_kwargs["prefetch"]
    assert "a.customer@example.com" not in prefetch[0].query.text
    assert response.query_pii_redactions["email"] == 1
    assert response.warnings


def test_search_passes_filters() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    store.search(
        RetrievalSearchRequest(
            query="portal offline",
            filters=RetrievalFilters(category="Incident", priority="high"),
        )
    )

    assert client.query_kwargs is not None
    query_filter = client.query_kwargs["query_filter"]
    assert query_filter is not None
    assert {condition.key for condition in query_filter.must} == {"category", "priority"}


def test_health_reports_count() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    health = store.health()

    assert health.status == "ok"
    assert health.document_count == 16000
    assert health.collection == "support_tickets_v1"


def test_health_raises_when_collection_missing() -> None:
    client = FakeClient()
    client.exists = False
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RetrievalCollectionMissing):
        store.health()


def test_unconfigured_store_does_not_create_client() -> None:
    store = QdrantTicketStore(
        settings=Settings(qdrant_url="", qdrant_api_key=""),
    )

    with pytest.raises(RetrievalNotConfigured):
        store.health()


def test_hybrid_is_the_default_mode() -> None:
    assert RetrievalSearchRequest(query="account portal offline").mode == "hybrid"


def test_dense_mode_queries_only_the_dense_vector() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    response = store.search(
        RetrievalSearchRequest(query="account portal offline", mode="dense", top_k=3)
    )

    assert client.query_kwargs is not None
    assert client.query_kwargs["using"] == "dense"
    assert "prefetch" not in client.query_kwargs
    assert client.query_kwargs["query"].model == "sentence-transformers/all-minilm-l6-v2"
    assert client.query_kwargs["limit"] == 3
    assert response.retrieval_mode == "dense"
    assert response.hits[0].score_type == "dense"


def test_sparse_mode_queries_only_the_bm25_vector() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    response = store.search(
        RetrievalSearchRequest(query="account portal offline", mode="sparse", top_k=3)
    )

    assert client.query_kwargs is not None
    assert client.query_kwargs["using"] == "bm25"
    assert "prefetch" not in client.query_kwargs
    assert client.query_kwargs["query"].model == "qdrant/bm25"
    assert response.retrieval_mode == "sparse"
    assert response.hits[0].score_type == "sparse"


def test_single_mode_applies_metadata_filters() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    store.search(
        RetrievalSearchRequest(
            query="portal offline",
            mode="sparse",
            filters=RetrievalFilters(category="Incident"),
        )
    )

    assert client.query_kwargs is not None
    query_filter = client.query_kwargs["query_filter"]
    assert query_filter is not None
    assert {condition.key for condition in query_filter.must} == {"category"}


def test_rrf_k_is_configurable() -> None:
    client = FakeClient()
    settings = _settings().model_copy(update={"retrieval_rrf_k": 17})
    store = QdrantTicketStore(settings=settings, client=client)  # type: ignore[arg-type]

    store.search(RetrievalSearchRequest(query="account portal offline"))

    assert client.query_kwargs is not None
    assert client.query_kwargs["query"].rrf.k == 17


def test_dimension_mismatch_raises() -> None:
    client = FakeClient()
    client.vector_size = 768
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RetrievalDimensionMismatch, match="does not match"):
        store.validate_collection_dimensions()


def test_health_fails_on_dimension_mismatch() -> None:
    client = FakeClient()
    client.vector_size = 768
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RetrievalDimensionMismatch):
        store.health()


def test_dimension_matches_collection() -> None:
    client = FakeClient()
    store = QdrantTicketStore(settings=_settings(), client=client)  # type: ignore[arg-type]

    assert store.validate_collection_dimensions() == 384
