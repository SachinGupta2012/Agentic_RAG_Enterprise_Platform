"""Read-only contracts against the live Qdrant collection."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from app.core.config import get_settings
from app.retrieval.schemas import RetrievalFilters, RetrievalSearchRequest
from app.retrieval.store import QdrantTicketStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_QDRANT_INTEGRATION") != "1",
        reason="set RUN_QDRANT_INTEGRATION=1 to run live Qdrant contracts",
    ),
]


@pytest.fixture(scope="module")
def store() -> Iterator[QdrantTicketStore]:
    settings = get_settings()
    if not settings.qdrant_url or not settings.qdrant_api_key_value:
        pytest.skip("Qdrant credentials are not configured")
    instance = QdrantTicketStore(settings=settings)
    try:
        yield instance
    finally:
        instance.close()


def test_collection_is_ready(store: QdrantTicketStore) -> None:
    health = store.health()

    assert health.status == "ok"
    assert health.document_count is not None
    assert health.document_count >= 16000


def test_configured_dimensions_match_collection(store: QdrantTicketStore) -> None:
    assert store.validate_collection_dimensions() == store.settings.qdrant_dense_vector_size


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_each_mode_returns_hits(store: QdrantTicketStore, mode: str) -> None:
    response = store.search(
        RetrievalSearchRequest(
            query="account portal offline and cannot log in",
            top_k=3,
            mode=mode,  # type: ignore[arg-type]
        )
    )

    assert response.retrieval_mode in {mode, "hybrid_rrf"}
    assert response.count >= 1
    assert all(hit.ticket_id for hit in response.hits)


def test_metadata_filter_is_applied(store: QdrantTicketStore) -> None:
    response = store.search(
        RetrievalSearchRequest(
            query="account portal offline",
            top_k=5,
            mode="hybrid",
            filters=RetrievalFilters(priority="high"),
        )
    )

    assert response.count >= 1
    assert all(hit.priority == "high" for hit in response.hits)
