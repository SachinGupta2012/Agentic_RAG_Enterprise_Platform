"""Hybrid retrieval endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.retrieval.dependencies import get_ticket_store
from app.retrieval.schemas import (
    RetrievalHealthResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from app.retrieval.store import (
    QdrantTicketStore,
    RetrievalCollectionMissing,
    RetrievalDimensionMismatch,
    RetrievalNotConfigured,
    RetrievalUnavailable,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
    summary="Search over the historical ticket corpus",
)
def search(
    request: RetrievalSearchRequest,
    store: Annotated[QdrantTicketStore, Depends(get_ticket_store)],
) -> RetrievalSearchResponse:
    """Run dense, sparse, or hybrid retrieval based on the request mode."""

    try:
        return store.search(request)
    except RetrievalNotConfigured as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not configured.",
        ) from exc
    except RetrievalDimensionMismatch as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index configuration does not match the collection.",
        ) from exc
    except RetrievalCollectionMissing as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index is not ready. Run the ingestion script.",
        ) from exc
    except RetrievalUnavailable as exc:
        logger.exception("Retrieval search failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is unavailable.",
        ) from exc


@router.get(
    "/health",
    response_model=RetrievalHealthResponse,
    summary="Retrieval index readiness",
)
def health(
    store: Annotated[QdrantTicketStore, Depends(get_ticket_store)],
) -> RetrievalHealthResponse:
    """Return collection readiness and document count."""

    try:
        return store.health()
    except RetrievalNotConfigured as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not configured.",
        ) from exc
    except RetrievalCollectionMissing as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index is not ready. Run the ingestion script.",
        ) from exc
    except RetrievalUnavailable as exc:
        logger.exception("Retrieval health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is unavailable.",
        ) from exc
