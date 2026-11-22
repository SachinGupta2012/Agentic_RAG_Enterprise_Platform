"""FastAPI dependencies for retrieval."""

from functools import lru_cache

from app.retrieval.store import QdrantTicketStore


@lru_cache(maxsize=1)
def get_ticket_store() -> QdrantTicketStore:
    """Return the process-wide Qdrant store."""

    return QdrantTicketStore()
