"""Lazy opt-in loading keeps the standalone API usable without the optional extra."""

from functools import lru_cache
from typing import Protocol

from fastapi import HTTPException

from app.agents.config import get_orchestration_settings
from app.agents.state import OrchestrationRequest, OrchestrationResult
from app.generation.service import ResponseService, TicketStore
from app.retrieval.schemas import RetrievalFilters


class GraphRunner(Protocol):
    async def run(
        self,
        request: OrchestrationRequest,
        store: TicketStore,
        generation: ResponseService,
        *,
        trusted_filters: RetrievalFilters | None = None,
    ) -> tuple[OrchestrationResult, int]: ...


def get_trusted_filters() -> RetrievalFilters:
    """Single-corpus development scope; replace at the authenticated API boundary."""
    return RetrievalFilters()


@lru_cache(maxsize=1)
def get_orchestrator() -> GraphRunner:
    settings = get_orchestration_settings()
    if not settings.enabled:
        raise HTTPException(404, "Orchestration is not enabled.")
    try:
        from app.agents.graph import Orchestrator
    except ImportError as exc:
        raise HTTPException(503, "Install the orchestration dependency extra.") from exc
    return Orchestrator(settings)
