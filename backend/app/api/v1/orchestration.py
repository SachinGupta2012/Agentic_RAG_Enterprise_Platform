"""Opt-in bounded orchestration; existing endpoints remain independent."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.agents.dependencies import GraphRunner, get_orchestrator, get_trusted_filters
from app.agents.state import OrchestrationRequest, OrchestrationResult
from app.generation.dependencies import get_response_service
from app.generation.service import ResponseService
from app.retrieval.dependencies import get_ticket_store
from app.retrieval.schemas import RetrievalFilters
from app.retrieval.store import QdrantTicketStore

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


@router.post("/respond", response_model=OrchestrationResult)
async def respond(
    request: OrchestrationRequest,
    graph: Annotated[GraphRunner, Depends(get_orchestrator)],
    store: Annotated[QdrantTicketStore, Depends(get_ticket_store)],
    generation: Annotated[ResponseService, Depends(get_response_service)],
    scope: Annotated[RetrievalFilters, Depends(get_trusted_filters)],
) -> OrchestrationResult | JSONResponse:
    try:
        result, status = await graph.run(request, store, generation, trusted_filters=scope)
    except PermissionError as exc:
        raise HTTPException(403, "Requested filters are outside the allowed scope.") from exc
    except Exception as exc:
        raise HTTPException(500, "Orchestration failed unexpectedly.") from exc
    if status != 200:
        headers = {"Retry-After": str(result.retry_after)} if result.retry_after else None
        return JSONResponse(
            status_code=status, content=result.model_dump(mode="json"), headers=headers
        )
    return result
