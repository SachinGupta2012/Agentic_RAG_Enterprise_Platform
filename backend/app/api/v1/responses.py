"""Grounded response generation API; provider selection never silently changes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.generation.config import get_generation_settings
from app.generation.dependencies import get_response_service
from app.generation.schemas import ResponseRequest, ResponseResult
from app.generation.service import ResponseService
from app.providers.chat import ProviderError, ProviderNotConfigured, ProviderRateLimited
from app.retrieval.dependencies import get_ticket_store
from app.retrieval.store import QdrantTicketStore, RetrievalError

router = APIRouter(prefix="/responses", tags=["responses"])


@router.post("/generate", response_model=ResponseResult)
def generate(
    request: ResponseRequest,
    service: Annotated[ResponseService, Depends(get_response_service)],
    store: Annotated[QdrantTicketStore, Depends(get_ticket_store)],
) -> ResponseResult:
    try:
        return service.respond(request, store)
    except ProviderRateLimited as exc:
        raise HTTPException(
            429,
            "Generation quota is temporarily unavailable.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except ProviderNotConfigured as exc:
        raise HTTPException(
            503, "Selected generation provider is unavailable or not configured."
        ) from exc
    except ProviderError as exc:
        raise HTTPException(502, "Response generation failed. Please retry later.") from exc
    except RetrievalError as exc:
        raise HTTPException(503, "Retrieval service is unavailable.") from exc


@router.get("/providers")
def providers() -> dict[str, object]:
    settings = get_generation_settings()
    return {
        "default": settings.provider,
        "automatic_fallback": False,
        "providers": [
            {
                "provider": provider,
                "model": settings.model_for(provider),
                "enabled": provider in settings.enabled_providers,
                "configured": settings.configured(provider),
            }
            for provider in ("groq", "gemini", "cloudflare")
        ],
    }
