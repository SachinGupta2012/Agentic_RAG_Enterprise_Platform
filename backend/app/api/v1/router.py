"""Version 1 API router."""

from fastapi import APIRouter

from app.api.v1 import responses, retrieval

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe for the API."""

    return {"status": "ok"}


api_router.include_router(retrieval.router)
api_router.include_router(responses.router)
