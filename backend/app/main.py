"""FastAPI application factory."""

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings


def health() -> dict[str, str]:
    """Liveness probe for the API."""

    return {"status": "ok"}


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Multi-agent RAG platform for enterprise customer support.",
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        tags=["health"],
    )
    return application


app = create_app()
