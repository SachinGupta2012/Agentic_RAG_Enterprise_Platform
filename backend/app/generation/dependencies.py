"""A cached response service keeps process-local request budgets across requests."""

from functools import lru_cache

from app.generation.config import get_generation_settings
from app.generation.service import ResponseService
from app.providers.chat import HTTPChatClient


@lru_cache(maxsize=1)
def get_response_service() -> ResponseService:
    settings = get_generation_settings()
    return ResponseService(
        settings,
        {provider: HTTPChatClient(provider, settings) for provider in settings.enabled_providers},
    )
