"""Settings for the offline evaluation workflow.

Kept separate from :mod:`app.core.config` so the runtime retrieval service never
depends on judge configuration. The same ``.env`` file is read.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.0-flash",
}


class EvaluationSettings(BaseSettings):
    """Configuration for LLM-assisted relevance judging."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    judge_provider: Literal["groq", "gemini"] = "groq"
    judge_model: str = ""
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_max_output_tokens: int = Field(default=1024, ge=32, le=8192)
    judge_reasoning_effort: str = Field(default="low", max_length=16)
    judge_timeout_seconds: float = Field(default=60.0, gt=0)
    judge_max_retries: int = Field(default=5, ge=0, le=20)
    judge_output_retries: int = Field(default=2, ge=0, le=10)
    judge_backoff_base_seconds: float = Field(default=2.0, gt=0)
    judge_backoff_max_seconds: float = Field(default=60.0, gt=0)
    judge_requests_per_minute: int = Field(default=0, ge=0)
    judge_max_document_chars: int = Field(default=4000, ge=200, le=20000)

    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("GROQ_API_KEY"),
    )
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("GEMINI_API_KEY"),
    )

    @property
    def resolved_model(self) -> str:
        """Return the configured model or the provider default."""

        return self.judge_model.strip() or DEFAULT_MODELS[self.judge_provider]

    def api_key_for(self, provider: str) -> str:
        """Return the API key for a provider without exposing it in reprs."""

        if provider == "groq":
            return self.groq_api_key.get_secret_value().strip()
        if provider == "gemini":
            return self.gemini_api_key.get_secret_value().strip()
        raise ValueError(f"Unsupported judge provider: {provider!r}")

    @property
    def min_request_interval_seconds(self) -> float:
        """Minimum spacing between requests implied by the configured rate limit."""

        if self.judge_requests_per_minute <= 0:
            return 0.0
        return 60.0 / self.judge_requests_per_minute


@lru_cache(maxsize=1)
def get_evaluation_settings() -> EvaluationSettings:
    """Return the process-wide evaluation settings."""

    return EvaluationSettings()
