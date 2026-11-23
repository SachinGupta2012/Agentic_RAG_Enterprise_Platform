"""Runtime LLM settings, independent of offline judging."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["groq", "gemini", "cloudflare"]


def _default_providers() -> list[Provider]:
    return ["groq"]


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GENERATION_",
        extra="ignore",
        case_sensitive=False,
    )
    provider: Provider = "groq"
    enabled_providers: list[Provider] = Field(default_factory=_default_providers)
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-3.5-flash-lite"
    cloudflare_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    temperature: float = Field(default=0.0, ge=0, le=1)
    max_output_tokens: int = Field(default=1536, ge=256, le=4096)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    context_chars: int = Field(default=8000, ge=2000, le=16000)
    requests_per_minute: int = Field(default=6, ge=1, le=30)
    tokens_per_minute: int = Field(default=6000, ge=2000, le=8000)
    groq_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GEMINI_API_KEY")
    cloudflare_api_token: SecretStr = Field(
        default=SecretStr(""), validation_alias="CLOUDFLARE_API_TOKEN"
    )
    cloudflare_account_id: SecretStr = Field(
        default=SecretStr(""), validation_alias="CLOUDFLARE_ACCOUNT_ID"
    )

    def model_for(self, provider: Provider) -> str:
        return str(getattr(self, f"{provider}_model"))

    def configured(self, provider: Provider) -> bool:
        if provider == "cloudflare":
            return bool(
                self.cloudflare_api_token.get_secret_value().strip()
                and self.cloudflare_account_id.get_secret_value().strip()
            )
        value: SecretStr = getattr(self, f"{provider}_api_key")
        return bool(value.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_generation_settings() -> GenerationSettings:
    return GenerationSettings()
