"""Opt-in orchestration and server-owned per-run bounds."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestrationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ORCHESTRATION_", extra="ignore")
    enabled: bool = False
    investigation_enabled: bool = True
    knowledge_enabled: bool = True
    timeout_seconds: float = Field(default=120, gt=0, le=120)
    max_retrievals: int = Field(default=2, ge=1, le=2)
    max_drafts: int = Field(default=2, ge=1, le=2)
    max_llm_attempts: int = Field(default=4, ge=1, le=4)
    max_tokens: int = Field(default=12000, ge=256, le=16000)
    max_steps: int = Field(default=16, ge=1, le=16)


@lru_cache(maxsize=1)
def get_orchestration_settings() -> OrchestrationSettings:
    return OrchestrationSettings()
