"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Enterprise Support Intelligence Platform"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/enterprise_support"

    # Qdrant Cloud. The alias keeps the old misspelled variable working.
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("QDRANT_API_KEY", "QUADRANT_API_KEY"),
    )
    qdrant_collection: str = "support_tickets_v1"
    qdrant_dense_model: str = "sentence-transformers/all-minilm-l6-v2"
    qdrant_sparse_model: str = "qdrant/bm25"
    qdrant_dense_vector: str = "dense"
    qdrant_sparse_vector: str = "bm25"
    qdrant_dense_vector_size: int = Field(default=384, ge=1)

    # Retrieval behaviour.
    retrieval_default_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    retrieval_prefetch_limit: int = 50
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_timeout_seconds: int = 60
    retrieval_batch_size: int = 64

    @property
    def qdrant_api_key_value(self) -> str:
        """Return the API key without exposing it in logs or reprs."""

        return self.qdrant_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()


# Shared preprocessing configuration; runtime Qdrant settings above remain independent.
COLUMN_ALIASES = {
    "subject": ("subject", "title", "ticket_subject"),
    "customer_query": (
        "body",
        "description",
        "query",
        "message",
        "ticket_description",
        "customer_query",
    ),
    "agent_response": ("answer", "response", "resolution", "agent_response"),
    "category": ("type", "category", "ticket_type"),
    "department": ("queue", "department", "team"),
    "priority": ("priority", "urgency"),
    "language": ("language", "lang"),
}
PRIORITIES = {
    "low": "low",
    "1": "low",
    "p1": "low",
    "medium": "medium",
    "normal": "medium",
    "moderate": "medium",
    "2": "medium",
    "p2": "medium",
    "high": "high",
    "3": "high",
    "p3": "high",
    "urgent": "high",
    "critical": "critical",
    "4": "critical",
    "p4": "critical",
    "severe": "critical",
}
CATEGORIES = {
    "incident": "Incident",
    "request": "Request",
    "problem": "Problem",
    "change": "Change",
    "question": "Request",
    "inquiry": "Request",
    "bug": "Problem",
}
LANGUAGES = {
    "en": "en",
    "english": "en",
    "de": "de",
    "german": "de",
    "deutsch": "de",
    "es": "es",
    "spanish": "es",
    "español": "es",
    "fr": "fr",
    "french": "fr",
    "français": "fr",
    "pt": "pt",
    "portuguese": "pt",
    "português": "pt",
}


DEPARTMENTS = {
    name.lower(): name
    for name in (
        "Technical Support",
        "Customer Service",
        "Billing and Payments",
        "Product Support",
        "IT Support",
        "Returns and Exchanges",
        "Sales and Pre-Sales",
        "Human Resources",
        "Service Outages and Maintenance",
        "General Inquiry",
    )
}


class PreprocessingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="PREPROCESS_", extra="ignore", env_ignore_empty=True
    )
    input: Path | None = None
    hf_dataset: str | None = None
    hf_revision: str | None = None
    source_dataset: str = "tobiasbueck/multilingual-customer-support-tickets"
    retrieval_rows: int = Field(default=16000, ge=1)
    dev_rows: int = Field(default=2000, ge=1)
    eval_rows: int = Field(default=2000, ge=1)
    random_seed: int = 42
    target_languages: list[str] = Field(default_factory=lambda: ["en"])
    min_query_chars: int = Field(default=20, ge=1)
    max_query_chars: int = Field(default=20000, ge=1)
    near_dup_threshold: float = Field(default=0.88, gt=0, le=1)
    hash_features: int = Field(default=65536, ge=1024)
    neighbor_batch_size: int = Field(default=256, ge=1, le=2048)
    output_root: Path = Path("data/preprocessing_runs")
    dense_model: str = "sentence-transformers/all-minilm-l6-v2"
    dense_max_tokens: int = Field(default=256, ge=16, le=256)
    sparse_max_tokens: int = Field(default=2048, ge=32, le=8192)
    sparse_issue_fraction: float = Field(default=0.5, gt=0, lt=1)
    column_aliases: dict[str, tuple[str, ...]] = Field(default_factory=lambda: dict(COLUMN_ALIASES))
    priority_map: dict[str, str] = Field(default_factory=lambda: dict(PRIORITIES))
    category_map: dict[str, str] = Field(default_factory=lambda: dict(CATEGORIES))
    department_map: dict[str, str] = Field(default_factory=lambda: dict(DEPARTMENTS))
    language_map: dict[str, str] = Field(default_factory=lambda: dict(LANGUAGES))

    @model_validator(mode="after")
    def valid_bounds(self) -> "PreprocessingSettings":
        if self.max_query_chars < self.min_query_chars:
            raise ValueError("Maximum query length must not be below minimum")
        if self.input is not None and self.hf_dataset:
            raise ValueError("Configure exactly one raw source, not both local and Hugging Face")
        if self.dense_model != "sentence-transformers/all-minilm-l6-v2":
            raise ValueError(
                "Token budget is pinned to the MiniLM tokenizer; other models require a new profile"
            )
        if not self.target_languages:
            raise ValueError("At least one explicit target language is required")
        issue_tokens = int(self.sparse_max_tokens * self.sparse_issue_fraction)
        if min(issue_tokens, self.sparse_max_tokens - issue_tokens) < 8:
            raise ValueError(
                "Sparse text must reserve at least eight reference tokens for each section"
            )
        return self
