"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
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

    # Retrieval behaviour.
    retrieval_prefetch_limit: int = 50
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
