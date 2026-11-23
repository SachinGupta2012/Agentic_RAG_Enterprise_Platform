"""Pydantic schemas for the retrieval API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["dense", "sparse", "hybrid"]


class RetrievalFilters(BaseModel):
    """Optional metadata filters applied to both dense and sparse retrieval."""

    category: str | None = None
    department: str | None = None
    priority: str | None = None
    language: str | None = None

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value.title() if value else None

    @field_validator("department")
    @classmethod
    def normalize_department(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("priority", "language")
    @classmethod
    def normalize_lowercase(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        return value or None


class RetrievalSearchRequest(BaseModel):
    """A hybrid retrieval request."""

    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    prefetch_limit: int | None = Field(default=None, ge=10, le=200)
    mode: RetrievalMode = "hybrid"
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("query must contain at least 3 characters")
        return value


class RetrievalHit(BaseModel):
    """A single retrieved historical ticket."""

    ticket_id: str
    score: float
    score_type: Literal["rrf", "dense", "sparse"] = "rrf"
    subject: str
    snippet: str
    resolution_snippet: str | None = None
    category: str | None = None
    department: str | None = None
    priority: str | None = None
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_ticket_id: str | None = None
    source_dataset: str | None = None
    source_row_index: int | None = None
    source_version: str | None = None
    pii_redactions: dict[str, int] = Field(default_factory=dict)


class RetrievalSearchResponse(BaseModel):
    """Results from a hybrid retrieval request."""

    query: str
    hits: list[RetrievalHit]
    count: int
    retrieval_mode: Literal["dense", "sparse", "hybrid_rrf"] = "hybrid_rrf"
    dense_model: str
    sparse_model: str
    collection: str
    latency_ms: float
    query_pii_redactions: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RetrievalHealthResponse(BaseModel):
    """Readiness details for the retrieval index."""

    status: Literal["ok", "unavailable"]
    collection: str
    document_count: int | None = None
    dense_model: str
    sparse_model: str
    detail: str | None = None
