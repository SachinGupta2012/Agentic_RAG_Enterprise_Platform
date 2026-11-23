"""API schemas and strict model output contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.config import Provider
from app.retrieval.schemas import RetrievalSearchRequest


class ResponseRequest(RetrievalSearchRequest):
    top_k: int = Field(default=3, ge=1, le=5)
    provider: Provider | None = None


class EvidenceSource(BaseModel):
    source_id: str
    ticket_id: str
    subject: str
    customer_query: str
    resolution: str


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=8)
    quote: str = Field(min_length=12, max_length=600)


class SupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=3, max_length=1200)
    evidence: list[EvidenceQuote] = Field(min_length=1, max_length=3)


class GeneratedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_evidence"]
    claims: list[SupportedClaim] = Field(max_length=5)


class ResponseResult(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    claims: list[SupportedClaim]
    sources: list[EvidenceSource]
    reason: Literal["supported_claims", "no_context", "model_abstained", "unverified_evidence"]
    provider: Provider
    model: str
    reported_model: str | None = None
    prompt_version: str = "response-v1"
    context_sha256: str
    llm_called: bool
    retrieval_mode: str
    collection: str
    retrieval_latency_ms: float | None
    generation_latency_ms: float
    total_latency_ms: float
    usage: dict[str, int] = Field(default_factory=dict)
    query_pii_redactions: dict[str, int] = Field(default_factory=dict)
