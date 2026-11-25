"""Versioned graph contracts. Credentials and dependencies never enter state."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.generation.config import Provider
from app.generation.schemas import EvidenceSource, ResponseRequest, ResponseResult

Node = Literal[
    "triage", "retrieval", "investigation", "resolution", "qa", "escalation", "knowledge", "end"
]
Outcome = Literal["answered", "needs_clarification", "escalated", "unavailable"]


class OrchestrationRequest(ResponseRequest):
    model_config = ConfigDict(extra="forbid")


class Event(BaseModel):
    event_id: str
    node: Node
    route: Node
    reason: str
    duration_ms: float
    llm_attempts: int = 0
    tokens_reserved: int = 0
    read_operations: int = 0
    context_sha256: str | None = None
    qa_checks: dict[str, bool] = Field(default_factory=dict)
    draft_usage: dict[str, int] = Field(default_factory=dict)


def merge_events(left: list[Event], right: list[Event]) -> list[Event]:
    """Deterministic and idempotent for replayed, identical event IDs."""
    events = {event.event_id: event for event in [*left, *right]}
    return [events[key] for key in sorted(events)][-32:]


class KnowledgeProposal(BaseModel):
    proposal_id: str
    kind: Literal["reuse_candidate"] = "reuse_candidate"
    ticket_ids: list[str]
    status: Literal["unreviewed"] = "unreviewed"
    publication_allowed: Literal[False] = False


class WorkData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_version: Literal["1"] = "1"
    run_id: str
    request: OrchestrationRequest
    original_query: str
    query: str
    provider: Provider
    model: str
    intent: str = "general_support"
    urgency: Literal["normal", "urgent"] = "normal"
    language: str = "und"
    redaction_counts: dict[str, int] = Field(default_factory=dict)
    next_node: Node = "triage"
    reason: str = "started"
    sources: list[EvidenceSource] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    context_version: int = 0
    draft_version: int | None = None
    draft: ResponseResult | None = None
    qa_checks: dict[str, bool] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    retrievals: int = 0
    drafts: int = 0
    investigations: int = 0
    collection: str = "support_tickets_v1"
    retrieval_mode: str = "hybrid_rrf"
    retrieval_ms: float = 0
    outcome: Outcome | None = None
    proposal: KnowledgeProposal | None = None
    http_status: int = 200
    retry_after: int | None = None


class WorkflowState(TypedDict):
    data: WorkData
    events: Annotated[list[Event], merge_events]


class OrchestrationResult(BaseModel):
    run_id: str
    graph_version: str = "orchestration-v1"
    outcome: Outcome
    reason: str
    response: ResponseResult | None
    qa_checks: dict[str, bool]
    semantic_quality_verified: Literal[False] = False
    knowledge_proposal: KnowledgeProposal | None
    events: list[Event]
    counters: dict[str, int]
    total_latency_ms: float
    retry_after: int | None = None
    handoff_summary: str | None = None
