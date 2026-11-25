"""Seven small sequential LangGraph nodes over the existing Phase 2/3 services."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langsmith.run_helpers import tracing_context

from app.agents.config import OrchestrationSettings
from app.agents.state import (
    Event,
    KnowledgeProposal,
    Node,
    OrchestrationRequest,
    OrchestrationResult,
    WorkData,
    WorkflowState,
    merge_events,
)
from app.core.io_budget import BudgetExceeded, RunBudget, active_budget
from app.generation.schemas import EvidenceSource
from app.generation.service import PROMPT_VERSION, ResponseService, TicketStore, bound_sources
from app.providers.chat import ProviderError, ProviderNotConfigured, ProviderRateLimited
from app.retrieval.pii import redact_pii
from app.retrieval.schemas import RetrievalFilters, RetrievalSearchRequest
from app.retrieval.store import RetrievalError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    settings: OrchestrationSettings
    store: TicketStore
    generation: ResponseService
    budget: RunBudget
    filters: RetrievalFilters  # Trusted, already narrowed API access scope; never model-controlled.


def update(data: WorkData, **changes: Any) -> WorkData:
    return WorkData.model_validate({**data.model_dump(), **changes})


def triage(data: WorkData, context: RunContext) -> WorkData:
    query = data.query.lower()
    if len(re.findall(r"\w+", query)) < 2 or query.strip(".!?") in {
        "please help",
        "help me",
        "hello there",
    }:
        return update(
            data,
            next_node="escalation",
            outcome="needs_clarification",
            reason="more_details_needed",
        )
    if re.match(r"(?:please )?(?:write (?:a |me a )?(?:poem|story)|tell (?:me )?a joke)\b", query):
        return update(
            data, next_node="escalation", outcome="escalated", reason="unsupported_request"
        )
    intent = "general_support"
    for label, pattern in [
        ("billing", r"\b(invoice|refund|billing|payment)\b"),
        ("account", r"\b(account|password|login|portal)\b"),
        ("technical", r"\b(error|crash|outage|bug)\b"),
    ]:
        if re.search(pattern, query):
            intent = label
            break
    return update(
        data,
        next_node="retrieval",
        intent=intent,
        urgency="urgent" if re.search(r"\b(urgent|outage)\b", query) else "normal",
        language=context.filters.language or "und",
        reason="supported_request",
    )


def can_investigate(data: WorkData, context: RunContext) -> bool:
    return context.settings.investigation_enabled and data.investigations == 0


def retrieval(data: WorkData, context: RunContext) -> WorkData:
    started = time.perf_counter()
    result = context.store.search(
        RetrievalSearchRequest(
            query=data.query,
            top_k=data.request.top_k,
            mode=data.request.mode,
            prefetch_limit=data.request.prefetch_limit,
            filters=context.filters.model_copy(deep=True),
        )
    )
    context.budget.remaining()
    payloads = (
        context.store.fetch_payloads([hit.ticket_id for hit in result.hits]) if result.hits else {}
    )
    sources = [
        EvidenceSource(
            source_id="",
            ticket_id=hit.ticket_id,
            subject=str(payloads[hit.ticket_id].get("subject") or ""),
            customer_query=str(payloads[hit.ticket_id].get("customer_query") or ""),
            resolution=str(payloads[hit.ticket_id].get("agent_response") or ""),
        )
        for hit in result.hits
        if payloads.get(hit.ticket_id, {}).get("split") == "retrieval"
    ]
    sources = bound_sources(sources, context.generation.settings.context_chars)
    issues: dict[str, set[str]] = {}
    for source in sources:
        if len(source.customer_query) >= 20:
            issues.setdefault(source.customer_query.casefold(), set()).add(
                source.resolution.casefold()
            )
    variance = any(len(resolutions) > 1 for resolutions in issues.values())
    reason = (
        "resolution_variance"
        if variance
        else "evidence_available"
        if sources
        else "no_usable_evidence"
    )
    target: Node = "resolution"
    if not sources or variance:
        target = "investigation" if can_investigate(data, context) else "escalation"
    return update(
        data,
        sources=sources,
        ranks={hit.ticket_id: i for i, hit in enumerate(result.hits, 1)},
        context_version=data.context_version + 1,
        draft=None,
        draft_version=None,
        retrieval_mode=result.retrieval_mode,
        collection=result.collection,
        retrieval_ms=data.retrieval_ms + (time.perf_counter() - started) * 1000,
        next_node=target,
        reason=reason,
        findings=[reason],
    )


def investigation(data: WorkData, context: RunContext) -> WorkData:
    if not can_investigate(data, context):
        return update(data, next_node="escalation", reason="investigation_exhausted")
    # Only remove conversational boilerplate; never invent facts or relax filters.
    refined = re.sub(
        r"^(?:hi[,! ]+|hello[,! ]+|please\s+|help me with\s+)+", "", data.query, flags=re.I
    ).strip()
    if (
        refined.casefold() != data.query.casefold()
        and len(refined) >= 3
        and data.retrievals < context.settings.max_retrievals
    ):
        return update(
            data, query=refined, investigations=1, next_node="retrieval", reason="query_refined"
        )
    if data.sources and data.reason == "qa_rejected" and data.drafts < context.settings.max_drafts:
        return update(data, investigations=1, next_node="resolution", reason="one_draft_repair")
    return update(
        data,
        investigations=1,
        next_node="escalation",
        reason="resolution_variance_unresolved"
        if data.reason == "resolution_variance"
        else "insufficient_evidence",
    )


def resolution(data: WorkData, context: RunContext) -> WorkData:
    draft = context.generation.generate(
        data.original_query,
        data.sources,
        provider=data.provider,
        retrieval_mode=data.retrieval_mode,
        collection=data.collection,
        retrieval_latency_ms=data.retrieval_ms,
    )
    draft.query_pii_redactions = dict(data.redaction_counts)
    target: Node = "qa" if draft.status == "answered" else "escalation"
    if draft.reason == "unverified_evidence" and can_investigate(data, context):
        target = "investigation"
    return update(
        data,
        draft=draft,
        draft_version=data.context_version,
        next_node=target,
        reason="qa_rejected" if draft.reason == "unverified_evidence" else draft.reason,
    )


def qa(data: WorkData, context: RunContext) -> WorkData:
    draft = data.draft
    if draft is None:
        return update(data, next_node="escalation", reason="missing_draft")
    sources = {source.source_id: source for source in draft.sources}
    original = {source.ticket_id: source for source in data.sources}
    expected_context = json.dumps(
        {"question": data.original_query, "sources": [s.model_dump() for s in draft.sources]},
        ensure_ascii=False,
    )
    citations_valid = bool(draft.claims) and all(
        claim.evidence
        and all(
            evidence.source_id in sources
            and len(" ".join(evidence.quote.split())) >= 12
            and " ".join(evidence.quote.split()) in sources[evidence.source_id].resolution
            for evidence in claim.evidence
        )
        for claim in draft.claims
    )
    lines = [
        claim.text
        + " "
        + " ".join(
            f"[{ticket}]"
            for ticket in dict.fromkeys(
                sources[e.source_id].ticket_id for e in claim.evidence if e.source_id in sources
            )
        )
        for claim in draft.claims
    ]
    checks = {
        "answered": draft.status == "answered" and draft.reason == "supported_claims",
        "context_version": data.draft_version == data.context_version,
        "provenance": draft.provider == data.provider
        and draft.model == data.model
        and draft.prompt_version == PROMPT_VERSION,
        "source_identity": bool(draft.sources)
        and len(sources) == len(draft.sources)
        and all(
            source.ticket_id in original and source == original[source.ticket_id]
            for source in draft.sources
        ),
        "context_hash": draft.context_sha256
        == hashlib.sha256(expected_context.encode()).hexdigest(),
        "citations_and_quotes": citations_valid,
        "answer_matches_claims": draft.answer == "\n\n".join(lines),
    }
    passed = all(checks.values())
    return update(
        data,
        qa_checks=checks,
        outcome="answered" if passed else None,
        next_node=("knowledge" if context.settings.knowledge_enabled else "end")
        if passed
        else "investigation"
        if can_investigate(data, context)
        else "escalation",
        reason="qa_passed" if passed else "qa_rejected",
    )


def escalation(data: WorkData, context: RunContext) -> WorkData:
    return update(
        data,
        next_node="end",
        outcome=data.outcome if data.outcome != "answered" and data.outcome else "escalated",
        draft=None,
        proposal=None,
    )


def knowledge(data: WorkData, context: RunContext) -> WorkData:
    if (
        data.outcome != "answered"
        or not data.qa_checks
        or not all(data.qa_checks.values())
        or data.draft is None
    ):
        return update(data, next_node="escalation", reason="knowledge_requires_qa")
    by_id = {s.source_id: s.ticket_id for s in data.draft.sources}
    tickets = sorted({by_id[e.source_id] for claim in data.draft.claims for e in claim.evidence})
    key = hashlib.sha256(json.dumps([tickets, PROMPT_VERSION], sort_keys=True).encode()).hexdigest()
    return update(
        data, proposal=KnowledgeProposal(proposal_id=key, ticket_ids=tickets), next_node="end"
    )


OPERATIONS: dict[Node, Callable[[WorkData, RunContext], WorkData]] = {
    "triage": triage,
    "retrieval": retrieval,
    "investigation": investigation,
    "resolution": resolution,
    "qa": qa,
    "escalation": escalation,
    "knowledge": knowledge,
}


def wrapped(node: Node) -> Callable[..., Any]:
    async def execute(state: WorkflowState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        context, data = runtime.context, state["data"]
        started = time.perf_counter()
        try:
            if node != "escalation":
                context.budget.remaining()
            if node == "retrieval":
                if data.retrievals >= context.settings.max_retrievals:
                    raise BudgetExceeded("retrieval_budget_exhausted")
                data = update(data, retrievals=data.retrievals + 1)
            if node == "resolution":
                if data.drafts >= context.settings.max_drafts:
                    raise BudgetExceeded("draft_budget_exhausted")
                data = update(data, drafts=data.drafts + 1)
            try:
                result = await asyncio.to_thread(OPERATIONS[node], data, context)
            except Exception:
                if node != "knowledge" or data.outcome != "answered":
                    raise
                result = update(
                    data, next_node="end", proposal=None, reason="knowledge_proposal_failed"
                )
            if node not in ("escalation", "knowledge"):
                context.budget.remaining()
        except BudgetExceeded as exc:
            result = (
                update(data, next_node="end", proposal=None, reason="knowledge_deferred")
                if node == "knowledge" and data.outcome == "answered"
                else update(
                    data,
                    next_node="escalation",
                    outcome="unavailable",
                    reason=str(exc),
                    http_status=503,
                )
            )
        except ProviderRateLimited as exc:
            result = update(
                data,
                next_node="escalation",
                outcome="unavailable",
                reason="provider_quota",
                http_status=429,
                retry_after=exc.retry_after,
            )
        except (ProviderError, RetrievalError) as exc:
            reason = (
                "provider_configuration"
                if isinstance(exc, ProviderNotConfigured)
                else "provider_unavailable"
                if isinstance(exc, ProviderError)
                else "retrieval_unavailable"
            )
            result = update(
                data, next_node="escalation", outcome="unavailable", reason=reason, http_status=503
            )
        except Exception:
            if node != "knowledge" or data.outcome != "answered":
                logger.error(
                    "orchestration_node_failed", extra={"run_id": data.run_id, "node": node}
                )
                raise RuntimeError("Unexpected orchestration failure") from None
            result = update(
                data, next_node="end", proposal=None, reason="knowledge_proposal_failed"
            )
        event = Event(
            event_id=f"{data.run_id}:{len(state['events']):03d}",
            node=node,
            route=result.next_node,
            reason=result.reason,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            llm_attempts=context.budget.llm_attempts,
            tokens_reserved=context.budget.tokens_reserved,
            read_operations=context.budget.reads,
            context_sha256=result.draft.context_sha256 if result.draft else None,
            qa_checks=result.qa_checks,
            draft_usage=result.draft.usage if node == "resolution" and result.draft else {},
        )
        logger.info("orchestration_node", extra={"run_id": data.run_id, **event.model_dump()})
        return {"data": result, "events": [event]}

    return execute


class Orchestrator:
    def __init__(self, settings: OrchestrationSettings) -> None:
        self.settings = settings
        builder = StateGraph(WorkflowState, context_schema=RunContext)
        for node in OPERATIONS:
            builder.add_node(node, wrapped(node))
            builder.add_conditional_edges(
                node,
                lambda state: state["data"].next_node,
                {name: name for name in OPERATIONS} | {"end": END},
            )
        builder.add_edge(START, "triage")
        self.graph = builder.compile()

    async def run(
        self,
        request: OrchestrationRequest,
        store: TicketStore,
        generation: ResponseService,
        *,
        trusted_filters: RetrievalFilters | None = None,
    ) -> tuple[OrchestrationResult, int]:
        started = time.perf_counter()
        query, redactions = redact_pii(request.query)
        filters = request.filters.model_copy(deep=True)
        if trusted_filters:
            for name, value in trusted_filters.model_dump().items():
                if value is not None:
                    if getattr(filters, name) not in (None, value):
                        raise PermissionError(
                            "Requested filters conflict with trusted access scope"
                        )
                    setattr(filters, name, value)
        safe_request = request.model_copy(update={"query": query, "filters": filters})
        provider = request.provider or generation.settings.provider
        data = WorkData(
            run_id=str(uuid4()),
            request=safe_request,
            original_query=query,
            query=query,
            provider=provider,
            model=generation.settings.model_for(provider),
            redaction_counts=redactions.as_dict(),
        )
        budget = RunBudget(
            self.settings.timeout_seconds,
            self.settings.max_llm_attempts,
            self.settings.max_tokens,
            reads=self.settings.max_retrievals * 2,
        )
        context = RunContext(
            self.settings, store, generation, budget, filters.model_copy(deep=True)
        )
        token = active_budget.set(budget)
        events: list[Event] = []
        try:
            # Prevent implicit environment-enabled LangSmith exports of state/business text.
            async with asyncio.timeout(self.settings.timeout_seconds):
                with tracing_context(enabled=False):
                    async for chunk in self.graph.astream(
                        {"data": data, "events": []},
                        config={"recursion_limit": self.settings.max_steps, "callbacks": []},
                        context=context,
                        stream_mode="updates",
                    ):
                        for values in chunk.values():
                            data = values["data"]
                            events = merge_events(events, values["events"])
                        if time.perf_counter() - started >= self.settings.timeout_seconds:
                            raise BudgetExceeded("deadline_exceeded")
        except (GraphRecursionError, BudgetExceeded, TimeoutError) as exc:
            reason = (
                "step_limit_exhausted"
                if isinstance(exc, GraphRecursionError)
                else str(exc)
                if isinstance(exc, BudgetExceeded)
                else "deadline_exceeded"
            )
            data = (
                update(data, proposal=None, reason="knowledge_deferred")
                if data.outcome == "answered"
                else update(
                    data,
                    outcome="unavailable",
                    draft=None,
                    proposal=None,
                    reason=reason,
                    http_status=503,
                )
            )
        finally:
            budget.cancel()
            active_budget.reset(token)
        result = OrchestrationResult(
            run_id=data.run_id,
            outcome=data.outcome or "unavailable",
            reason=data.reason,
            response=data.draft if data.outcome == "answered" else None,
            qa_checks=data.qa_checks,
            knowledge_proposal=data.proposal,
            events=events,
            retry_after=data.retry_after,
            counters={
                "retrievals": data.retrievals,
                "drafts": data.drafts,
                "investigations": data.investigations,
                "llm_http_attempts": budget.llm_attempts,
                "tokens_reserved": budget.tokens_reserved,
                "read_operations": budget.reads,
            },
            total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            handoff_summary=None
            if data.outcome == "answered"
            else "No verified answer was produced. Use the reason code to clarify or route this request.",
        )
        logger.info(
            "orchestration_complete",
            extra={
                "run_id": data.run_id,
                "outcome": result.outcome,
                "provider": data.provider,
                "model": data.model,
                "graph_version": result.graph_version,
                "prompt_version": PROMPT_VERSION,
                "context_sha256": data.draft.context_sha256 if data.draft else None,
                "usage": data.draft.usage if data.draft else {},
                "total_latency_ms": result.total_latency_ms,
                **result.counters,
            },
        )
        return result, data.http_status
