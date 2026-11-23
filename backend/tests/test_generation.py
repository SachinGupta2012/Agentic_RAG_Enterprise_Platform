"""Fail-closed grounding, PII handling and explicit provider API contracts."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.generation.config import GenerationSettings
from app.generation.dependencies import get_response_service
from app.generation.schemas import EvidenceSource
from app.generation.service import ResponseService, bound_sources
from app.main import app
from app.providers.chat import (
    Completion,
    HTTPChatClient,
    ProviderError,
    ProviderRateLimited,
    RequestBudget,
)
from app.retrieval.dependencies import get_ticket_store
from app.retrieval.schemas import RetrievalHit, RetrievalSearchResponse
from app.retrieval.store import RetrievalUnavailable

RESOLUTION = "Clear the browser cache and sign in again."
SOURCE = EvidenceSource(
    source_id="S1",
    ticket_id="ticket-1",
    subject="Sign in failure",
    customer_query="Cannot sign in",
    resolution=RESOLUTION,
)


def settings(**updates: object) -> GenerationSettings:
    return GenerationSettings(_env_file=None, groq_api_key="fake-secret", **updates)


def answer(source_id: str = "S1", quote: str = RESOLUTION) -> str:
    return json.dumps(
        {
            "status": "answered",
            "claims": [
                {
                    "text": "Clear your browser cache.",
                    "evidence": [{"source_id": source_id, "quote": quote}],
                }
            ],
        }
    )


class FakeChat:
    def __init__(self, output: str = "") -> None:
        self.output = output or answer()
        self.calls = 0
        self.user = ""
        self.error: Exception | None = None

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        self.user = user
        if self.error:
            raise self.error
        return Completion(self.output, {"total_tokens": 25}, "test-model")


def test_answer_has_verified_citation_and_redacted_context() -> None:
    chat = FakeChat()
    service = ResponseService(settings(), {"groq": chat})
    result = service.generate("Cannot sign in; email me at user@example.com", [SOURCE])
    assert result.status == "answered"
    assert "[ticket-1]" in result.answer
    assert result.claims[0].evidence[0].quote == RESOLUTION
    assert result.usage == {"total_tokens": 25}
    assert result.retrieval_latency_ms is None
    assert "user@example.com" not in chat.user
    assert result.query_pii_redactions["email"] == 1


@pytest.mark.parametrize(
    "output",
    [
        answer("S9"),
        answer(quote="Invented support quotation."),
        '{"status":"answered","claims":[]}',
        "not JSON",
    ],
)
def test_unverified_answer_is_never_returned(output: str) -> None:
    result = ResponseService(settings(), {"groq": FakeChat(output)}).generate(
        "Cannot sign in", [SOURCE]
    )
    assert result.status == "insufficient_evidence"
    assert result.reason == "unverified_evidence"
    assert result.claims == []


def test_no_resolution_skips_model_and_model_can_abstain() -> None:
    chat = FakeChat('{"status":"insufficient_evidence","claims":[]}')
    service = ResponseService(settings(), {"groq": chat})
    assert service.generate("Cannot sign in", []).llm_called is False
    assert chat.calls == 0
    assert service.generate("Cannot sign in", [SOURCE]).reason == "model_abstained"
    assert chat.calls == 1


def test_context_is_redacted_deduplicated_and_bounded() -> None:
    source = SOURCE.model_copy(
        update={"resolution": "Contact person@example.com then " + "step " * 1000}
    )
    sources = bound_sources([source, source], 2000)
    assert len(sources) == 1
    assert "person@example.com" not in sources[0].resolution
    assert len(json.dumps([s.model_dump() for s in sources], ensure_ascii=False)) <= 2000


def test_budget_is_shared_and_returns_retry_hint_without_sleeping() -> None:
    budget = RequestBudget(settings(), clock=lambda: 100)
    budget.reserve(4000)
    with pytest.raises(ProviderRateLimited) as error:
        budget.reserve(4000)
    assert error.value.retry_after == 60


def test_budget_releases_unused_output_reservation() -> None:
    budget = RequestBudget(settings(), clock=lambda: 100)
    reservation = budget.reserve(4000)
    budget.reconcile(reservation, 500)
    budget.reserve(4000)


def test_citation_must_quote_resolution_not_customer_request() -> None:
    result = ResponseService(
        settings(), {"groq": FakeChat(answer(quote="Cannot sign in to my account."))}
    ).generate(
        "Cannot sign in",
        [SOURCE.model_copy(update={"customer_query": "Cannot sign in to my account."})],
    )
    assert result.reason == "unverified_evidence"


def test_provider_retries_one_short_server_failure_but_rejects_truncation() -> None:
    statuses = [500, 200]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("call")
        code = statuses.pop(0)
        return httpx.Response(
            code, json={"choices": [{"finish_reason": "stop", "message": {"content": answer()}}]}
        )

    chat = HTTPChatClient(
        "groq", settings(), transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    assert chat.complete("system", "user").text == answer()
    assert len(calls) == 2

    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "length", "message": {"content": answer()}}]}
        )

    with pytest.raises(ProviderError, match="incomplete"):
        HTTPChatClient("groq", settings(), transport=httpx.MockTransport(truncated)).complete(
            "system", "user"
        )


@pytest.mark.parametrize("provider", ["groq", "gemini", "cloudflare"])
def test_provider_request_and_response_contracts(provider: str) -> None:
    config = settings(
        enabled_providers=[provider],
        gemini_api_key="fake-secret",
        cloudflare_api_token="fake-secret",
        cloudflare_account_id="test-account",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "fake-secret" not in str(request.url)
        payload = json.loads(request.content)
        if provider == "gemini":
            assert request.headers["x-goog-api-key"] == "fake-secret"
            assert payload["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"finishReason": "STOP", "content": {"parts": [{"text": answer()}]}}
                    ]
                },
            )
        assert request.headers["authorization"] == "Bearer fake-secret"
        assert payload["response_format"]["type"] == "json_object"
        if provider == "cloudflare":
            return httpx.Response(200, json={"success": True, "result": {"response": answer()}})
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer()}}]}
        )

    chat = HTTPChatClient(provider, config, transport=httpx.MockTransport(handler))
    assert chat.complete("system", "user").text == answer()


def test_cloudflare_accepts_already_parsed_json_mode_response() -> None:
    config = settings(
        enabled_providers=["cloudflare"],
        cloudflare_api_token="fake",
        cloudflare_account_id="account",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"success": True, "result": {"response": json.loads(answer())}}
        )

    client = HTTPChatClient("cloudflare", config, transport=httpx.MockTransport(handler))
    assert json.loads(client.complete("system", "user").text) == json.loads(answer())


def test_quota_errors_and_timeouts_do_not_retry_or_expose_provider_body() -> None:
    calls: list[str] = []

    def limited(request: httpx.Request) -> httpx.Response:
        calls.append("call")
        return httpx.Response(
            429,
            headers={"retry-after": "120"},
            json={"error": "fake-secret sensitive ticket content"},
        )

    chat = HTTPChatClient("groq", settings(), transport=httpx.MockTransport(limited))
    with pytest.raises(ProviderRateLimited) as error:
        chat.complete("system", "user")
    assert error.value.retry_after == 120
    with pytest.raises(ProviderRateLimited):
        chat.complete("system", "user")
    assert len(calls) == 1
    assert "sensitive" not in str(error.value)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fake-secret", request=request)

    with pytest.raises(ProviderError, match="uncertain"):
        HTTPChatClient("groq", settings(), transport=httpx.MockTransport(timeout)).complete(
            "system", "user"
        )


class FakeStore:
    error: Exception | None = None

    def search(self, request: object) -> RetrievalSearchResponse:
        if self.error:
            raise self.error
        return RetrievalSearchResponse(
            query="Cannot sign in",
            count=1,
            hits=[
                RetrievalHit(
                    ticket_id="ticket-1", subject="Sign in", snippet="Cannot sign in", score=0.1
                )
            ],
            dense_model="dense",
            sparse_model="bm25",
            collection="support_tickets_v1",
            latency_ms=1,
        )

    def fetch_payloads(self, ids: object) -> dict:
        return {
            "ticket-1": {
                "split": "retrieval",
                "subject": "Sign in",
                "customer_query": "Cannot sign in",
                "agent_response": RESOLUTION,
            }
        }


def test_non_retrieval_payload_cannot_reach_the_model() -> None:
    from app.generation.schemas import ResponseRequest

    class WrongSplitStore(FakeStore):
        def fetch_payloads(self, ids: object) -> dict:
            return {"ticket-1": {"split": "dev", "agent_response": RESOLUTION}}

    chat = FakeChat()
    result = ResponseService(settings(), {"groq": chat}).respond(
        ResponseRequest(query="Cannot sign in"), WrongSplitStore()
    )
    assert result.reason == "no_context"
    assert result.sources == []
    assert chat.calls == 0


def test_api_generates_maps_errors_and_rejects_disabled_fallback() -> None:
    chat = FakeChat()
    service = ResponseService(settings(), {"groq": chat})
    store = FakeStore()
    app.dependency_overrides[get_response_service] = lambda: service
    app.dependency_overrides[get_ticket_store] = lambda: store
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/responses/generate", json={"query": "Cannot sign in"})
            assert response.status_code == 200
            assert response.json()["claims"][0]["evidence"][0]["source_id"] == "S1"
            assert (
                client.post(
                    "/api/v1/responses/generate",
                    json={"query": "Cannot sign in", "provider": "gemini"},
                ).status_code
                == 503
            )
            chat.error = ProviderRateLimited(90)
            response = client.post("/api/v1/responses/generate", json={"query": "Cannot sign in"})
            assert response.status_code == 429
            assert response.headers["retry-after"] == "90"
            store.error = RetrievalUnavailable("private details")
            response = client.post("/api/v1/responses/generate", json={"query": "Cannot sign in"})
            assert response.status_code == 503
            assert "private" not in response.text
    finally:
        app.dependency_overrides.clear()
