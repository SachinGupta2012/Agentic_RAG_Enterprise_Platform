"""Tests for the LLM relevance judge with mocked providers."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.evaluation.candidates import PoolDocument
from app.evaluation.config import EvaluationSettings
from app.evaluation.judge import (
    GeminiChatClient,
    GroqChatClient,
    JudgeConfigurationError,
    JudgeOutputError,
    JudgePermanentError,
    JudgeTransientError,
    LLMJudge,
    build_chat_client,
    build_judge_messages,
    parse_judgment,
)

VALID_JSON = '{"relevance": 2, "reason": "Same outage with a working fix."}'


def _document() -> PoolDocument:
    return PoolDocument(
        doc_id="doc-a",
        subject="Portal outage",
        customer_query="The account portal is offline.",
        agent_response="Restart the gateway and clear the cache.",
        category="ZZCategory",
        department="ZZDepartment",
        priority="ZZPriority",
        language="en",
        tags=("ZZTag",),
    )


def _settings(**overrides: Any) -> EvaluationSettings:
    payload: dict[str, Any] = {
        "judge_provider": "groq",
        "judge_model": "test-model",
        "judge_max_retries": 2,
        "judge_output_retries": 1,
        "judge_backoff_base_seconds": 0.01,
        "judge_backoff_max_seconds": 0.02,
        "judge_requests_per_minute": 0,
        "groq_api_key": "test-key",
        "gemini_api_key": "test-key",
    }
    payload.update(overrides)
    return EvaluationSettings(**payload)


class FlakyClient:
    """Chat client that yields scripted results."""

    provider = "groq"
    model = "test-model"

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return str(item)


def test_prompt_excludes_metadata_and_fences_ticket() -> None:
    system, user = build_judge_messages("portal offline", _document())

    assert "untrusted" in system.lower()
    assert "<ticket>" in user and "</ticket>" in user
    assert "portal offline" in user
    assert "ZZCategory" not in user
    assert "ZZDepartment" not in user
    assert "ZZPriority" not in user
    assert "ZZTag" not in user


def test_prompt_truncates_long_documents() -> None:
    document = _document()
    long_document = PoolDocument(
        doc_id=document.doc_id,
        subject=document.subject,
        customer_query="word " * 5000,
        agent_response=document.agent_response,
        category=document.category,
        department=document.department,
        priority=document.priority,
        language=document.language,
        tags=document.tags,
    )

    _, user = build_judge_messages("portal offline", long_document, max_document_chars=300)

    assert len(user) < 2000


def test_parse_judgment_accepts_plain_json() -> None:
    payload = parse_judgment(VALID_JSON)

    assert payload.relevance == 2
    assert "outage" in payload.reason


def test_parse_judgment_accepts_fenced_json() -> None:
    payload = parse_judgment(f"```json\n{VALID_JSON}\n```")

    assert payload.relevance == 2


def test_parse_judgment_rejects_out_of_range_grade() -> None:
    with pytest.raises(JudgeOutputError):
        parse_judgment('{"relevance": 3, "reason": "no"}')


def test_parse_judgment_rejects_non_json() -> None:
    with pytest.raises(JudgeOutputError):
        parse_judgment("I think it is relevant.")


def test_groq_client_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_JSON}}]})

    client = GroqChatClient(
        "test-key", "test-model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    assert client.complete("system", "user") == VALID_JSON


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(429, JudgeTransientError), (500, JudgeTransientError), (401, JudgePermanentError)],
)
def test_groq_client_classifies_errors(status_code: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "nope"})

    client = GroqChatClient(
        "secret-key", "test-model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(expected) as excinfo:
        client.complete("system", "user")

    assert "secret-key" not in str(excinfo.value)


def test_groq_client_sends_reasoning_effort() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_JSON}}]})

    client = GroqChatClient(
        "test-key",
        "test-model",
        reasoning_effort="low",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.complete("system", "user")

    assert captured["reasoning_effort"] == "low"


def test_groq_truncated_json_is_retryable_output_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Failed to generate JSON.",
                    "code": "json_validate_failed",
                }
            },
        )

    client = GroqChatClient(
        "test-key", "test-model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(JudgeOutputError):
        client.complete("system", "user")


def test_groq_auth_failure_is_a_configuration_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    client = GroqChatClient(
        "test-key", "test-model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(JudgeConfigurationError):
        client.complete("system", "user")


def test_gemini_client_keeps_key_out_of_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "test-key" not in str(request.url)
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": VALID_JSON}]}}]},
        )

    client = GeminiChatClient(
        "test-key", "test-model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    assert client.complete("system", "user") == VALID_JSON


def test_build_chat_client_requires_key() -> None:
    with pytest.raises(JudgePermanentError, match="GROQ_API_KEY"):
        build_chat_client(_settings(groq_api_key=""))


def test_judge_retries_transient_failures() -> None:
    client = FlakyClient([JudgeTransientError("429"), VALID_JSON])
    judge = LLMJudge(client, _settings(), sleep=lambda _: None)

    result = judge.judge("portal offline", _document())

    assert result.relevance == 2
    assert client.calls == 2


def test_judge_raises_after_transient_retries() -> None:
    client = FlakyClient([JudgeTransientError("429")] * 5)
    judge = LLMJudge(client, _settings(judge_max_retries=1), sleep=lambda _: None)

    with pytest.raises(JudgeTransientError):
        judge.judge("portal offline", _document())


def test_judge_retries_malformed_output() -> None:
    client = FlakyClient(["not json", VALID_JSON])
    judge = LLMJudge(client, _settings(), sleep=lambda _: None)

    assert judge.judge("portal offline", _document()).relevance == 2
    assert client.calls == 2


def test_judge_raises_after_output_retries() -> None:
    client = FlakyClient(["not json", "still not json", VALID_JSON])
    judge = LLMJudge(client, _settings(judge_output_retries=1), sleep=lambda _: None)

    with pytest.raises(JudgeOutputError):
        judge.judge("portal offline", _document())


def test_judge_does_not_retry_permanent_errors() -> None:
    client = FlakyClient([JudgePermanentError("401"), VALID_JSON])
    judge = LLMJudge(client, _settings(), sleep=lambda _: None)

    with pytest.raises(JudgePermanentError):
        judge.judge("portal offline", _document())
    assert client.calls == 1


def test_judge_throttles_requests() -> None:
    sleeps: list[float] = []
    client = FlakyClient([VALID_JSON, VALID_JSON])
    judge = LLMJudge(
        client,
        _settings(judge_requests_per_minute=60),
        sleep=sleeps.append,
        monotonic=lambda: 100.0,
    )

    judge.judge("portal offline", _document())
    judge.judge("portal offline", _document())

    assert sleeps == [pytest.approx(1.0)]


def test_judgment_record_provenance() -> None:
    client = FlakyClient([VALID_JSON])
    judge = LLMJudge(client, _settings(), sleep=lambda _: None)

    record = judge.judge("portal offline", _document()).to_record("dev-q0001", "doc-a")

    assert record.query_id == "dev-q0001"
    assert record.doc_id == "doc-a"
    assert record.relevance == 2
    assert record.judgment_source == "llm"
    assert record.judge_provider == "groq"
    assert record.prompt_version == "judge-v1"
