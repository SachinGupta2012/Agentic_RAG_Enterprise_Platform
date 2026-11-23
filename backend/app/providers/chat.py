"""Small runtime provider adapters. No dependency on evaluation/judging code."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from app.generation.config import GenerationSettings, Provider


class ProviderError(Exception):
    """Sanitized provider failure; never contains a raw provider error body."""


class ProviderNotConfigured(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    def __init__(self, retry_after: float = 60) -> None:
        super().__init__("Generation quota unavailable; retry later.")
        self.retry_after = max(1, int(retry_after + 0.999))


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    reported_model: str | None = None


class ChatBackend(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


@dataclass
class Reservation:
    timestamp: float
    tokens: int


def duration(value: str) -> float:
    try:
        return max(0, float(value))
    except ValueError:
        units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
        return sum(float(n) * units[u] for n, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", value))


class RequestBudget:
    """Thread-safe per-process admission control; remote quotas remain authoritative."""

    def __init__(
        self, settings: GenerationSettings, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settings = settings
        self.clock = clock
        self._lock = threading.Lock()
        self._events: deque[Reservation] = deque()
        self._cooldown = 0.0

    def reserve(self, tokens: int) -> Reservation:
        with self._lock:
            now = self.clock()
            while self._events and now - self._events[0].timestamp >= 60:
                self._events.popleft()
            if now < self._cooldown:
                raise ProviderRateLimited(self._cooldown - now)
            if tokens > self.settings.tokens_per_minute:
                raise ProviderError("Configured context exceeds the per-request token budget.")
            if self._events and (
                len(self._events) >= self.settings.requests_per_minute
                or sum(event.tokens for event in self._events) + tokens
                > self.settings.tokens_per_minute
            ):
                raise ProviderRateLimited(60 - (now - self._events[0].timestamp))
            reservation = Reservation(now, tokens)
            self._events.append(reservation)
            return reservation

    def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        with self._lock:
            reservation.tokens = actual_tokens

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            self._cooldown = max(self._cooldown, self.clock() + seconds)


class HTTPChatClient:
    def __init__(
        self,
        provider: Provider,
        settings: GenerationSettings,
        *,
        budget: RequestBudget | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.model = settings.model_for(provider)
        self.budget = budget or RequestBudget(settings)
        self.transport = transport
        self.sleep = sleep

    def complete(self, system: str, user: str) -> Completion:
        if self.provider not in self.settings.enabled_providers or not self.settings.configured(
            self.provider
        ):
            raise ProviderNotConfigured(
                "Selected generation provider is disabled or not configured."
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": "Bearer " + self.settings.groq_api_key.get_secret_value()}
            if self.model.startswith("openai/gpt-oss"):
                payload["reasoning_effort"] = "low"
        elif self.provider == "gemini":
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + quote(self.model, safe="")
                + ":generateContent"
            )
            headers = {"x-goog-api-key": self.settings.gemini_api_key.get_secret_value()}
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.settings.temperature,
                    "maxOutputTokens": self.settings.max_output_tokens,
                    "responseMimeType": "application/json",
                },
            }
        else:
            account = quote(self.settings.cloudflare_account_id.get_secret_value(), safe="")
            url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + quote(
                self.model, safe="@/"
            )
            headers = {
                "Authorization": "Bearer " + self.settings.cloudflare_api_token.get_secret_value()
            }
            payload.pop("model")
        tokens = (len(system.encode()) + len(user.encode())) // 3 + self.settings.max_output_tokens
        with httpx.Client(
            timeout=self.settings.timeout_seconds, transport=self.transport
        ) as client:
            for attempt in range(2):
                reservation = self.budget.reserve(tokens)
                try:
                    response = client.post(url, headers=headers, json=payload)
                except httpx.TransportError as exc:
                    # A timeout can occur after billing/completion; never blindly resend.
                    raise ProviderError(
                        "Generation request failed; remote completion is uncertain."
                    ) from exc
                if response.status_code == 429:
                    delay = max(60, duration(response.headers.get("retry-after", "60")))
                    if response.headers.get("x-ratelimit-remaining-requests") == "0":
                        delay = max(
                            delay,
                            duration(response.headers.get("x-ratelimit-reset-requests", "86400")),
                        )
                    self.budget.cooldown(delay)
                    raise ProviderRateLimited(delay)
                if response.status_code >= 500 and attempt == 0:
                    delay = max(0.5, duration(response.headers.get("retry-after", "0")))
                    if delay <= 1:
                        self.sleep(delay)
                        continue
                if response.status_code in (401, 403, 404):
                    raise ProviderNotConfigured(
                        f"Generation provider configuration rejected (HTTP {response.status_code})."
                    )
                if response.status_code != 200:
                    raise ProviderError(
                        f"Generation provider failed (HTTP {response.status_code})."
                    )
                if response.headers.get("x-ratelimit-remaining-requests") == "0":
                    self.budget.cooldown(
                        duration(response.headers.get("x-ratelimit-reset-requests", "86400"))
                    )
                try:
                    body = response.json()
                    if self.provider == "gemini":
                        candidate = body["candidates"][0]
                        if candidate.get("finishReason", "STOP") != "STOP":
                            raise ValueError("unfinished generation")
                        text = "".join(
                            part.get("text", "")
                            for part in candidate["content"]["parts"]
                            if not part.get("thought")
                        )
                        raw_usage = body.get("usageMetadata", {})
                        usage = {
                            key: raw_usage[value]
                            for key, value in {
                                "prompt_tokens": "promptTokenCount",
                                "completion_tokens": "candidatesTokenCount",
                                "total_tokens": "totalTokenCount",
                            }.items()
                            if value in raw_usage
                        }
                        reported_model = body.get("modelVersion")
                    elif self.provider == "cloudflare":
                        if body.get("success") is not True:
                            raise ValueError("unsuccessful generation")
                        text = body["result"]["response"]
                        # Workers AI JSON mode can return an already parsed object.
                        if isinstance(text, dict):
                            text = json.dumps(text)
                        usage = body["result"].get("usage", {})
                        reported_model = body["result"].get("model")
                    else:
                        choice = body["choices"][0]
                        if choice.get("finish_reason", "stop") != "stop":
                            raise ValueError("unfinished generation")
                        text = choice["message"]["content"]
                        usage = body.get("usage", {})
                        reported_model = body.get("model")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("empty generation")
                    counts = {
                        key: value
                        for key, value in usage.items()
                        if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        and type(value) is int
                        and value >= 0
                    }
                    if "total_tokens" in counts:
                        self.budget.reconcile(reservation, counts["total_tokens"])
                    return Completion(text, counts, reported_model)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ProviderError(
                        "Generation provider returned an incomplete or invalid response."
                    ) from exc
        raise ProviderError("Generation failed.")
