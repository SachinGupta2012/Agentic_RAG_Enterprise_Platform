"""PII detection and redaction for ticket text.

Ticket content is untrusted input. Redaction is applied before text is sent to
Qdrant Cloud Inference, before it is stored in the retrieval payload, and again
before snippets are returned by the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-]|\.[\w-])",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
# Requires a separator so dates and long numeric identifiers are not redacted.
PHONE_PATTERN = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)[\s.-]?|\d{3,4}[\s.-])"
    r"\d{3,4}[\s.-]?\d{3,4}(?![\w])"
)

EMAIL_TOKEN = "[EMAIL_REDACTED]"
PHONE_TOKEN = "[PHONE_REDACTED]"
URL_TOKEN = "[URL_REDACTED]"
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}'\""


@dataclass(frozen=True)
class RedactionCounts:
    """Number of values replaced for each PII category."""

    email: int = 0
    phone: int = 0
    url: int = 0

    @property
    def total(self) -> int:
        """Return the total number of redactions."""

        return self.email + self.phone + self.url

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-friendly representation."""

        return {"email": self.email, "phone": self.phone, "url": self.url}

    def merge(self, other: RedactionCounts) -> RedactionCounts:
        """Combine counts from multiple text fields."""

        return RedactionCounts(
            email=self.email + other.email,
            phone=self.phone + other.phone,
            url=self.url + other.url,
        )


def _replace_url(match: re.Match[str]) -> str:
    value = match.group(0)
    trailing = ""
    while value and value[-1] in _TRAILING_URL_PUNCTUATION:
        trailing = value[-1] + trailing
        value = value[:-1]
    return f"{URL_TOKEN}{trailing}"


def redact_pii(text: str) -> tuple[str, RedactionCounts]:
    """Redact email addresses, URLs, and phone-like values from text.

    The function is idempotent: running it on already-redacted text leaves the
    placeholders untouched and returns zero counts.
    """

    if not text:
        return "", RedactionCounts()

    text, url_count = URL_PATTERN.subn(_replace_url, text)
    text, email_count = EMAIL_PATTERN.subn(EMAIL_TOKEN, text)
    text, phone_count = PHONE_PATTERN.subn(PHONE_TOKEN, text)

    return text, RedactionCounts(email=email_count, phone=phone_count, url=url_count)


def contains_pii(text: str) -> bool:
    """Return whether any supported PII pattern is present."""

    if not text:
        return False
    return any(
        pattern.search(text) is not None for pattern in (EMAIL_PATTERN, URL_PATTERN, PHONE_PATTERN)
    )
