"""Tests for PII redaction."""

from __future__ import annotations

from app.retrieval.pii import contains_pii, redact_pii


def test_email_at_end_of_sentence_is_redacted_without_losing_punctuation() -> None:
    redacted, counts = redact_pii("Contact person@example.com.")
    assert redacted == "Contact [EMAIL_REDACTED]."
    assert counts.email == 1


def test_redacts_email_url_and_phone() -> None:
    text = (
        "Email a.customer@example.com or call +1 (555) 123-4567. "
        "See https://example.com/help?ticket=1."
    )

    redacted, counts = redact_pii(text)

    assert "a.customer@example.com" not in redacted
    assert "555" not in redacted
    assert "https://example.com" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted
    assert "[URL_REDACTED]" in redacted
    assert counts.email == 1
    assert counts.phone == 1
    assert counts.url == 1
    assert counts.total == 3


def test_redaction_is_idempotent() -> None:
    text = "Contact a@b.com or +1 555-123-4567."
    first, first_counts = redact_pii(text)
    second, second_counts = redact_pii(first)

    assert first == second
    assert first_counts.total == 2
    assert second_counts.total == 0


def test_dates_and_long_identifiers_are_not_redacted() -> None:
    text = "Created 2024-11-22 for order 12345678901234567890."

    redacted, counts = redact_pii(text)

    assert redacted == text
    assert counts.total == 0


def test_empty_text() -> None:
    redacted, counts = redact_pii("")

    assert redacted == ""
    assert counts.total == 0


def test_contains_pii() -> None:
    assert contains_pii("write to a@b.com")
    assert contains_pii("call (555) 123-4567")
    assert contains_pii("visit https://example.com")
    assert not contains_pii("no sensitive values here")
