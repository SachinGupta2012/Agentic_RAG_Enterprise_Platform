"""Recovery must fail closed on drift, concurrent writers and uncertain requests."""

import json
from pathlib import Path

import httpx
import pytest

from app.evaluation.judge import JudgeQuotaError, JudgeTransientError, _status_error, reset_seconds
from app.evaluation.recovery import recovery_guard, validate_resume


def test_resume_rejects_changed_input_and_pending_request(tmp_path: Path) -> None:
    output = tmp_path / "qrels.jsonl"
    source = tmp_path / "inputs.jsonl"
    source.write_text("initial")
    validate_resume(output, [source], "groq", "model", "v1", {})
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        validate_resume(output, [source], "groq", "model", "v1", {})
    source.write_text("initial")
    output.with_suffix(".jsonl.pending.json").write_text(
        json.dumps({"query_id": "q", "doc_id": "d"})
    )
    with pytest.raises(ValueError, match="Uncertain"):
        validate_resume(output, [source], "groq", "model", "v1", {})


def test_lock_rejects_concurrent_writer_and_releases(tmp_path: Path) -> None:
    output = tmp_path / "qrels.jsonl"
    with recovery_guard(output), pytest.raises(FileExistsError), recovery_guard(output):
        pass
    with recovery_guard(output):
        pass


def test_retry_metadata_is_safe_and_long_quota_stops() -> None:
    error = _status_error(
        httpx.Response(
            429,
            headers={"retry-after": "7"},
            json={"error": {"message": "private ticket secret", "code": "rate_limit_exceeded"}},
        ),
        "groq",
    )
    assert isinstance(error, JudgeTransientError)
    assert error.retry_after == 7
    assert "private" not in str(error)
    assert "429" in str(error)
    assert isinstance(
        _status_error(httpx.Response(429, headers={"retry-after": "3600"}), "groq"), JudgeQuotaError
    )
    assert reset_seconds("2m59.56s") == pytest.approx(179.56)


def test_daily_token_limit_cooldown_is_extracted_without_message() -> None:
    error = _status_error(
        httpx.Response(
            429,
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "private data; Please try again in 2h3m4s.",
                }
            },
        ),
        "groq",
    )
    assert isinstance(error, JudgeQuotaError)
    assert "7384.0s" in str(error)
    assert "private" not in str(error)


def test_resume_rejects_provider_and_model_drift(tmp_path: Path) -> None:
    from app.evaluation.qrels import QrelsRecord, write_qrels

    output = tmp_path / "qrels.jsonl"
    write_qrels(
        output,
        [
            QrelsRecord(
                query_id="q",
                doc_id="d",
                relevance=1,
                judge_provider="groq",
                judge_model="original",
                prompt_version="v1",
                judgment_source="llm",
                judgment_timestamp="2026-11-23T00:00:00+00:00",
            )
        ],
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_resume(output, [], "gemini", "original", "v1", {})
    with pytest.raises(ValueError, match="provenance"):
        validate_resume(output, [], "groq", "different", "v1", {})
    pending = output.with_suffix(".jsonl.pending.json")
    pending.write_text(json.dumps({"query_id": "q", "doc_id": "d"}))
    validate_resume(output, [], "groq", "original", "v1", {})
    assert not pending.exists()
