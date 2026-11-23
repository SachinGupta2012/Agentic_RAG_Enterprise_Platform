"""The continuation worker must never automatically retry uncertain requests."""

from datetime import UTC, datetime
from pathlib import Path

from scripts import complete_phase2 as worker


def test_worker_stops_on_non_quota_interruption(tmp_path: Path, monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setattr(worker, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(worker, "QRELS", tmp_path / "qrels.jsonl")
    monkeypatch.setattr(worker, "STATE", tmp_path / "status.json")
    monkeypatch.setattr(worker, "STOP", tmp_path / "stop")
    monkeypatch.setattr(worker, "remaining", lambda: 1)
    calls: list[list[str]] = []
    monkeypatch.setattr(worker, "command", lambda args, log: calls.append(args) or 2)
    assert worker.main() == 1
    assert len(calls) == 1
    assert "needs_attention" in worker.STATE.read_text()


def test_worker_waits_after_quota_instead_of_looping_requests(
    tmp_path: Path, monkeypatch: object
) -> None:
    from pytest import MonkeyPatch

    from app.evaluation.jsonl import write_jsonl

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setattr(worker, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(worker, "QRELS", tmp_path / "qrels.jsonl")
    monkeypatch.setattr(worker, "STATE", tmp_path / "status.json")
    monkeypatch.setattr(worker, "STOP", tmp_path / "stop")
    monkeypatch.setattr(worker, "remaining", lambda: 1)
    calls: list[list[str]] = []

    def quota(args: list[str], log: Path) -> int:
        calls.append(args)
        write_jsonl(
            tmp_path / "judgment_failures.jsonl",
            [
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error_type": "JudgeQuotaError",
                    "error": "daily request quota exhausted",
                }
            ],
        )
        return 3

    waits: list[float] = []
    monkeypatch.setattr(worker, "command", quota)
    monkeypatch.setattr(worker, "pause", lambda delay: waits.append(delay) or False)
    assert worker.main() == 1
    assert len(calls) == 1
    assert waits[0] > 86300
