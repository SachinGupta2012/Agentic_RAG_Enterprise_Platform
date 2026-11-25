"""Synthetic-only tests for the explicitly authorized legacy repair."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.preprocessing.clean_legacy import clean_legacy
from app.preprocessing.isolation import load_isolation, verify_frame


@pytest.fixture
def legacy(tmp_path: Path) -> Path:
    source = tmp_path / "original"
    source.mkdir()
    files = {}
    for split, questions in {
        "retrieval": ["Password RESET", "Printer unavailable"],
        "dev": ["  password   reset  ", "Invoice charged twice"],
        "eval": [
            "\uff30\uff41\uff53\uff53\uff57\uff4f\uff52\uff44 RESET",
            "Backup archive corrupted",
        ],
    }.items():
        frame = pd.DataFrame(
            [
                {
                    "ticket_id": f"{split}-{i}",
                    "source_ticket_id": f"source-{split}-{i}",
                    "customer_query": query,
                    "metadata": json.dumps(
                        {"split": split, "near_dup_cluster_id": f"cluster-{split}-{i}"}
                    ),
                }
                for i, query in enumerate(questions)
            ]
        )
        data = frame.to_parquet(index=False)
        name = f"tickets_{split}.parquet"
        (source / name).write_bytes(data)
        files[split] = {
            "path": name,
            "rows": len(frame),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (source / "split_manifest.json").write_text(
        json.dumps({"source": {"dataset": "synthetic"}, "files": files}), encoding="utf-8"
    )
    return source


def test_cleanup_preserves_originals_and_binds_clean_evidence(legacy: Path) -> None:
    before = {path.name: path.read_bytes() for path in legacy.iterdir()}
    queries = legacy.parent / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps({"query_id": query_id, "source_ticket_id": ticket, "text": text})
            for query_id, ticket, text in [
                ("q1", "dev-0", "password reset"),
                ("q2", "dev-1", "Invoice charged twice"),
            ]
        ),
        encoding="utf-8",
    )
    pilot = legacy.parent / "pilot.json"
    pilot.write_text(json.dumps({"selected_query_ids": ["q2"]}), encoding="utf-8")
    destination = legacy.parent / "clean"
    report = clean_legacy(
        legacy, destination, authorize_heldout_read=True, queries_path=queries, pilot_path=pilot
    )
    assert report["remaining_rows"] == {"retrieval": 2, "dev": 1, "eval": 1}
    assert report["benchmark"]["affected_query_ids"] == ["q1"]
    assert report["benchmark"]["affected_pilot_query_ids"] == []
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == before
    assert (destination / "tickets_retrieval.parquet").read_bytes() == before[
        "tickets_retrieval.parquet"
    ]
    manifest_path = destination / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence = load_isolation(manifest, manifest_path)
    # Routine verification uses only retrieval/dev and the sidecar, never eval.
    for split in ("retrieval", "dev"):
        verify_frame(pd.read_parquet(destination / f"tickets_{split}.parquet"), split, evidence)
    with pytest.raises(ValueError, match="must be new"):
        clean_legacy(legacy, destination, authorize_heldout_read=True)


def test_authorization_required_before_any_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="authorization"):
        clean_legacy(tmp_path / "missing", tmp_path / "clean")
    assert not (tmp_path / "clean").exists()


def test_bad_source_checksum_fails_before_export(legacy: Path) -> None:
    with (legacy / "tickets_eval.parquet").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        clean_legacy(legacy, legacy.parent / "clean", authorize_heldout_read=True)
    assert not (legacy.parent / "clean").exists()


def test_other_isolation_violations_are_not_silently_repaired(legacy: Path) -> None:
    path = legacy / "tickets_dev.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "ticket_id"] = "retrieval-1"
    data = frame.to_parquet(index=False)
    path.write_bytes(data)
    manifest_path = legacy / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["dev"]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Cross-split leakage"):
        clean_legacy(legacy, legacy.parent / "clean", authorize_heldout_read=True)
    assert not (legacy.parent / "clean").exists()
