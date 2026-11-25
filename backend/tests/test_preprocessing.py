"""Synthetic-only ingestion tests: token bounds, group isolation and quarantine."""

import builtins
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.ingest_qdrant import main as ingest
from scripts.validate_splits import validate

from app.core.config import PreprocessingSettings
from app.preprocessing.cli import main, read_raw
from app.preprocessing.isolation import check_disjoint, load_isolation, verify_frame
from app.preprocessing.pipeline import cluster_issues, export, normalize, preprocess
from app.preprocessing.tokens import token_count, token_prefix
from app.retrieval.ingestion import (
    IngestionValidationError,
    load_parquet,
    record_to_document,
    validate_checksum,
    validate_split_isolation,
)

QUERIES = [
    "Account password reset instructions unavailable",
    "Invoice payment charged twice yesterday",
    "Printer paper jam inside device",
    "Parcel tracking shipment address incorrect",
    "Wireless network router disconnected repeatedly",
    "Subscription cancellation confirmation email missing",
    "Browser upload attachment size exceeded",
    "Backup restore archive corrupted unexpectedly",
    "Monitor display flickering after update",
    "Database connection certificate expired",
    "Headphones microphone not detected",
    "Calendar invitation timezone mismatch",
]
SOURCE = {"method": "local_csv", "path": "synthetic.csv", "sha256": "0" * 64}


def config(**changes: object) -> PreprocessingSettings:
    return PreprocessingSettings(
        _env_file=None,
        retrieval_rows=4,
        dev_rows=2,
        eval_rows=2,
        source_dataset="synthetic/tickets",
        **changes,
    )


def raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": f"Support {i}",
                "description": query,
                "resolution": f"Follow the documented resolution steps for case {i}.",
                "lang": "English",
                "urgency": "HIGH",
                "ticket_type": "incident",
            }
            for i, query in enumerate(QUERIES)
        ]
    )


@pytest.fixture(scope="module")
def prepared():
    return preprocess(raw(), config(), SOURCE)


def forbid_eval_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Trap opening, hashing or parsing an evaluation file, including synthetic ones."""
    calls: list[str] = []
    real_builtin = builtins.open
    real_path_open = Path.open
    real_pandas = pd.read_parquet

    def check(path, mode="r"):
        if (
            isinstance(path, (str, Path))
            and Path(path).name.startswith("tickets_eval")
            and "r" in mode
        ):
            raise AssertionError("Reserved evaluation file was opened")

    def guarded_open(path, mode="r", *args, **kwargs):
        check(path, mode)
        return real_builtin(path, mode, *args, **kwargs)

    def guarded_path(path, mode="r", *args, **kwargs):
        check(path, mode)
        return real_path_open(path, mode, *args, **kwargs)

    def guarded_pandas(path, *args, **kwargs):
        check(path)
        calls.append(Path(path).name)
        return real_pandas(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path)
    monkeypatch.setattr(pd, "read_parquet", guarded_pandas)
    return calls


def test_deterministic_groups_and_no_id_issue_or_cluster_leakage() -> None:
    frame = raw()
    duplicate = frame.iloc[[0]].copy()
    duplicate["resolution"] = "A different historical resolution for the same underlying query."
    frame = pd.concat([frame, duplicate], ignore_index=True)
    first = preprocess(frame, config(), SOURCE)
    second = preprocess(frame, config(), SOURCE)
    assert first.isolation == second.isolation
    check_disjoint(first.isolation["splits"])
    owners = [
        split for split, part in first.splits.items() if QUERIES[0] in set(part["customer_query"])
    ]
    assert len(owners) == 1
    assert sum((part["customer_query"] == QUERIES[0]).sum() for part in first.splits.values()) == 2
    assert all(
        json.loads(meta)["split"] == split
        for split, part in first.splits.items()
        for meta in part["metadata"]
    )


def test_preprocessing_does_not_fit_a_shared_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.feature_extraction.text import HashingVectorizer

    monkeypatch.setattr(
        HashingVectorizer,
        "fit",
        lambda *args, **kwargs: pytest.fail("Grouping must not fit shared vocabulary/IDF"),
    )
    result = preprocess(raw(), config(), SOURCE)
    assert result.summary["exported_rows"] == 8


def test_neighbor_grouping_covers_more_than_ten_matching_records() -> None:
    frame = raw().iloc[[0] * 15].reset_index(drop=True)
    frame["description"] = [QUERIES[0] + "!" * i for i in range(15)]
    normalized = normalize(frame, config())
    assert len(set(cluster_issues(normalized, config()))) == 1


@pytest.mark.parametrize("field", ["ticket_hashes", "issue_hashes", "cluster_hashes"])
def test_registry_rejects_cross_split_leakage(prepared, field: str) -> None:
    evidence = json.loads(json.dumps(prepared.isolation["splits"]))
    evidence["dev"][field][0] = evidence["retrieval"][field][0]
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint(evidence)


def test_validate_and_dry_run_ingestion_never_open_eval(
    tmp_path: Path, prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = export(prepared, tmp_path / "fresh", SOURCE)
    reads = forbid_eval_reads(monkeypatch)
    monkeypatch.setattr(
        "scripts.ingest_qdrant.QdrantTicketStore",
        lambda **kwargs: pytest.fail("Dry run must not create a Qdrant client"),
    )
    assert validate(output) == 0
    assert (
        ingest(
            [
                "--input",
                str(output / "tickets_retrieval.parquet"),
                "--manifest",
                str(output / "split_manifest.json"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert set(reads) == {"tickets_retrieval.parquet", "tickets_dev.parquet"}


def test_eval_paths_are_blocked_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_eval_reads(monkeypatch)
    path = tmp_path / "tickets_eval.parquet"
    with pytest.raises(ValueError, match="quarantined"):
        load_parquet(path)
    with pytest.raises(ValueError, match="quarantined"):
        validate_checksum(path, {}, "eval")
    with pytest.raises(ValueError, match="quarantined"):
        read_raw(path)


def test_legacy_manifest_fails_closed_without_reading_any_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "split_manifest.json").write_text('{"files":{}}')
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: pytest.fail("Legacy manifest cannot certify isolation"),
    )
    assert validate(tmp_path) == 1


def test_missing_or_tampered_sidecar_and_wrong_split_are_rejected(tmp_path: Path, prepared) -> None:
    destination = export(prepared, tmp_path / "fresh", SOURCE)
    manifest_path = destination / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    isolation = load_isolation(manifest, manifest_path)
    with pytest.raises(ValueError, match="wrong split"):
        verify_frame(prepared.splits["dev"], "retrieval", isolation)
    with pytest.raises(IngestionValidationError):
        validate_split_isolation(prepared.splits["dev"], manifest, manifest_path)
    (destination / "split_isolation.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_isolation(manifest, manifest_path)


@pytest.mark.parametrize(
    "text",
    [
        "complicated punctuation!? " * 1000,
        "修复账户登录 " * 1000,
        "unbelievablylongwordpiece " * 1000,
    ],
)
def test_token_limits_include_special_tokens_and_preserve_substrings(text: str) -> None:
    result = token_prefix(text, 32)
    assert result.truncated and result.tokens == token_count(result.text) <= 32
    assert text.startswith(result.text)


def test_long_ticket_keeps_resolution_in_sparse_text_and_full_payload(prepared) -> None:
    row = prepared.splits["retrieval"].iloc[0].to_dict()
    row["customer_query"] = "A long support question with detailed diagnostics. " * 500
    row["agent_response"] = "Restore the saved configuration and restart the service. " * 500
    doc = record_to_document(row, config())
    assert token_count(doc.dense_text) <= 256
    assert token_count(doc.sparse_text) <= 2048
    assert "Resolution:" in doc.sparse_text and "Restore the saved configuration" in doc.sparse_text
    assert (
        doc.customer_query == row["customer_query"].strip()
        and doc.agent_response == row["agent_response"].strip()
    )
    assert doc.text_token_counts["dense_truncated"] == 1
    assert doc.text_token_counts["sparse_truncated"] == 1


def test_aliases_cleaning_pii_and_missing_required_columns() -> None:
    frame = raw()
    frame.loc[0, "description"] = "&lt;test&gt;　Contact person@example.com. " + QUERIES[0]
    result = normalize(frame, config())
    assert "person@example.com" not in result.iloc[0]["customer_query"]
    assert result.iloc[0]["category"] == "Incident" and result.iloc[0]["priority"] == "high"
    with pytest.raises(ValueError, match="requires"):
        normalize(frame.drop(columns=["resolution"]), config())


def test_dry_run_creates_nothing_and_export_never_overwrites(tmp_path: Path, prepared) -> None:
    source = tmp_path / "raw.csv"
    raw().to_csv(source, index=False)
    output = tmp_path / "fresh"
    args = [
        "--input",
        str(source),
        "--retrieval-rows",
        "4",
        "--dev-rows",
        "2",
        "--eval-rows",
        "2",
        "--output-dir",
        str(output),
    ]
    assert main([*args, "--dry-run"]) == 0 and not output.exists()
    assert main(args) == 0
    before = hashlib.sha256((output / "split_manifest.json").read_bytes()).hexdigest()
    assert main(args) == 1
    assert hashlib.sha256((output / "split_manifest.json").read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="never overwritten"):
        export(prepared, output, SOURCE)


def test_impossible_cluster_sizes_fail_instead_of_leaking() -> None:
    frame = raw().iloc[[0] * 12].reset_index(drop=True)
    frame["resolution"] = [f"Distinct resolution number {i}" for i in range(12)]
    with pytest.raises(ValueError, match="Whole duplicate groups"):
        preprocess(frame, config(), SOURCE)


def test_writing_existing_qdrant_collection_is_blocked_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.ingest_qdrant.load_manifest",
        lambda *args: pytest.fail("Existing collection must be protected before data loading"),
    )
    assert ingest([]) == 1


def test_preprocessing_settings_use_core_config_and_preserve_env_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import PreprocessingSettings as CentralSettings
    from app.preprocessing.config import PreprocessingSettings as CompatibilitySettings

    assert CentralSettings is CompatibilitySettings
    assert CentralSettings.__module__ == "app.core.config"
    monkeypatch.setenv("PREPROCESS_DENSE_MAX_TOKENS", "128")
    settings = CentralSettings(_env_file=None)
    assert settings.dense_max_tokens == 128
    assert settings.retrieval_rows == 16000


def test_config_rejects_incompatible_tokenizer_and_invalid_limits() -> None:
    with pytest.raises(ValueError):
        config(dense_model="another-embedding-model")
    with pytest.raises(ValueError):
        config(min_query_chars=100, max_query_chars=10)


def test_export_rejects_mutated_split_provenance(tmp_path: Path, prepared) -> None:
    from copy import deepcopy

    changed = deepcopy(prepared)
    meta = json.loads(changed.splits["retrieval"].iloc[0]["metadata"])
    meta["split"] = "eval"
    changed.splits["retrieval"].loc[0, "metadata"] = json.dumps(meta)
    with pytest.raises(ValueError, match="provenance changed"):
        export(changed, tmp_path / "fresh", SOURCE)
    assert not (tmp_path / "fresh").exists()


def test_public_hf_source_requires_revision_and_does_not_send_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    from app.preprocessing.cli import read_huggingface

    seen = []

    def loader(dataset, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(to_pandas=raw)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    with pytest.raises(ValueError, match="exact"):
        read_huggingface("owner/public", "main")
    frame, source = read_huggingface("owner/public", "1" * 40)
    assert len(frame) == len(QUERIES) and source["method"] == "huggingface"
    assert source["url"] == "https://huggingface.co/datasets/owner/public/tree/" + "1" * 40
    assert seen[0]["token"] is False and seen[0]["revision"] == "1" * 40
