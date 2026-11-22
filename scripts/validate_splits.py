"""Validate processed ticket splits, manifest checksums, and leakage constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "ticket_id",
    "source_ticket_id",
    "subject",
    "customer_query",
    "agent_response",
    "category",
    "department",
    "priority",
    "language",
    "tags",
    "resolution_status",
    "source_dataset",
    "created_at",
    "metadata",
]
SPLIT_FILES = {
    "retrieval": "tickets_retrieval.parquet",
    "dev": "tickets_dev.parquet",
    "eval": "tickets_eval.parquet",
}
REQUIRED_FIELDS = [
    "ticket_id",
    "source_ticket_id",
    "subject",
    "customer_query",
    "category",
    "priority",
    "language",
]
REQUIRED_METADATA_FIELDS = {
    "source_system",
    "source_dataset",
    "source_dataset_url",
    "source_has_native_id",
    "source_ticket_id_kind",
    "acquisition_method",
    "acquisition_path",
    "source_row_index",
    "source_version",
    "business_type",
    "near_dup_cluster_id",
    "near_dup_cluster_size",
    "split",
    "has_source_timestamp",
}
ALLOWED_ACQUISITION_METHODS = {"kaggle_csv", "local_csv", "huggingface"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for name, filename in SPLIT_FILES.items():
        path = processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = pd.read_parquet(path)
    return splits


def validate(processed_dir: Path) -> int:
    errors: list[str] = []
    splits = load_splits(processed_dir)

    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        errors.append(f"Missing manifest: {manifest_path}")
        manifest: dict[str, Any] = {"files": {}}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for name, frame in splits.items():
        if list(frame.columns) != EXPECTED_COLUMNS:
            errors.append(f"{name}: unexpected columns {list(frame.columns)}")
        if frame["ticket_id"].duplicated().any():
            errors.append(f"{name}: duplicate ticket_id values")
        for field in REQUIRED_FIELDS:
            null_count = int(frame[field].isna().sum())
            if null_count:
                errors.append(f"{name}: {null_count} null values in {field}")

        for value in frame["ticket_id"]:
            try:
                uuid.UUID(str(value))
            except (TypeError, ValueError):
                errors.append(f"{name}: invalid ticket_id UUID: {value!r}")
                break

        unresolved = int((frame["resolution_status"] != "resolved").sum())
        if unresolved:
            errors.append(f"{name}: {unresolved} rows are not resolved")

        datasets = sorted(str(value) for value in frame["source_dataset"].dropna().unique())
        if len(datasets) != 1:
            errors.append(f"{name}: expected one source_dataset, found {datasets}")

        if frame["created_at"].notna().any():
            errors.append(f"{name}: created_at must be null because the source has no timestamps")

        for dataset, locator in zip(
            frame["source_dataset"], frame["source_ticket_id"], strict=True
        ):
            prefix = f"{dataset}:"
            if not str(locator).startswith(prefix) or not str(locator)[len(prefix) :].isdigit():
                errors.append(f"{name}: invalid source_ticket_id {locator!r}")
                break

        file_info = manifest.get("files", {}).get(name, {})
        if file_info:
            expected_rows = file_info.get("rows")
            if expected_rows != len(frame):
                errors.append(f"{name}: manifest rows={expected_rows}, parquet rows={len(frame)}")
            expected_hash = file_info.get("sha256")
            actual_hash = sha256(processed_dir / SPLIT_FILES[name])
            if expected_hash != actual_hash:
                errors.append(f"{name}: SHA-256 mismatch")

        for value in frame["tags"]:
            try:
                parsed_tags = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                errors.append(f"{name}: invalid JSON in tags: {value!r}")
                break
            if not isinstance(parsed_tags, list):
                errors.append(f"{name}: tags is not a JSON list: {value!r}")
                break

        metadata_rows: list[dict[str, Any]] = []
        metadata_error = False
        for value in frame["metadata"]:
            try:
                parsed_metadata = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                errors.append(f"{name}: invalid JSON in metadata: {value!r}")
                metadata_error = True
                break
            if not isinstance(parsed_metadata, dict):
                errors.append(f"{name}: metadata is not a JSON object: {value!r}")
                metadata_error = True
                break
            metadata_rows.append(parsed_metadata)

        if metadata_rows:
            missing_keys = REQUIRED_METADATA_FIELDS - set(metadata_rows[0])
            if missing_keys:
                errors.append(f"{name}: metadata missing keys {sorted(missing_keys)}")
            for index, parsed_metadata in enumerate(metadata_rows):
                if parsed_metadata.get("split") != name:
                    errors.append(
                        f"{name}: metadata split {parsed_metadata.get('split')!r} at row {index}"
                    )
                    break
                if parsed_metadata.get("source_has_native_id") is not False:
                    errors.append(f"{name}: source_has_native_id is not false at row {index}")
                    break
                if parsed_metadata.get("source_ticket_id_kind") != "derived_locator":
                    errors.append(
                        f"{name}: source_ticket_id_kind is not 'derived_locator' at row {index}"
                    )
                    break
                if parsed_metadata.get("acquisition_method") not in ALLOWED_ACQUISITION_METHODS:
                    errors.append(
                        f"{name}: invalid acquisition_method "
                        f"{parsed_metadata.get('acquisition_method')!r} at row {index}"
                    )
                    break
        elif not metadata_error:
            errors.append(f"{name}: metadata is empty")

    split_ids = {name: set(frame["ticket_id"]) for name, frame in splits.items()}
    overlaps = {
        "retrieval_eval": split_ids["retrieval"] & split_ids["eval"],
        "retrieval_dev": split_ids["retrieval"] & split_ids["dev"],
        "dev_eval": split_ids["dev"] & split_ids["eval"],
    }
    for label, overlap in overlaps.items():
        if overlap:
            errors.append(f"{label}: {len(overlap)} overlapping ticket_id values")

    all_ids = pd.concat([frame["ticket_id"] for frame in splits.values()])
    if all_ids.duplicated().any():
        errors.append("ticket_id is not globally unique across splits")

    print("Split summary:")
    for name, frame in splits.items():
        print(f"  {name:10s} {len(frame):>6,} rows  {frame['category'].nunique()} categories")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nValidation passed: schema, checksums, JSON fields, and split isolation.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the processed Parquet splits and manifest.",
    )
    args = parser.parse_args()
    return validate(args.processed_dir)


if __name__ == "__main__":
    raise SystemExit(main())
