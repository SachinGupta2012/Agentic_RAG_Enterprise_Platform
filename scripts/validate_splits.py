"""Validate retrieval/dev and recorded isolation without opening evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.preprocessing.isolation import load_isolation, reject_evaluation_path, verify_frame
from app.preprocessing.pipeline import FINAL_COLUMNS
from app.retrieval.ingestion import (
    load_manifest,
    load_parquet,
    validate_checksum,
    validate_retrieval_frame,
)


def validate(processed_dir: Path) -> int:
    try:
        manifest_path = processed_dir / "split_manifest.json"
        manifest = load_manifest(manifest_path)
        isolation = load_isolation(manifest, manifest_path)
        for split in ("retrieval", "dev"):
            entry = manifest["files"][split]
            path = (processed_dir / entry["path"]).resolve()
            if path.parent != processed_dir.resolve():
                raise ValueError("Split files must be beside the manifest")
            reject_evaluation_path(path)
            validate_checksum(path, manifest, split)
            frame = load_parquet(path)
            if list(frame.columns) != FINAL_COLUMNS or len(frame) != entry["rows"]:
                raise ValueError("Split schema or row count differs from manifest")
            if frame["ticket_id"].duplicated().any():
                raise ValueError("Duplicate ticket IDs")
            if any(not isinstance(json.loads(value), list) for value in frame["tags"]):
                raise ValueError("Tags must be JSON arrays")
            verify_frame(frame, split, isolation)
            if split == "retrieval":
                validate_retrieval_frame(frame, entry["rows"])
            print(f"{split}: {len(frame)} rows validated")
        print(
            "Evaluation file not opened. ID, issue and group isolation verified from the bound sidecar."
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Validation failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return validate(parser.parse_args(argv).processed_dir)


if __name__ == "__main__":
    raise SystemExit(main())
