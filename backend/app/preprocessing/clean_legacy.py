"""Explicit, local-only repair of legacy held-out question overlap.

This exceptional operation requires authorization to read held-out data. Routine
validation and ingestion retain their evaluation-file quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from app.preprocessing.isolation import SPLITS, check_disjoint, issue_digest, split_evidence


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def clean_legacy(
    source: Path,
    destination: Path,
    *,
    authorize_heldout_read: bool = False,
    queries_path: Path | None = None,
    pilot_path: Path | None = None,
) -> dict[str, Any]:
    """Preserve retrieval bytes; exclude held-out questions occurring in retrieval.

    Abort on any other isolation violation. Never overwrite an existing directory.
    The source manifest/checksums are verified before any Parquet is decoded.
    """
    if not authorize_heldout_read:
        raise ValueError("Explicit authorization to read held-out data is required")
    if destination.exists():
        raise ValueError("Output directory must be new")
    manifest_bytes = (source / "split_manifest.json").read_bytes()
    parent = json.loads(manifest_bytes)
    original: dict[str, bytes] = {}
    for split in SPLITS:
        entry = parent["files"][split]
        path = (source / entry["path"]).resolve()
        if path.parent != source.resolve() or path.suffix != ".parquet":
            raise ValueError("Source splits must be Parquet files beside their manifest")
        original[split] = path.read_bytes()
        if hashlib.sha256(original[split]).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {split}")
    frames = {split: pd.read_parquet(BytesIO(data)) for split, data in original.items()}
    for split, frame in frames.items():
        if len(frame) != parent["files"][split]["rows"] or any(
            json.loads(value).get("split") != split for value in frame["metadata"]
        ):
            raise ValueError(f"Source row count or split metadata mismatch: {split}")
    retrieval_hashes = set(frames["retrieval"]["customer_query"].map(issue_digest))
    removed: dict[str, Any] = {}
    removed_dev_ids: set[str] = set()
    for split in ("dev", "eval"):
        frame = frames[split]
        hashes = frame["customer_query"].map(issue_digest)
        mask = hashes.isin(retrieval_hashes)
        removed[split] = {
            "rows": int(mask.sum()),
            "issue_hashes": sorted(set(hashes[mask])),
        }
        if split == "dev":
            removed_dev_ids = set(frame.loc[mask, "ticket_id"].astype(str))
        frames[split] = frame.loc[~mask].reset_index(drop=True)
    evidence = {split: split_evidence(frame) for split, frame in frames.items()}
    check_disjoint(evidence)
    # Source identifiers are separate from internal ticket identifiers.
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1 :]:
            if set(frames[split]["source_ticket_id"].astype(str)) & set(
                frames[other]["source_ticket_id"].astype(str)
            ):
                raise ValueError("Cross-split source ticket ID overlap")
    report: dict[str, Any] = {
        "policy": "Exclude dev/eval questions matching retrieval after NFKC, whitespace and case normalization",
        "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "original_files": parent["files"],
        "removed": removed,
        "remaining_rows": {split: len(frame) for split, frame in frames.items()},
        "retrieval_byte_identical": True,
        "scope": "Recorded ID, normalized-question and existing group isolation; not semantic deduplication",
    }
    if queries_path is not None:
        query_bytes = queries_path.read_bytes()
        queries = [json.loads(line) for line in query_bytes.splitlines() if line.strip()]
        affected = sorted(
            row["query_id"]
            for row in queries
            if str(row["source_ticket_id"]) in removed_dev_ids
            or issue_digest(row["text"]) in retrieval_hashes
        )
        report["benchmark"] = {
            "query_file_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "queries": len(queries),
            "affected_query_ids": affected,
        }
        if pilot_path is not None:
            pilot_bytes = pilot_path.read_bytes()
            pilot = json.loads(pilot_bytes)
            report["benchmark"]["pilot_report_sha256"] = hashlib.sha256(pilot_bytes).hexdigest()
            report["benchmark"]["affected_pilot_query_ids"] = sorted(
                set(affected) & set(pilot["selected_query_ids"])
            )
    elif pilot_path is not None:
        raise ValueError("A pilot impact check requires the benchmark query file")
    buffers = {
        split: original[split] if split == "retrieval" else frame.to_parquet(index=False)
        for split, frame in frames.items()
    }
    sidecar = _json_bytes({"schema_version": "split-isolation-v1", "splits": evidence})
    manifest = {
        "schema_version": "ticket-splits-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": parent["source"],
        "repair": report,
        "files": {
            split: {
                "path": f"tickets_{split}.parquet",
                "rows": len(frames[split]),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for split, data in buffers.items()
        },
        "isolation": {
            "path": "split_isolation.json",
            "sha256": hashlib.sha256(sidecar).hexdigest(),
        },
    }
    artifacts = {f"tickets_{split}.parquet": data for split, data in buffers.items()}
    artifacts.update({"split_isolation.json": sidecar, "cleanup_report.json": _json_bytes(report)})
    # Write the manifest last: an interrupted export has no completion marker.
    artifacts["split_manifest.json"] = _json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--authorize-heldout-read", action="store_true", required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--pilot-report", type=Path)
    args = parser.parse_args()
    report = clean_legacy(
        args.source,
        args.destination,
        authorize_heldout_read=args.authorize_heldout_read,
        queries_path=args.queries,
        pilot_path=args.pilot_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
