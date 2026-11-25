"""Cross-split evidence created before evaluation files are quarantined."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPLITS = ("retrieval", "dev", "eval")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_digest(value: str) -> str:
    return digest(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold())


def reject_evaluation_path(path: Path) -> None:
    if path.name.lower().startswith("tickets_eval") or path.resolve().name.lower().startswith(
        "tickets_eval"
    ):
        raise ValueError("Evaluation data is quarantined; use the split-isolation sidecar")


def split_evidence(frame: Any) -> dict[str, Any]:
    metadata = [json.loads(value) for value in frame["metadata"]]
    if not all("near_dup_cluster_id" in row for row in metadata):
        raise ValueError("Missing near-duplicate cluster provenance")
    return {
        "rows": len(frame),
        "ticket_hashes": sorted(digest(str(value)) for value in frame["ticket_id"]),
        "issue_hashes": sorted({issue_digest(str(value)) for value in frame["customer_query"]}),
        "cluster_hashes": sorted({digest(str(row["near_dup_cluster_id"])) for row in metadata}),
    }


def check_disjoint(evidence: Mapping[str, Any]) -> None:
    if set(evidence) != set(SPLITS):
        raise ValueError("Isolation evidence must cover retrieval, dev and eval")
    for split in SPLITS:
        entry = evidence[split]
        if not isinstance(entry, dict):
            raise ValueError("Invalid isolation split entry")
        if not isinstance(entry.get("rows"), int) or entry["rows"] < 1:
            raise ValueError("Invalid isolation row count")
        for field in ("ticket_hashes", "issue_hashes", "cluster_hashes"):
            values = entry[field]
            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in values
                )
            ):
                raise ValueError("Invalid isolation fingerprints")
            if len(values) != len(set(values)):
                raise ValueError("Duplicate isolation fingerprints")
        if len(entry["ticket_hashes"]) != entry["rows"]:
            raise ValueError("Ticket fingerprint count differs from split row count")
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1 :]:
            for field in ("ticket_hashes", "issue_hashes", "cluster_hashes"):
                if set(evidence[split][field]) & set(evidence[other][field]):
                    raise ValueError(f"Cross-split leakage: {split}/{other}/{field}")


def load_isolation(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    reference = manifest.get("isolation")
    if not isinstance(reference, dict):
        raise ValueError(
            "Missing trusted split-isolation sidecar; legacy data cannot be certified without it"
        )
    path = (manifest_path.parent / reference["path"]).resolve()
    if path.parent != manifest_path.parent.resolve() or path.suffix != ".json":
        raise ValueError("Isolation sidecar must be a JSON file beside the manifest")
    reject_evaluation_path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.get("sha256"):
        raise ValueError("Isolation sidecar checksum mismatch")
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("schema_version") != "split-isolation-v1":
        raise ValueError("Unsupported isolation schema")
    check_disjoint(result["splits"])
    for split in SPLITS:
        if result["splits"][split]["rows"] != manifest["files"][split]["rows"]:
            raise ValueError("Manifest and isolation row counts differ")
    return result


def verify_frame(frame: Any, split: str, isolation: Mapping[str, Any]) -> None:
    if split not in {"retrieval", "dev"}:
        raise ValueError("Evaluation frames may not be loaded for validation")
    if any(json.loads(value).get("split") != split for value in frame["metadata"]):
        raise ValueError("Row metadata has the wrong split")
    if split_evidence(frame) != isolation["splits"][split]:
        raise ValueError("Frame differs from recorded isolation evidence")
