"""Pure, deterministic preprocessing; no vector database or LLM access."""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from app.core.config import PreprocessingSettings
from app.preprocessing.isolation import SPLITS, check_disjoint, digest, issue_digest, split_evidence
from app.retrieval.pii import redact_pii

FINAL_COLUMNS = [
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


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    return re.sub(r"\s+", " ", text).strip()


def normalize(raw: Any, config: PreprocessingSettings) -> Any:
    columns = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}
    if len(columns) != len(raw.columns):
        raise ValueError("Ambiguous duplicate source column names")
    resolved = {
        key: next((columns[alias] for alias in aliases if alias in columns), None)
        for key, aliases in config.column_aliases.items()
    }
    if any(
        resolved.get(field) is None for field in ("customer_query", "agent_response", "language")
    ):
        raise ValueError("Raw source requires customer query, agent response and language columns")
    result = pd.DataFrame(index=range(len(raw)))
    raw = raw.reset_index(drop=True)
    result["_source_row_index"] = range(len(raw))
    for field in (
        "subject",
        "customer_query",
        "agent_response",
        "category",
        "department",
        "priority",
        "language",
    ):
        column = resolved.get(field)
        result[field] = raw[column].map(clean_text) if column else ""
    for field in ("subject", "customer_query", "agent_response", "category", "department"):
        result[field] = result[field].map(lambda value: redact_pii(value)[0])
    result["subject"] = result["subject"].replace("", "(no subject)")
    result["category"] = result["category"].map(
        lambda v: config.category_map.get(v.lower(), v.title() or "Unknown")
    )
    result["department"] = result["department"].map(
        lambda v: config.department_map.get(v.lower(), v.title() or "Unknown")
    )
    result["priority"] = result["priority"].map(
        lambda v: config.priority_map.get(v.lower(), "unknown")
    )
    result["language"] = result["language"].map(
        lambda v: config.language_map.get(v.lower(), v.lower() or "unknown")
    )
    tags = [c for c in raw.columns if re.fullmatch(r"tag_\d+", str(c))]
    result["tags"] = (
        raw.apply(
            lambda row: json.dumps(
                list(
                    dict.fromkeys(
                        redact_pii(clean_text(row[c]))[0] for c in tags if clean_text(row[c])
                    )
                )
            ),
            axis=1,
        )
        if tags
        else "[]"
    )
    result["source_version"] = raw["version"].map(clean_text) if "version" in raw else ""
    result["business_type"] = raw["business_type"].map(clean_text) if "business_type" in raw else ""
    valid = (
        result["customer_query"].str.len().between(config.min_query_chars, config.max_query_chars)
    )
    valid &= result["agent_response"].str.len().gt(0) & result["language"].isin(
        config.target_languages
    )
    result = result.loc[valid].copy()
    result["_exact_key"] = [
        digest(
            "\x1f".join(
                str(row[f]).casefold() for f in ("subject", "customer_query", "agent_response")
            )
        )
        for row in result.to_dict("records")
    ]
    return result.drop_duplicates("_exact_key").reset_index(drop=True)


def cluster_issues(frame: Any, config: PreprocessingSettings) -> list[int]:
    """Stateless features: no vocabulary/IDF fitted on future held-out examples.

    Exhaustive radius-neighbor queries in batches avoid a top-k duplicate blind spot.
    Grouping features are discarded, never used as retrieval embeddings.
    """
    count = len(frame)
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    for index, query in enumerate(frame["customer_query"]):
        key = issue_digest(query)
        if key in exact:
            union(index, exact[key])
        else:
            exact[key] = index
    corpus = [
        re.sub(r"\b(?:dear support|best regards|kind regards|thank you)\b", " ", value.casefold())
        for value in frame["customer_query"]
    ]
    matrix = HashingVectorizer(
        n_features=config.hash_features, alternate_sign=False, ngram_range=(1, 2), norm="l2"
    ).transform(corpus)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(matrix)
    for start in range(0, count, config.neighbor_batch_size):
        links = neighbors.radius_neighbors(
            matrix[start : start + config.neighbor_batch_size],
            radius=1 - config.near_dup_threshold + 1e-10,
            return_distance=False,
        )
        for offset, matches in enumerate(links):
            for match in matches:
                union(start + offset, int(match))
    return [find(i) for i in range(count)]


@dataclass
class PreparedData:
    splits: dict[str, Any]
    isolation: dict[str, Any]
    summary: dict[str, Any]
    config: PreprocessingSettings


def preprocess(raw: Any, config: PreprocessingSettings, source: dict[str, str]) -> PreparedData:
    frame = normalize(raw, config)
    targets = {"retrieval": config.retrieval_rows, "dev": config.dev_rows, "eval": config.eval_rows}
    if len(frame) < sum(targets.values()):
        raise ValueError("Not enough valid unique rows to satisfy the configured split sizes")
    frame["_cluster_id"] = cluster_issues(frame, config)
    sizes = frame.groupby("_cluster_id").size().to_dict()
    remaining = set(sizes)
    assignment = {}
    for offset, split in enumerate(("eval", "dev", "retrieval")):
        groups = sorted(remaining)
        random.Random(config.random_seed + offset).shuffle(groups)
        groups.sort(key=lambda group: sizes[group], reverse=True)
        used = 0
        for group in groups:
            if used + sizes[group] <= targets[split]:
                assignment[group] = split
                remaining.remove(group)
                used += sizes[group]
        if used != targets[split]:
            raise ValueError(
                "Whole duplicate groups cannot satisfy split targets; refusing to split a group"
            )
    selected = frame.loc[frame["_cluster_id"].isin(assignment)].copy()
    selected["split"] = selected["_cluster_id"].map(assignment)
    selected["ticket_id"] = selected["_source_row_index"].map(
        lambda i: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{config.source_dataset}:{i}"))
    )
    selected["source_ticket_id"] = selected["_source_row_index"].map(
        lambda i: f"{config.source_dataset}:{i}"
    )
    selected["source_dataset"] = config.source_dataset
    selected["resolution_status"] = "resolved"
    selected["created_at"] = pd.Series(pd.NaT, index=selected.index, dtype="datetime64[ns]")
    selected["metadata"] = selected.apply(
        lambda row: json.dumps(
            {
                "source_system": "ticket_dataset",
                "source_dataset": config.source_dataset,
                "source_dataset_url": source.get("url"),
                "source_has_native_id": False,
                "source_ticket_id_kind": "derived_locator",
                "acquisition_method": source["method"],
                "acquisition_path": source["path"],
                "source_sha256": source["sha256"],
                "source_row_index": int(row["_source_row_index"]),
                "source_version": row["source_version"],
                "business_type": row["business_type"],
                "near_dup_cluster_id": int(row["_cluster_id"]),
                "near_dup_cluster_size": int(sizes[row["_cluster_id"]]),
                "split": row["split"],
                "has_source_timestamp": False,
                "resolution_status_inferred_from_answer": True,
                "preprocessing_version": "ticket-preprocess-v2",
            },
            sort_keys=True,
        ),
        axis=1,
    )
    splits = {
        split: selected.loc[selected["split"] == split, FINAL_COLUMNS]
        .sort_values("source_ticket_id")
        .reset_index(drop=True)
        for split in SPLITS
    }
    evidence = {split: split_evidence(part) for split, part in splits.items()}
    check_disjoint(evidence)
    return PreparedData(
        splits,
        {"schema_version": "split-isolation-v1", "splits": evidence},
        {
            "raw_rows": len(raw),
            "valid_unique_rows": len(frame),
            "exported_rows": len(selected),
            "split_rows": targets,
            "discarded_valid_rows": len(frame) - len(selected),
            "split_leakage_detected": False,
            "pii_redacted": True,
            "resolution_status_policy": "Non-empty answer is an inferred resolution, not a verified source status",
            "near_duplicate_method": "stateless hashing word/bigram cosine radius clustering; not semantic equivalence",
        },
        config,
    )


def export(prepared: PreparedData, destination: Path, source: dict[str, str]) -> Path:
    """Create fresh artifacts only; hash eval bytes in memory before quarantine."""
    if destination.exists():
        raise ValueError("Output directory must be new; existing datasets are never overwritten")
    fresh = {split: split_evidence(frame) for split, frame in prepared.splits.items()}
    check_disjoint(fresh)
    if fresh != prepared.isolation["splits"] or any(
        json.loads(value).get("split") != split
        for split, frame in prepared.splits.items()
        for value in frame["metadata"]
    ):
        raise ValueError("Prepared split provenance changed before export")
    buffers = {split: frame.to_parquet(index=False) for split, frame in prepared.splits.items()}
    isolation = (json.dumps(prepared.isolation, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema_version": "ticket-splits-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "configuration": prepared.config.model_dump(mode="json"),
        "files": {
            split: {
                "path": f"tickets_{split}.parquet",
                "rows": len(prepared.splits[split]),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for split, data in buffers.items()
        },
        "isolation": {
            "path": "split_isolation.json",
            "sha256": hashlib.sha256(isolation).hexdigest(),
        },
    }
    artifacts = {f"tickets_{split}.parquet": data for split, data in buffers.items()}
    artifacts.update(
        {
            "split_isolation.json": isolation,
            "data_quality_report.json": (json.dumps(prepared.summary, indent=2) + "\n").encode(),
            "split_manifest.json": (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
        }
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    return destination
