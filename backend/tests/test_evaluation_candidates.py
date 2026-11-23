"""Tests for candidate pooling and deduplication."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.candidates import (
    PoolDocument,
    PoolError,
    RetrievalRef,
    load_pool,
    load_pool_documents,
    pool_candidates,
    pool_entry,
    write_pool_documents,
)
from app.evaluation.jsonl import write_jsonl


def test_pool_dedupes_and_records_provenance() -> None:
    pooled = pool_candidates({"dense": ["a", "b"], "sparse": ["b", "c"], "hybrid": ["a", "c"]})

    assert [candidate.doc_id for candidate in pooled] == ["a", "b", "c"]
    assert pooled[0].retrieved_by == (
        RetrievalRef("dense", 1),
        RetrievalRef("hybrid", 1),
    )
    assert pooled[1].retrieved_by == (
        RetrievalRef("dense", 2),
        RetrievalRef("sparse", 1),
    )
    assert pooled[2].retrieved_by == (
        RetrievalRef("sparse", 2),
        RetrievalRef("hybrid", 2),
    )


def test_pool_requires_at_least_one_mode() -> None:
    with pytest.raises(PoolError):
        pool_candidates({})


def test_pool_entry_shape() -> None:
    entry = pool_entry("dev-q0001", "query text", pool_candidates({"dense": ["a"]}))

    assert entry["query_id"] == "dev-q0001"
    assert entry["query"] == "query text"
    assert entry["candidates"] == [{"doc_id": "a", "retrieved_by": [{"mode": "dense", "rank": 1}]}]


def test_load_pool_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    write_jsonl(
        path,
        [
            pool_entry("dev-q0001", "q1", pool_candidates({"dense": ["a", "b"]})),
            pool_entry("dev-q0002", "q2", pool_candidates({"sparse": ["c"]})),
        ],
    )

    assert load_pool(path) == {"dev-q0001": ["a", "b"], "dev-q0002": ["c"]}


def test_load_pool_rejects_duplicate_candidates(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    write_jsonl(
        path,
        [
            {
                "query_id": "dev-q0001",
                "query": "q",
                "candidates": [{"doc_id": "a"}, {"doc_id": "a"}],
            }
        ],
    )

    with pytest.raises(PoolError, match="duplicate doc_id"):
        load_pool(path)


def test_load_pool_rejects_duplicate_queries(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    write_jsonl(
        path,
        [
            {"query_id": "dev-q0001", "query": "q", "candidates": []},
            {"query_id": "dev-q0001", "query": "q", "candidates": []},
        ],
    )

    with pytest.raises(PoolError, match="duplicate query_id"):
        load_pool(path)


def test_pool_documents_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    document = PoolDocument(
        doc_id="doc-a",
        subject="Subject",
        customer_query="Issue",
        agent_response="Resolution",
        category="Incident",
        department="Support",
        priority="high",
        language="en",
        tags=("Account",),
    )

    assert write_pool_documents(path, [document]) == 1

    loaded = load_pool_documents(path)
    assert loaded["doc-a"] == document
