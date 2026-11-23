"""Offline metrics from saved pooling ranks, without inventing latency samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.evaluation.candidates import MODES
from app.evaluation.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.qrels import QrelsRecord, qrels_by_query


def evaluate_pilot(
    pool: Sequence[dict[str, Any]],
    records: Sequence[QrelsRecord],
    *,
    expected_queries: int = 23,
) -> dict[str, Any]:
    """Select fully judged pools and reconstruct validated saved top-10 rankings."""
    relevance = qrels_by_query(records)
    query_ids = [row["query_id"] for row in pool]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate pool query ID")
    selected = [
        row
        for row in pool
        if row["candidates"]
        and all(c["doc_id"] in relevance.get(row["query_id"], {}) for c in row["candidates"])
    ]
    if len(selected) != expected_queries:
        raise ValueError(f"Expected {expected_queries} fully judged queries; found {len(selected)}")
    per_query: list[dict[str, Any]] = []
    for row in selected:
        query_id = row["query_id"]
        doc_ids = [c["doc_id"] for c in row["candidates"]]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Duplicate candidate within a query")
        grades = {doc_id: relevance[query_id][doc_id] for doc_id in doc_ids}
        for mode in MODES:
            ranked_refs = [
                (ref["rank"], candidate["doc_id"])
                for candidate in row["candidates"]
                for ref in candidate["retrieved_by"]
                if ref["mode"] == mode
            ]
            ranked_refs.sort()
            if [rank for rank, _ in ranked_refs] != list(range(1, 11)):
                raise ValueError(f"Expected unique contiguous ranks 1..10 for {query_id}/{mode}")
            ranked = [doc_id for _, doc_id in ranked_refs]
            per_query.append(
                {
                    "query_id": query_id,
                    "mode": mode,
                    "ranked_doc_ids": ranked,
                    "pooled_candidates": len(grades),
                    "judged_relevant_documents": sum(grade >= 1 for grade in grades.values()),
                    "recall_at_5": recall_at_k(ranked, grades, 5),
                    "recall_at_10": recall_at_k(ranked, grades, 10),
                    "precision_at_5": precision_at_k(ranked, grades, 5),
                    "mrr": reciprocal_rank(ranked, grades),
                    "ndcg_at_5": ndcg_at_k(ranked, grades, 5),
                    "judgment_coverage": 1.0,
                }
            )
    metric_names = ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "ndcg_at_5")
    modes = []
    for mode in MODES:
        rows = [row for row in per_query if row["mode"] == mode]
        evaluated = sum(row["judged_relevant_documents"] > 0 for row in rows)
        metrics = {name: mean([row[name] for row in rows]) for name in metric_names}
        modes.append(
            {
                "mode": mode,
                "queries": len(rows),
                "queries_evaluated": evaluated,
                "queries_without_relevant_judgments": len(rows) - evaluated,
                "judgment_coverage": 1.0,
                **{
                    key: None if value is None else round(value, 4)
                    for key, value in metrics.items()
                },
            }
        )
    selected_ids = [row["query_id"] for row in selected]
    selected_pairs = {(row["query_id"], c["doc_id"]) for row in selected for c in row["candidates"]}
    return {
        "selected_query_ids": selected_ids,
        "queries": len(selected),
        "judged_pairs": len(selected_pairs),
        "unique_documents": len({doc_id for _, doc_id in selected_pairs}),
        "pool_queries_total": len(pool),
        "excluded_queries": len(pool) - len(selected),
        "modes": modes,
        "per_query": per_query,
    }
