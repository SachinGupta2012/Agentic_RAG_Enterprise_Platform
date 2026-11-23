"""Aggregation of offline retrieval metrics into report structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.evaluation.metrics import (
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.qrels import QrelsRecord, judgment_coverage
from app.evaluation.runner import ModeRun

METRICS_CAVEAT = (
    "Pooled-relevance estimate from LLM-generated judgments, not human ground truth "
    "and not absolute corpus recall."
)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def evaluate_mode(
    mode: str,
    runs: Sequence[ModeRun],
    relevance_by_query: Mapping[str, Mapping[str, int]],
    records: Sequence[QrelsRecord],
) -> dict[str, Any]:
    """Compute metrics for one retrieval mode over the shared qrels."""

    mode_runs = [run for run in runs if run.mode == mode]
    if not mode_runs:
        raise ValueError(f"no runs recorded for mode {mode!r}")

    failed = [run for run in mode_runs if run.error]
    successful = [run for run in mode_runs if not run.error]

    recalls_5: list[float | None] = []
    recalls_10: list[float | None] = []
    precisions_5: list[float | None] = []
    reciprocal_ranks: list[float | None] = []
    ndcgs_5: list[float | None] = []
    unjudged_queries = 0
    empty_queries = 0

    for run in successful:
        if run.empty:
            empty_queries += 1
        relevance = relevance_by_query.get(run.query_id, {})
        if not any(grade >= 1 for grade in relevance.values()):
            unjudged_queries += 1
            continue
        ranked = list(run.ranked_doc_ids)
        recalls_5.append(recall_at_k(ranked, relevance, 5))
        recalls_10.append(recall_at_k(ranked, relevance, 10))
        precisions_5.append(precision_at_k(ranked, relevance, 5))
        reciprocal_ranks.append(reciprocal_rank(ranked, relevance))
        ndcgs_5.append(ndcg_at_k(ranked, relevance, 5))

    latencies = [run.wall_ms for run in successful]
    ranked_by_query = {run.query_id: list(run.ranked_doc_ids) for run in successful}

    return {
        "mode": mode,
        "queries": len(mode_runs),
        "queries_evaluated": len(mode_runs) - len(failed) - unjudged_queries,
        "queries_without_relevant_judgments": unjudged_queries,
        "empty_result_queries": empty_queries,
        "failed_queries": len(failed),
        "recall_at_5": _round(mean(recalls_5)),
        "recall_at_10": _round(mean(recalls_10)),
        "precision_at_5": _round(mean(precisions_5)),
        "mrr": _round(mean(reciprocal_ranks)),
        "ndcg_at_5": _round(mean(ndcgs_5)),
        "p50_latency_ms": _round(percentile(latencies, 50), 2),
        "p95_latency_ms": _round(percentile(latencies, 95), 2),
        "judgment_coverage": _round(judgment_coverage(ranked_by_query, records)),
    }


def build_report(
    *,
    modes: Sequence[str],
    runs: Sequence[ModeRun],
    records: Sequence[QrelsRecord],
    relevance_by_query: Mapping[str, Mapping[str, int]],
    metadata: Mapping[str, Any],
    failures: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Build the full evaluation report."""

    metrics = [evaluate_mode(mode, runs, relevance_by_query, records) for mode in modes]
    return {
        "metrics_label": "llm-judgment-based pooled estimate",
        "caveat": METRICS_CAVEAT,
        "policy": {
            "relevant_grade_threshold": 1,
            "unjudged_documents": "treated as non-relevant for ranking; coverage reported",
            "queries_without_relevant_judgments": "excluded from aggregates and counted",
            "failed_runs": "excluded from latency percentiles and counted",
            "recall": "pooled relevance-based estimate; not absolute corpus recall",
        },
        "metadata": dict(metadata),
        "modes": metrics,
        "failures": list(failures),
    }
