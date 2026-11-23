"""Ranking metrics for offline retrieval evaluation.

Evaluation policy
-----------------
Relevance is graded ``0`` (not relevant), ``1`` (partially relevant) and ``2``
(highly relevant). A document counts as *relevant* for recall, precision and MRR
when its grade is at least ``1``; nDCG uses the graded gains.

Judgments come from a pooled candidate set, so every metric here is a
*pooled-relevance estimate*: a document that no retrieval mode surfaced was
never judged, and cannot be credited. Recall@k is therefore reported as a
pooled estimate with explicit coverage, never as absolute corpus recall.

Queries without any judged-relevant document are excluded from aggregates by
the reporting layer, because the qrels cannot distinguish a good ranking from a
bad one for those queries. Unjudged documents are treated as non-relevant while
ranking, which is the standard pooled-evaluation assumption; judgment coverage
is reported alongside every metric so that assumption stays visible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

RelevanceMap = Mapping[str, int]
RELEVANT_THRESHOLD = 1


def _relevant_ids(relevance: RelevanceMap, threshold: int = RELEVANT_THRESHOLD) -> set[str]:
    return {doc_id for doc_id, grade in relevance.items() if grade >= threshold}


def recall_at_k(ranked: Sequence[str], relevance: RelevanceMap, k: int) -> float | None:
    """Return pooled Recall@k, or ``None`` when the query has no judged relevant doc."""

    relevant = _relevant_ids(relevance)
    if not relevant or k <= 0:
        return None
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevance: RelevanceMap, k: int) -> float | None:
    """Return Precision@k over the first ``k`` slots, or ``None`` if undefined."""

    if not _relevant_ids(relevance) or k <= 0:
        return None
    hits = sum(1 for doc_id in ranked[:k] if relevance.get(doc_id, 0) >= RELEVANT_THRESHOLD)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], relevance: RelevanceMap) -> float | None:
    """Return the reciprocal rank of the first relevant document."""

    if not _relevant_ids(relevance):
        return None
    for position, doc_id in enumerate(ranked, start=1):
        if relevance.get(doc_id, 0) >= RELEVANT_THRESHOLD:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevance: RelevanceMap, k: int) -> float | None:
    """Return graded nDCG@k using ``2**grade - 1`` gains."""

    if not _relevant_ids(relevance) or k <= 0:
        return None

    gains = [2 ** relevance.get(doc_id, 0) - 1 for doc_id in ranked[:k]]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal_gains = [2**grade - 1 for grade in ideal_grades]
    idcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(ideal_gains, start=1))
    if idcg == 0:
        return None
    return dcg / idcg


def percentile(values: Sequence[float], p: float) -> float:
    """Return the ``p``-th percentile with linear interpolation.

    Uses the same definition as ``numpy.percentile`` with ``method="linear"`` so
    reported latency percentiles match common tooling.
    """

    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])

    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def mean(values: Sequence[float | None]) -> float | None:
    """Average the defined values, returning ``None`` when nothing is defined."""

    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
