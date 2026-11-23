"""Manual verification sampling and human/LLM agreement statistics.

The worksheet is generated with blank human fields. LLM grades are preserved and
never overwritten; the reviewer assigns independent grades and the comparison is
reported separately.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from app.evaluation.qrels import QrelsRecord

GRADES = (0, 1, 2)


def stratified_sample(
    records: Sequence[QrelsRecord],
    size: int,
    seed: int = 7,
) -> list[QrelsRecord]:
    """Sample judgments across relevance grades as evenly as the data allows."""

    if size < 1:
        raise ValueError("size must be at least 1")

    by_grade: dict[int, list[QrelsRecord]] = {}
    for record in sorted(records, key=lambda item: (item.query_id, item.doc_id)):
        by_grade.setdefault(record.relevance, []).append(record)

    rng = random.Random(seed)
    for bucket in by_grade.values():
        rng.shuffle(bucket)

    grades = sorted(by_grade)
    selected: list[QrelsRecord] = []
    while len(selected) < size and any(by_grade[grade] for grade in grades):
        for grade in grades:
            if len(selected) >= size:
                break
            if by_grade[grade]:
                selected.append(by_grade[grade].pop())

    return selected


def cohen_kappa(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Return Cohen's kappa for two raters over the same graded items."""

    if not pairs:
        return None

    total = len(pairs)
    observed = sum(1 for human, llm in pairs if human == llm) / total
    human_counts = Counter(human for human, _ in pairs)
    llm_counts = Counter(llm for _, llm in pairs)
    expected = sum(
        (human_counts[grade] / total) * (llm_counts[grade] / total)
        for grade in set(human_counts) | set(llm_counts)
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Summarize agreement between human and LLM grades."""

    if not pairs:
        return {
            "verified_pairs": 0,
            "exact_agreement": None,
            "within_one_agreement": None,
            "cohen_kappa": None,
            "confusion": {},
        }

    total = len(pairs)
    confusion: dict[str, int] = {}
    for human, llm in pairs:
        key = f"human_{human}_llm_{llm}"
        confusion[key] = confusion.get(key, 0) + 1

    return {
        "verified_pairs": total,
        "exact_agreement": round(sum(1 for human, llm in pairs if human == llm) / total, 4),
        "within_one_agreement": round(
            sum(1 for human, llm in pairs if abs(human - llm) <= 1) / total, 4
        ),
        "cohen_kappa": _round(cohen_kappa(pairs)),
        "confusion": dict(sorted(confusion.items())),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def extract_verification_pairs(
    worksheet: Sequence[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return ``(human, llm)`` pairs for rows that already have a human grade."""

    pairs: list[tuple[int, int]] = []
    for row in worksheet:
        human = row.get("human_relevance")
        llm = row.get("llm_relevance")
        if human is None or llm is None:
            continue
        human_grade = int(human)
        if human_grade not in GRADES:
            raise ValueError(f"human_relevance must be 0, 1 or 2, got {human_grade}")
        pairs.append((human_grade, int(llm)))
    return pairs
