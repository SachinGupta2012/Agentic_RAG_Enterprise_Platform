"""Generate the approved 23-query pilot using existing local artifacts only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.jsonl import read_jsonl, sha256_file, write_json, write_jsonl
from app.evaluation.pilot import evaluate_pilot
from app.evaluation.qrels import load_qrels, qrels_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--expected-queries", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    data = args.data_dir
    inputs = {
        name: data / name
        for name in (
            "candidate_pool.jsonl",
            "candidate_pool_summary.json",
            "dev_queries.jsonl",
            "pool_documents.jsonl",
            "qrels_llm_v1.jsonl",
            "latency_benchmark.json",
            "manual_verification_recovery_pilot.jsonl",
        )
    }
    summary = json.loads(inputs["candidate_pool_summary.json"].read_text(encoding="utf-8"))
    for name, key in (
        ("candidate_pool.jsonl", "pool_file_sha256"),
        ("dev_queries.jsonl", "query_file_sha256"),
        ("pool_documents.jsonl", "documents_file_sha256"),
    ):
        if sha256_file(inputs[name]) != summary[key]:
            raise ValueError(f"Input fingerprint mismatch: {name}")
    records = load_qrels(inputs["qrels_llm_v1.jsonl"])
    fingerprint = qrels_fingerprint(records)
    if (
        fingerprint["providers"],
        fingerprint["models"],
        fingerprint["prompt_versions"],
        fingerprint["sources"],
    ) != (
        ["groq"],
        ["openai/gpt-oss-120b"],
        ["judge-v1"],
        ["llm"],
    ):
        raise ValueError("Unexpected judge provenance")
    result = evaluate_pilot(
        read_jsonl(inputs["candidate_pool.jsonl"]), records, expected_queries=args.expected_queries
    )
    selected_ids = set(result["selected_query_ids"])
    selected_records = [record for record in records if record.query_id in selected_ids]
    worksheet = read_jsonl(inputs["manual_verification_recovery_pilot.jsonl"])
    selected_pairs = {(record.query_id, record.doc_id) for record in selected_records}
    if len(worksheet) != 20 or not all(
        (row["query_id"], row["doc_id"]) in selected_pairs
        and row.get("human_relevance") is None
        and not row.get("human_comment")
        for row in worksheet
    ):
        raise ValueError("Expected 20 pilot worksheet pairs with blank human labels")
    latency = json.loads(inputs["latency_benchmark.json"].read_text(encoding="utf-8"))
    if latency["query_file_sha256"] != summary["query_file_sha256"]:
        raise ValueError("Latency benchmark used a different source query file")
    now = datetime.now(UTC)
    output = args.output_dir or data / ("pilot_23_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    per_query = result.pop("per_query")
    write_jsonl(output / "per_query_metrics.jsonl", per_query)
    write_jsonl(output / "pilot_qrels.jsonl", (record.model_dump() for record in selected_records))
    write_jsonl(output / "manual_verification.jsonl", worksheet)
    report = {
        "status": "pilot_complete_full_150_query_evaluation_deferred",
        "generated_at": now.isoformat(),
        "metrics_label": "LLM-judgment-based pooled estimates on a 23-query pilot",
        "methodology": {
            "rankings": "Reconstructed from saved retrieved_by mode/rank references; no new retrieval",
            "selection": "All 23 fully judged queries at interruption, dev-q0001..dev-q0023",
            "aggregation": "Macro average across queries with at least one judged-relevant pooled candidate",
            "relevant_grade_threshold": 1,
            "ndcg_gain": "2^grade - 1",
            "mrr_depth": 10,
            "pilot_pool_judgment_coverage": 1.0,
            "new_llm_requests": 0,
            "new_retrieval_requests": 0,
        },
        **result,
        "pilot_qrels": qrels_fingerprint(selected_records),
        "grade_distribution": dict(
            sorted(Counter(str(record.relevance) for record in selected_records).items())
        ),
        "full_pool": {
            "pairs": summary["pooled_candidates"],
            "saved_judgments": len(records),
            "remaining_pairs": summary["pooled_candidates"] - len(records),
            "judgment_coverage": len(records) / summary["pooled_candidates"],
        },
        "retrieval_configuration": {
            key: summary[key]
            for key in ("collection", "dense_model", "sparse_model", "depth_per_mode", "modes")
        },
        "latency": {
            "source": str(inputs["latency_benchmark.json"]),
            "scope": "Reused historical benchmark; 30 measured requests per mode on the source query set, not a fresh 23-query measurement",
            **latency,
        },
        "manual_verification": {
            "pairs": len(worksheet),
            "human_labels": "blank",
            "human_agreement": None,
            "worksheet": "manual_verification.jsonl",
        },
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()
        },
        "limitations": [
            "Small interruption-selected prefix of ticket-ID-sorted queries; not a newly randomized or representative sample.",
            "Pooled LLM relevance is not human ground truth or absolute corpus recall.",
            "No judgments exist outside the pooled candidates; MRR is evaluated only to depth 10.",
            "Dense and sparse index different text, which limits attribution of differences to retrieval methods alone.",
            "Historical latency is reported separately; no pilot latency observations were fabricated.",
            "The original candidate-pool summary does not record RRF k; the separate historical latency benchmark records k=60.",
            "Human verification is pending. Full 150-query evaluation remains deferred.",
        ],
    }
    write_json(output / "pilot_report.json", report)
    print(f"Report: {output / 'pilot_report.json'}")
    print(f"Queries: {result['queries']}; judgments: {result['judged_pairs']}; coverage: 100%")
    for mode in result["modes"]:
        print(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
