"""Bounded Phase 3 smoke/response pilot; cached Phase 2 context, no Qdrant calls.

Only the explicitly selected providers are called. Defaults to three real
Groq responses plus answerable/unrelated synthetic controls. No LLM relevance
judging, provider switching, background work, or reserved evaluation data.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.jsonl import read_jsonl, sha256_file, write_json, write_jsonl
from app.evaluation.qrels import load_qrels, qrels_by_query
from app.generation.config import GenerationSettings, Provider
from app.generation.schemas import EvidenceSource
from app.generation.service import ResponseService
from app.providers.chat import HTTPChatClient, ProviderError, ProviderRateLimited
from app.retrieval.pii import redact_pii


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="groq",
        help="Explicit comma-separated providers; no automatic fallback",
    )
    parser.add_argument("--real-queries", type=int, default=3, choices=range(0, 4))
    parser.add_argument("--pilot-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = args.providers.split(",")
    if not selected or any(p not in ("groq", "gemini", "cloudflare") for p in selected):
        parser.error("Unknown provider")
    providers: list[Provider] = [p for p in ("groq", "gemini", "cloudflare") if p in selected]
    settings = GenerationSettings(enabled_providers=providers)
    service = ResponseService(settings, {p: HTTPChatClient(p, settings) for p in providers})
    pilot = json.loads((args.pilot_dir / "pilot_report.json").read_text())
    # Pin cached source files to the approved retrieval pilot.
    for name in ("dev_queries.jsonl", "pool_documents.jsonl"):
        source = pilot["provenance"][name]
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("Pilot source fingerprint changed")
    queries = {
        row["query_id"]: row["text"]
        for row in read_jsonl(Path(pilot["provenance"]["dev_queries.jsonl"]["path"]))
    }
    documents = {
        row["doc_id"]: row
        for row in read_jsonl(Path(pilot["provenance"]["pool_documents.jsonl"]["path"]))
    }
    qrels = qrels_by_query(load_qrels(args.pilot_dir / "pilot_qrels.jsonl"))
    ranked = {
        row["query_id"]: row["ranked_doc_ids"]
        for row in read_jsonl(args.pilot_dir / "per_query_metrics.jsonl")
        if row["mode"] == "hybrid"
    }
    sample = random.Random(7).sample(sorted(ranked), args.real_queries)
    cases: list[tuple[Provider, str, str, list[EvidenceSource], str | None]] = []
    for query_id in sample:
        sources = [
            EvidenceSource(
                source_id="",
                ticket_id=doc_id,
                subject=documents[doc_id]["subject"],
                customer_query=documents[doc_id]["customer_query"],
                resolution=documents[doc_id]["agent_response"],
            )
            for doc_id in ranked[query_id][:3]
        ]
        if "groq" in providers:
            cases.append(("groq", query_id, queries[query_id], sources, None))
    synthetic = EvidenceSource(
        source_id="S1",
        ticket_id="synthetic-cache-ticket",
        subject="Browser sign-in problem",
        customer_query="Cannot sign in to the portal",
        resolution="Clear the browser cache, then sign in again. If the issue persists, contact support.",
    )
    for provider in providers:
        cases.extend(
            [
                (
                    provider,
                    "synthetic_answerable",
                    "How can I resolve a browser sign-in problem?",
                    [synthetic],
                    "answered",
                ),
                (
                    provider,
                    "synthetic_unrelated",
                    "What is the warranty period of my washing machine?",
                    [synthetic],
                    "insufficient_evidence",
                ),
            ]
        )
    output = Path("data/eval") / (
        "phase3_generation_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir()
    results = []
    unavailable: set[Provider] = set()
    started = time.monotonic()
    for provider, query_id, query, sources, expected in cases:
        query = redact_pii(query)[0]
        row: dict[str, object] = {
            "provider": provider,
            "model": settings.model_for(provider),
            "case_id": query_id,
            "expected_status": expected,
            "query": query,
        }
        if provider in unavailable or time.monotonic() - started > 300:
            row["error"] = "Skipped: provider unavailable or five-minute pilot deadline reached"
        else:
            for attempt in range(2):
                try:
                    response = service.generate(
                        query, sources, provider=provider, retrieval_mode="cached_hybrid_top3"
                    )
                    row["response"] = response.model_dump()
                    row["expectation_met"] = response.status == expected if expected else None
                    row["human_groundedness"] = None
                    row["human_usefulness"] = None
                    row["human_abstention_appropriate"] = None
                    if query_id in qrels:
                        source_ids = {s.source_id: s.ticket_id for s in response.sources}
                        cited_ids = {
                            source_ids[e.source_id]
                            for claim in response.claims
                            for e in claim.evidence
                        }
                        row["llm_qrel_citation_relevance"] = {
                            doc_id: qrels[query_id].get(doc_id) for doc_id in cited_ids
                        }
                    break
                except ProviderRateLimited as exc:
                    if (
                        attempt == 0
                        and exc.retry_after <= 60
                        and time.monotonic() - started + exc.retry_after <= 300
                    ):
                        print(f"{provider}: bounded quota wait of {exc.retry_after}s", flush=True)
                        time.sleep(exc.retry_after)
                        continue
                    row["error"] = "Provider quota unavailable; no provider/model substitution"
                    row["retry_after"] = exc.retry_after
                    unavailable.add(provider)
                except ProviderError as exc:
                    row["error"] = str(exc)
                    unavailable.add(provider)
                break
        results.append(row)
        write_jsonl(output / "response_review.jsonl", [row], append=True)
        print(
            f"{provider} / {query_id}: {'recorded' if 'response' in row else 'unavailable'}",
            flush=True,
        )
    successes = [row for row in results if "response" in row]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Bounded response-generation pilot on cached hybrid top-3 context",
        "retrieval_pilot": str(args.pilot_dir),
        "retrieval_pilot_report_sha256": sha256_file(args.pilot_dir / "pilot_report.json"),
        "configured_models": {p: settings.model_for(p) for p in providers},
        "real_query_ids": sample,
        "cases": len(results),
        "responses_received": len(successes),
        "errors_or_skips": len(results) - len(successes),
        "new_qdrant_requests": 0,
        "answered": sum(row["response"]["status"] == "answered" for row in successes),
        "unverified_evidence_abstentions": sum(
            row["response"]["reason"] == "unverified_evidence" for row in successes
        ),
        "synthetic_controls_passed": sum(row.get("expectation_met") is True for row in successes),
        "synthetic_controls_planned": 2 * len(providers),
        "semantic_groundedness": "not independently measured; quotation matching does not establish entailment",
        "human_quality_review": "pending; labels are blank in response_review.jsonl",
        "note": "Synthetic controls are plumbing checks, not relevance judgments or a representative quality benchmark. LLM qrels indicate source-query relevance only, not claim correctness.",
    }
    write_json(output / "generation_report.json", report)
    print(f"Report: {output / 'generation_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
