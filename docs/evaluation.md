# Retrieval Evaluation

Offline evaluation for dense, sparse and hybrid retrieval. Everything here lives outside the
runtime API and outside any agent workflow. Nothing in this document changes application behaviour.

## Scope and isolation

| Rule | Detail |
| --- | --- |
| Query source | `tickets_dev.parquet` only |
| Eval split | Never read by the evaluation scripts or the judge |
| Retrieval corpus | The existing 16,000-record `support_tickets_v1` collection, unmodified |
| Qdrant writes | None; all evaluation access is read-only |
| Runtime coupling | None; `app.evaluation` is imported only by scripts under `scripts/` |

The evaluation scripts do not tune the corpus, do not re-index, and do not modify processed
datasets.

## Pipeline

```text
dev split
  -> select 150 queries            scripts/build_dev_queries.py
  -> pool candidates (dense/sparse/hybrid top-10)
                                   scripts/build_candidate_pool.py
  -> LLM judge pooled pairs        scripts/generate_qrels.py
  -> manual verification worksheet scripts/verify_qrels_sample.py
  -> metrics for all three modes   scripts/evaluate_retrieval.py
  -> latency benchmark             scripts/benchmark_latency.py
```

All artifacts are written to git-ignored `data/eval/`. Only sanitized summaries and this
documentation are committed.

## Query selection

`scripts/build_dev_queries.py`:

- reads only `tickets_dev.parquet`, verified against the SHA-256 in `split_manifest.json`;
- uses the `customer_query` text, whitespace-normalized;
- drops rows shorter than three characters;
- collapses duplicate normalized query texts;
- takes a deterministic sample (`--seed`, default `42`);
- sorts by `ticket_id` and assigns stable IDs `dev-q0001`...`dev-q0150`.

The dev ticket is **not** treated as a gold answer. Phase 1 assigned whole near-duplicate clusters to
one split, so a dev ticket has no near-duplicate in the retrieval corpus. Relevance must therefore be
judged, not assumed. This avoids constructing query-document pairs that are trivially relevant.

## Candidate pooling

`scripts/build_candidate_pool.py` runs each query through all three modes with `--depth 10`, merges
the results, and deduplicates by stable `ticket_id` while recording which modes returned each
document and at which rank.

The same pooled judgments are then used to score every mode. No mode receives its own judgment set,
so the comparison is like-for-like.

Depth is configurable for future experiments (`--depth`).

## LLM relevance judging

`scripts/generate_qrels.py` uses a free provider (Groq by default, Gemini as a fallback) through the
existing `httpx` dependency. No paid API and no local generative model is used.

### Structured output

The judge must return strict JSON:

```json
{"relevance": 0, "reason": "short explanation"}
```

Grades:

| Grade | Meaning |
| ---: | --- |
| `0` | Not relevant: a different underlying issue, or no useful resolution |
| `1` | Partially relevant: related issue, or a resolution that only partly helps |
| `2` | Highly relevant: same underlying issue and a directly useful resolution |

Output is validated against a Pydantic model. Malformed output is retried and then recorded as a
failure.

### Untrusted content

Ticket text is untrusted data. It is fenced inside `<ticket>` tags, and the system prompt instructs
the model to ignore any instructions found inside it. The judge never follows ticket-embedded
instructions and never treats ticket text as instructions.

Category, department, priority, language and tags are deliberately excluded from the judge prompt so
relevance cannot be inferred from metadata alone.

### Failures and resumability

- Transient failures (429, 5xx, timeouts, transport errors) are retried with exponential backoff.
- Permanent failures (auth, bad request, unknown model) stop the run immediately.
- Malformed output is retried a configurable number of times.
- Every judgment is appended immediately, and existing `(query_id, doc_id)` pairs are skipped on
  re-run.
- Failed pairs are written to `judgment_failures.jsonl` and are **never** converted into grade `0`.

### Rate limiting

`JUDGE_REQUESTS_PER_MINUTE` sets a minimum interval between requests. `0` means unlimited. The
implementation does not assume any particular provider quota.

### Pilot before scaling

Run a small pilot first and inspect validity, token usage and failures before judging everything:

```bash
uv run python scripts/generate_qrels.py --queries-limit 10 --max-pairs 60
```

## Qrels format

One judgment per `(query_id, doc_id)` pair, independent of retrieval mode. JSONL fields:

| Field | Meaning |
| --- | --- |
| `query_id` | Stable dev query ID |
| `doc_id` | Qdrant point ID of the judged ticket |
| `relevance` | `0`, `1` or `2` |
| `judge_provider` | `groq` or `gemini` |
| `judge_model` | Exact model string used |
| `judgment_source` | `llm` (human grades live only in the verification worksheet) |
| `judgment_timestamp` | ISO-8601 UTC |
| `reason` | Short supporting explanation |
| `prompt_version` | `judge-v1` |

Duplicate pairs and invalid grades or timestamps are rejected at load time.

## Manual verification

`scripts/verify_qrels_sample.py` builds a stratified worksheet of ~20 pairs across grades `0`, `1`
and `2`. Each row shows the query, the candidate ticket, the LLM grade and the LLM explanation, plus
blank `human_relevance` and `human_comment` fields.

Rules:

- LLM grades are preserved and never overwritten.
- Human grades are assigned independently by a reviewer; this repository does not fabricate them.
- Re-running the script keeps human grades already recorded for the same pair.
- Once human grades exist, the report computes exact agreement, within-one agreement, Cohen's kappa
  and a confusion matrix.

Manual verification is an offline QA step. It is not Human-in-the-Loop and does not add approval
checkpoints to the live application.

## Metrics and evaluation policy

| Metric | Definition |
| --- | --- |
| Recall@5 / Recall@10 | Share of judged-relevant documents found in the top 5 / 10 |
| Precision@5 | Share of the first five slots that are judged relevant |
| MRR | Reciprocal rank of the first judged-relevant document |
| nDCG@5 | Graded gain (`2^grade - 1`) with logarithmic discount |
| P50 / P95 latency | Client-observed request latency, linear interpolation |

A document counts as relevant at grade `>= 1`.

### Pooled-relevance policy

Judgments come from a pooled candidate set. A document that no mode surfaced was never judged. All
reported metrics are therefore **pooled relevance-based estimates**. Recall@k in particular is not
absolute corpus recall, because the corpus was never fully judged.

Every report states:

- the number of queries evaluated;
- the number of judged documents;
- judgment coverage (share of retrieved documents that carry a judgment);
- how many queries had no judged-relevant documents.

Unjudged documents are treated as non-relevant while ranking. That is the standard pooled assumption,
and coverage is reported next to every metric so the assumption stays visible. Queries with no
judged-relevant document are excluded from aggregates and counted separately; they are never scored
as zero.

Failed retrieval runs are counted and excluded from latency percentiles.

## Latency methodology

`scripts/benchmark_latency.py`:

- uses the same fixed query set for every mode;
- discards `--warmup` requests per mode;
- measures `--repeats` full requests per mode;
- reports P50 and P95 with linear interpolation;
- reports both the client-observed wall time and the Qdrant-reported request time.

Client-observed latency includes Qdrant Cloud Inference for query embedding and network time.
Server-side inference cannot be separated from retrieval client-side, so it is included rather than
estimated. LLM judging time is never part of a retrieval latency measurement.

## Reproducibility

Reports record:

- query file path and SHA-256;
- candidate pool and pooled-document SHA-256 (pool summary);
- qrels file SHA-256;
- collection name;
- dense and sparse model names;
- RRF `k` and `top_k`;
- judge provider, model and prompt version;
- generation timestamps.

## Commands

```bash
# 1. Queries
uv run python scripts/build_dev_queries.py --limit 150

# 2. Candidate pool (read-only, live Qdrant)
uv run python scripts/build_candidate_pool.py --depth 10

# 3. Judge pilot, then the rest
uv run python scripts/generate_qrels.py --queries-limit 10 --max-pairs 60
uv run python scripts/generate_qrels.py

# 4. Manual verification worksheet
uv run python scripts/verify_qrels_sample.py --size 20

# 5. Metrics for all modes
uv run python scripts/evaluate_retrieval.py --modes dense,sparse,hybrid --top-k 10

# 6. Latency
uv run python scripts/benchmark_latency.py --warmup 5 --repeats 30
```

If no provider key is configured, `generate_qrels.py` exits with a clear blocker message and writes
no judgments. The evaluation infrastructure still works; only live judging is unavailable.

## Limitations

- Relevance is LLM-generated and only spot-checked by a human sample. Metrics are estimates, not
  ground truth.
- Recall is a pooled estimate, not absolute corpus recall.
- The dense and sparse modes index different text (dense excludes the agent response), which biases
  the comparison.
- LLM judgments may inherit model bias and are not perfectly stable across reruns or model versions.
- Cross-encoder reranking is not implemented, so no reranked arm is reported.

## Recovery safeguards (November 2026)

The recovery run preserves the existing `groq` / `openai/gpt-oss-120b` /
`judge-v1` judgments. Before recovery, the existing evaluation artifacts were
copied to `data/eval/recovery_snapshots/` with a SHA-256 manifest.

`generate_qrels.py` defaults to eight requests per minute. Override explicitly
with `--requests-per-minute`; zero is rejected by this runner. Groq response
headers additionally gate the next request against remaining token capacity.
HTTP 429 responses honor provider cooldowns; a cooldown longer than one minute
or exhausted daily request capacity stops the run for a later quota window.
Retries include bounded jitter. Provider/model changes are never automatic.

The runner now writes these sidecars alongside qrels:

- `.jsonl.resume.json`: pinned input hashes and generation settings. Existing
  qrels must match the selected provider, model and prompt version.
- `.jsonl.lock`: exclusive single-writer guard. A stale lock requires inspecting
  the recorded process ID before removing it; never remove a live writer's lock.
- `.jsonl.pending.json`: the pair about to be requested. A saved pair is safely
  reconciled on restart. An unsaved pending pair stops automatic resending because
  remote completion may have occurred. Inspect it before deciding how to proceed.
- `.requests.jsonl`: HTTP statuses, quota headers and numeric token usage; no
  ticket text, keys, generated reasons, or raw provider error bodies.

Each qrel and failure is flushed and synced to disk. Run summaries receive unique
timestamped filenames, preserving earlier summaries. An interrupted in-flight
request is not automatically retried. This prevents blind duplicate requests;
it does not claim distributed exactly-once delivery. Historical judgments
predate generation-settings pinning and per-request telemetry, so their complete
generation settings and token usage cannot be reconstructed.

For a bounded pilot:

```bash
uv run python scripts/generate_qrels.py --provider groq --model openai/gpt-oss-120b --max-pairs 3
```

For continuation across quota windows and automatic finalization:

```bash
uv run python scripts/complete_phase2.py
```

The continuation worker waits for an existing judge writer, then resumes missing
pairs. It waits through explicit quota cooldowns (24 hours if no useful reset
is available), but stops on uncertain requests, configuration drift and other
failures. It makes at most 30 quota-window attempts. Keep the computer awake;
this is a local process, not a hosted scheduler. Restart the same command after
machine restart; inspect stale locks first. `data/eval/recovery_status.json`
records its phase, remaining pairs, cooldown deadline, or final report path.
Creating `data/eval/recovery.stop` stops the supervisor at its next checkpoint;
an active judge child finishes its current run before the supervisor stops.

Only after all pooled pairs are judged does the worker generate a new 20-pair
worksheet with blank human labels, evaluate all three retrieval modes, benchmark
latency, and run the quality gates including six read-only live Qdrant contracts.
New reports have timestamped names. Completion requires full retrieved judgment
coverage and no retrieval failures. Metrics remain **LLM-judgment-based pooled
estimates**. Blank human labels do not establish human agreement.

## Approved 23-query pilot results

The full 150-query judging run was stopped at the user's request. Its 481 saved
judgments are preserved; 23 queries have fully judged pools containing 462 pairs
and 453 unique documents. The other 19 saved judgments belong to an incomplete
query and are excluded from this pilot. The full pool still has 2,596 unjudged
pairs. The background worker remains stopped, with its stop marker in place.

The pilot reconstructs each mode's top-10 ranking from saved `retrieved_by`
mode/rank references. It makes no LLM or retrieval API calls and checks input
hashes, judge provenance, contiguous ranks, and complete pair coverage. Run:

```bash
uv run python scripts/evaluate_pilot.py
```

Each run creates a new timestamped `data/eval/pilot_23_*/` directory containing
`pilot_report.json`, `per_query_metrics.jsonl`, frozen `pilot_qrels.jsonl`, and a
copy of the existing 20-pair worksheet with blank human labels. No source
artifacts are overwritten. The expected fully judged query count defaults to 23
and a changed count stops execution rather than silently changing the scope.

**These are LLM-judgment-based pooled estimates on a 23-query pilot.** All 23
queries have at least one judged-relevant candidate; judgment coverage is 100%
within this pilot. Metrics are macro averages, relevance is grade >= 1, nDCG
uses gain `2^grade - 1`, and MRR is truncated at rank 10.

| Mode | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.2608 | 0.5250 | 0.9565 | 0.9710 | 0.7823 |
| Sparse | 0.2420 | 0.4908 | 0.8957 | 0.9710 | 0.6757 |
| Hybrid | 0.2658 | 0.5170 | 0.9739 | 0.9710 | 0.7541 |

Dense leads on Recall@10 and nDCG@5; hybrid leads on Recall@5 and Precision@5.
These small observed differences do not establish statistical superiority.

Latency is reused from the historical benchmark generated at
2026-11-23T08:03:44.856752+00:00, with 30 measured requests per mode after five
warmups. It is not a fresh latency measurement of the 23-query subset.

| Mode | Historical P50 (ms) | Historical P95 (ms) |
| --- | ---: | ---: |
| Dense | 330.98 | 7735.72 |
| Sparse | 244.82 | 305.88 |
| Hybrid | 323.81 | 457.92 |

The pilot uses `dev-q0001` through `dev-q0023`, a processing-order prefix of the
ticket-ID-sorted source sample, rather than a newly randomized sample. Selection
bias and the small sample limit generalization. Grades are strongly positive:
30 irrelevant, 168 partially relevant, 264 highly relevant. Human review is
pending, so the high precision values do not establish independently verified
support usefulness. Recall is relative to the judged pool, not all 16,000
corpus tickets. Dense and sparse also index different text. The original pool
summary omits RRF k; only the separate historical latency artifact records k=60.

The pilot evaluation is complete. The original 150-query evaluation remains
deferred, and no reranking, agents, or mandatory HITL were introduced.
