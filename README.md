# Enterprise Support Intelligence Platform

Multi-agent RAG platform for enterprise customer support. The backend currently provides data
preprocessing, hybrid retrieval, and grounded LLM response generation. LangGraph orchestration is
planned next; the React + TypeScript frontend and PostgreSQL-backed persistence remain future work.

## Current status

- **Phase 0** — project structure, Python environment, dependency management, Ruff, mypy, pytest,
  pre-commit.
- **Phase 1** — dataset ingestion and preprocessing notebook for Kaggle. 20,000 English resolved
  tickets split into 16,000 retrieval / 2,000 dev / 2,000 eval.
- **Phase 2** — Qdrant Cloud knowledge base, PII redaction, dense + sparse + hybrid retrieval,
  retrieval API, and an offline retrieval evaluation harness. No local database or embedding model
  is required.
- **Phase 2 evaluation** — the approved 23-query pilot is complete; the full 150-query evaluation
  remains deferred. Retrieval metrics are LLM-judgment-based pooled estimates.
- **Phase 3** — retrieval-backed response generation is accepted as functionally complete.
  Independent semantic answer-quality evaluation remains deferred.
- **Phase 4** — LangGraph multi-agent orchestration is planned below; implementation has not begun.

## Phase 4 plan: LangGraph multi-agent orchestration

Preserve the existing retrieval/generation services and endpoints. Add an opt-in orchestration
layer under `backend/app/agents/`, with a separate proposed `POST /api/v1/orchestration/respond`
endpoint. Use small typed `StateGraph` nodes and deterministic conditional routes initially; an
agent role does not require its own LLM call. Keep all phase planning and progress in this README.

### Shared state and service reuse

Use a versioned `TypedDict` state, with Pydantic validation at request and node-output boundaries:

- Request: run/request IDs, redacted query, intent, urgency, language, and validated filters.
- Evidence: retrieval query/version, bounded redacted sources, ticket IDs, ranks, and context hash.
- Work: evidence-linked findings, explicitly separate hypotheses, draft `ResponseResult`, QA checks,
  routing reason, final outcome, and an optional knowledge proposal.
- Control: deadline, retrieval/draft/repair counters, provider-attempt and token budgets, pinned
  provider/model/prompt versions, sanitized errors, and a bounded event history.

Give each node ownership of its fields and return partial updates. Replace draft/evidence fields
on revision; use deterministic, event-ID-deduplicating reducers only for appendable events. Start
with sequential execution. Put service instances, credentials and trusted authorization context
in runtime dependencies, outside serializable state. Begin with request-scoped state; defer
durable checkpoints to the PostgreSQL persistence work, including retention and replay controls.

Retrieval calls `QdrantTicketStore.search` and `fetch_payloads`, retaining the retrieval-split guard,
redaction, filters and context limits. Resolution calls `ResponseService.generate` on that evidence;
calling `respond` here would retrieve a second time. Reuse `bound_sources`, provider adapters and
their budgets. Keep the standalone Phase 3 endpoint working. Any extraction of shared validation
helpers must preserve its behavior. Pin explicit provider/model selection for the entire run.

### Seven agent responsibilities

| Agent | Responsibility and output |
| --- | --- |
| Triage | Validate/redact input; classify supported intent, urgency and language; request clarification or route unsupported cases. Start with rules. |
| Retrieval | Read the existing Qdrant collection; return bounded evidence with ticket provenance. No ingestion or reranking. |
| Investigation | Identify missing/conflicting evidence; produce evidence-linked findings or one bounded query refinement. Hypotheses cannot become facts. |
| Resolution | Generate a cited draft through the existing response service, or preserve its insufficient-evidence decision. No claimed account actions. |
| QA | Check schema, citation/source identity, quotations, context version and outcome consistency. Initially deterministic; quotation matching does not prove semantic correctness. |
| Escalation | Return a terminal clarification/escalation/unavailable outcome, reason and redacted handoff summary. No external ticket creation or mandatory human approval. |
| Knowledge | After QA passes, prepare a deduplicated, evidence-linked knowledge-gap/reuse proposal. QA approval alone cannot establish a new fact; no automatic corpus writes. |

### Routing

```mermaid
flowchart TD
    S([START]) --> T[Triage]
    T -->|supported| R[Retrieval]
    T -->|clarification or unsupported| E[Escalation]
    R -->|usable evidence| D[Resolution]
    R -->|specific evidence gap and budget remains| I[Investigation]
    R -->|no recoverable evidence| E
    I -->|one refined query| R
    I -->|existing evidence sufficient| D
    I -->|unresolved| E
    D -->|answered draft| Q[QA]
    D -->|abstention| E
    Q -->|repairable and repair unused| I
    Q -->|failed or exhausted| E
    Q -->|passed| K[Knowledge proposal]
    K --> F([END])
    E --> F
```

The MVP routes QA success directly to END; Investigation and Knowledge are added later. Only a
QA-passed draft can become an answered outcome. Knowledge is best-effort and cannot invalidate that
answer. Use a workflow envelope with explicit `answered`, `needs_clarification`, `escalated` or
`unavailable` outcomes, carrying the existing response schema where applicable. Every conditional
route checks remaining visit, I/O and deadline budgets; exhaustion routes to Escalation. Preserve
HTTP 429/503 semantics and `Retry-After` for service failures. No indefinite loops.

### Failure bounds, security and observability

Proposed configurable starting caps: two retrieval passes, two draft attempts, one Investigation
visit/repair episode, four total LLM HTTP attempts including provider retries, a 120-second run
deadline, and an explicit graph step limit of 16. Reserve shared token/attempt budgets before I/O;
count nested service retries and respect remaining time. The provider layer owns transport retries;
do not stack whole-node generation retries on top. Retry only safe transient reads within budget.
Do not automatically resend uncertain LLM timeouts. Quota failures return `Retry-After` promptly;
auth/configuration failures stop immediately. Infrastructure failures yield `unavailable`, not a
fabricated support resolution. Log unexpected programming errors safely and return a sanitized 500.

Treat queries, tickets and agent outputs as untrusted data. Allowlist tools/routes and validate all
updates; model text cannot change budgets, authorization filters, provider settings or tool access.
Carry trusted access scope from the API boundary; never infer tenant access from ticket text. Keep
credentials and raw content out of state logs/traces. Preserve the existing corpus and reserved
evaluation split; introduce no paid APIs, autonomous actions, automatic knowledge ingestion or HITL.

Emit local structured events per run/node: route/reason, duration, attempt counts, token usage,
provider/model/prompt and graph versions, context hashes, QA results and terminal outcome. Keep
business content redacted and bounded. Reuse existing timing/usage fields; external Langfuse tracing
is optional later, without introducing a paid dependency or enabling content export by default.

### Incremental delivery and acceptance

1. **Contracts and skeleton:** confirm the locked LangGraph API; define state, outcome schemas,
   dependency injection, budgets and fake-node routing tests. No live LLM calls required.
2. **Small working graph:** implement Triage → Retrieval → Resolution → QA, plus terminal
   Escalation. Reuse current services; add the opt-in endpoint and compare against Phase 3 behavior.
3. **Bounded Investigation:** add one evidence-gap/refinement path; test useful recovery and forced
   termination. Add LLM investigation only if deterministic findings prove insufficient and its
   separate call fits the same run-wide budgets.
4. **Knowledge and hardening:** add proposal-only Knowledge, deduplication, failure isolation,
   trace summaries and a capped development pilot. Persistence and approved publishing come later.

Test shared-state validation/reducers, every route, empty/conflicting evidence, invalid citations,
prompt injection, access-filter preservation, quota/auth/timeout failures, deadline/step exhaustion,
retry amplification and request isolation. Assert that Knowledge cannot write to Qdrant and failed
QA cannot return an answer. Run existing API regressions and Ruff/mypy/pytest/notebook gates. Use
cached pilot evidence and mocked providers first; live tests remain explicitly capped and optional.
Acceptance requires deterministic routing, enforced bounds, traceable terminal outcomes and unchanged
standalone retrieval/generation behavior. The small retrieval pilot and structural QA do not replace
the deferred semantic answer-quality evaluation or establish production readiness.

Design references: [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
for typed state, nodes, conditional routing and runtime dependencies, and
[Thinking in the Graph API](https://docs.langchain.com/oss/python/langgraph/thinking-in-graph-api)
for incremental workflow decomposition. The budgets and staged rollout above are project proposals.

## Repository layout

```text
.
├── backend/
│   ├── app/
│   │   ├── api/v1/          # REST API routes
│   │   ├── core/            # settings and cross-cutting concerns
│   │   ├── db/              # database session and base classes
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # application services
│   │   ├── agents/          # LangGraph agents
│   │   ├── retrieval/       # dense, sparse and hybrid retrieval
│   │   └── providers/       # LLM provider abstraction
│   └── tests/
├── data/
│   ├── raw/                 # downloaded source data (git-ignored)
│   ├── interim/             # intermediate artifacts (git-ignored)
│   ├── processed/           # standardized splits (git-ignored)
│   └── eval/                # evaluation-only artifacts (git-ignored)
├── notebooks/
│   └── 01_dataset_ingestion.ipynb
├── docs/
└── pyproject.toml
```

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Git
- For data ingestion: a Kaggle account and API token (`~/.kaggle/kaggle.json`), or internet access
  for the Hugging Face fallback.
- For retrieval: a free Qdrant Cloud cluster URL and API key.

## Setup

```bash
uv sync --extra dev
```

This creates `.venv` with the backend and development tooling. Optional extras are installed only
when needed:

- `data` — pandas, pyarrow, scikit-learn, Kaggle/Hugging Face downloaders, plotting. Needed for
  local notebook runs and split validation.
- `retrieval` — pandas and pyarrow only. Needed to run the Qdrant ingestion script locally.
- `notebook` — JupyterLab and ipykernel. Needed only for local notebook execution.
- `ai` — LangGraph, LangChain, Langfuse, RAGAS, Redis (Phase 3+).

Local data processing:

```bash
uv sync --extra data --extra notebook --extra dev
```

AI development:

```bash
uv sync --extra ai --extra data --extra notebook --extra dev
```

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_notebooks.py
```

Install the hooks once:

```bash
uv run pre-commit install
```

## Data pipeline

Source: [`tobiasbueck/multilingual-customer-support-tickets`](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
(CC BY 4.0), mirrored on Hugging Face as `miku1211/customer-support-tickets`.

Run `notebooks/01_dataset_ingestion.ipynb` on Kaggle or locally. See `notebooks/README.md` for both
workflows.

Outputs:

- `tickets_retrieval.parquet` — 16,000-record historical retrieval corpus.
- `tickets_dev.parquet` — 2,000 records for prompt and retrieval tuning.
- `tickets_eval.parquet` — 2,000 held-out records for evaluation.
- `data_quality_report.json` — cleaning counts, distributions, duplicate statistics, leakage check.
- `split_manifest.json` — row counts and SHA-256 checksums.

The evaluation split is quarantined and must never be ingested into Qdrant.

After the notebook finishes, validate the exported files (requires the `data` extra):

```bash
uv run python scripts/validate_splits.py
```

## Retrieval index

The retrieval corpus is stored in Qdrant Cloud. Embeddings are generated with Qdrant Cloud
Inference (`sentence-transformers/all-minilm-l6-v2` and `qdrant/bm25`), both free models, so no
embedding model or vector database is installed on the application host.

1. Create a free Qdrant Cloud cluster.
2. Put the cluster URL and API key in `.env` (see `.env.example`).
3. Install the ingestion dependencies:

   ```bash
   uv sync --extra retrieval --extra dev
   ```

4. Validate the local data and prepare documents without connecting:

   ```bash
   uv run python scripts/ingest_qdrant.py --dry-run
   ```

5. Create the collection and ingest the 16,000 retrieval tickets:

   ```bash
   uv run python scripts/ingest_qdrant.py --recreate
   ```

The script verifies the retrieval checksum from `split_manifest.json`, rejects unresolved or
non-English rows, and asserts that retrieval identifiers do not overlap dev or eval. It writes
`data/processed/qdrant_ingestion_report.json`.

Run the API:

```bash
uv run uvicorn app.main:app --reload
```

Endpoints:

- `POST /api/v1/retrieval/search` — `dense`, `sparse` or `hybrid` retrieval (`hybrid` is the default
  and fuses dense + BM25 with RRF).
- `GET /api/v1/retrieval/health` — collection readiness, document count and dimension validation.
- `GET /health` — root-level liveness probe.

Search modes:

```json
{"query": "account portal offline", "top_k": 5, "mode": "dense"}
{"query": "account portal offline", "top_k": 5, "mode": "sparse"}
{"query": "account portal offline", "top_k": 5, "mode": "hybrid"}
```

Use the dev split for proxy checks only:

```bash
uv run python scripts/dev_retrieval_check.py --limit 20 --top-k 5
```

The eval split is never used by the retrieval pipeline.

## Retrieval evaluation

Retrieval quality is measured offline against LLM-generated relevance judgments, with a small manual
verification sample. Dense, sparse and hybrid modes are scored on the same query set and the same
judgments. Metrics are pooled-relevance estimates, not human ground truth and not absolute corpus
recall.

```bash
uv run python scripts/build_dev_queries.py --limit 150
uv run python scripts/build_candidate_pool.py --depth 10
uv run python scripts/generate_qrels.py --queries-limit 10 --max-pairs 60   # pilot
uv run python scripts/generate_qrels.py                                     # full run, resumable
uv run python scripts/verify_qrels_sample.py --size 20
uv run python scripts/evaluate_retrieval.py --modes dense,sparse,hybrid
uv run python scripts/benchmark_latency.py --warmup 5 --repeats 30
```

All evaluation artifacts are written to git-ignored `data/eval/`. Judging requires a free Groq or
Gemini key in `.env`. See [`docs/evaluation.md`](docs/evaluation.md) for the full methodology, the
qrels format, the evaluation policy and its limitations.

## Important notes

Phase 3 response generation is implemented at `POST /api/v1/responses/generate`.
It uses the existing retrieval service, returns ticket citations with verified
source quotations, and abstains when usable evidence is absent or output fails
validation. `GET /api/v1/responses/providers` shows configuration status. Groq is
the default; alternatives require explicit enablement and selection. See
[`docs/generation.md`](docs/generation.md) for configuration, API behavior,
bounded evaluation, and limitations.

- Generated ticket identifiers are UUIDs. Source provenance is preserved separately in
  `source_ticket_id` and the `metadata` JSON column.
- No local vector database or embedding model is installed. Qdrant Cloud stores vectors and payload;
  Qdrant Cloud Inference generates dense and BM25 vectors.
- `all-MiniLM-L6-v2` replaces the originally specified `BAAI/bge-small-en-v1.5` because only the
  former is a free Qdrant Cloud Inference model. Both are 384-dimensional and interface-compatible.
  See [`docs/retrieval.md`](docs/retrieval.md) for the trade-off.
- Cross-encoder reranking (`BAAI/bge-reranker-base`) is **not implemented**. It is deferred until the
  hybrid baseline has been measured.
- PII redaction is regex-based and limited to email, URL and phone-like patterns. It is a safety net,
  not a compliance product.
- Human-in-the-loop approval is intentionally **not** part of this design. The workflow is
  autonomous, with QA as the final gate and escalation as a terminal outcome. The manual verification
  worksheet is offline evaluation QA, not HITL.
- No paid LLM APIs or local generative models are introduced. The Phase 3 runtime
  uses configured free-tier providers separately from the offline judge.
