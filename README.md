# Enterprise Support Intelligence Platform

Multi-agent RAG platform for enterprise customer support. This repository currently contains the
backend foundation and the data-engineering pipeline. The AI layer and frontend are added in later
phases.

## Current status

- **Phase 0** — project structure, Python environment, dependency management, Ruff, mypy, pytest,
  pre-commit.
- **Phase 1** — dataset ingestion and preprocessing notebook for Kaggle. 20,000 English resolved
  tickets split into 16,000 retrieval / 2,000 dev / 2,000 eval.
- **Phase 2** — Qdrant Cloud knowledge base, PII redaction, hybrid dense + BM25 retrieval, and
  retrieval API. No local database or embedding model is required.

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
│   │   ├── retrieval/       # hybrid retrieval and reranking
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

- `POST /api/v1/retrieval/search` — hybrid dense + BM25 search fused with RRF.
- `GET /api/v1/retrieval/health` — collection readiness and document count.

Use the dev split for proxy checks only:

```bash
uv run python scripts/dev_retrieval_check.py --limit 20 --top-k 5
```

The eval split is never used by the retrieval pipeline.

## Important notes

- Generated ticket identifiers are UUIDs. Source provenance is preserved separately in
  `source_ticket_id` and the `metadata` JSON column.
- No local vector database or embedding model is installed. Qdrant Cloud stores vectors and payload;
  Qdrant Cloud Inference generates dense and BM25 vectors.
- Human-in-the-loop approval is intentionally **not** part of this design. The workflow is
  autonomous, with QA as the final gate and escalation as a terminal outcome.
- No paid LLM APIs or local generative models are used. Provider integration is added in Phase 4.
