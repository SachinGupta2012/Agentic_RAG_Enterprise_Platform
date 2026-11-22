# Retrieval

Phase 2 builds the searchable knowledge base from the 16,000-record retrieval split and exposes it
through a hybrid retrieval API.

## Storage and inference

| Concern | Choice | Where it runs |
| --- | --- | --- |
| Vector + payload storage | Qdrant Cloud collection `support_tickets_v1` | Qdrant Cloud (free tier) |
| Dense embeddings | `sentence-transformers/all-minilm-l6-v2`, 384 dimensions | Qdrant Cloud Inference (free) |
| Sparse embeddings | `qdrant/bm25` | Qdrant Cloud Inference (free) |
| Fusion | Reciprocal Rank Fusion, `k = 60` | Qdrant |
| Local database / embedding model | None | — |

The application host only runs the `qdrant-client` HTTP client. No vector database, embedding
model, PyTorch, or FastEmbed is installed.

## Collection layout

Each point uses the Phase 1 UUID5 `ticket_id` as its Qdrant point ID and carries:

- named dense vector `dense` (384 dimensions, cosine)
- named sparse vector `bm25`
- payload: `subject`, `customer_query`, `agent_response`, `category`, `department`, `priority`,
  `language`, `tags`, `resolution_status`, `source_ticket_id`, `source_dataset`,
  `source_row_index`, `source_version`, `pii_redactions`, `schema_version`, `split`

Payload indexes exist for `category`, `department`, `priority`, `language`, `resolution_status`,
`source_dataset`, `source_ticket_id`, `tags`, and `source_row_index`.

The dense text is `subject + customer query`. The sparse text additionally includes the agent
response so keyword retrieval can match resolution terms. Both are truncated before inference.

## Safety checks

`scripts/ingest_qdrant.py` refuses to run when:

- the retrieval file checksum does not match `split_manifest.json`;
- the row count is not 16,000;
- any required column is missing or contains nulls;
- any `ticket_id` is duplicated or is not a UUID;
- any row is not `resolved` or not `language == "en"`;
- any retrieval `ticket_id` also appears in dev or eval.

The script never reads `tickets_eval.parquet` for indexing; eval is used only for an identifier
overlap check. `--dry-run` validates and redacts without contacting Qdrant.

## PII redaction

Retrieved ticket content is untrusted. Email addresses, URLs, and phone-like values are redacted
before text is sent to Qdrant Cloud Inference and before it is stored. Queries are redacted before
embedding, and the API returns the redacted query plus a redaction count. Redaction counts for each
stored document are kept in the payload.

When later phases build prompts, retrieved ticket text must be inserted as quoted data and must
never be treated as model instructions. A retrieval result cannot override system behaviour.

PII redaction is intentionally conservative: it only replaces values with clear separators, so
dates and long numeric identifiers are not redacted. It is a safety net, not a compliance product.

## API

### `POST /api/v1/retrieval/search`

Request:

```json
{
  "query": "The account portal is offline and I cannot log in.",
  "top_k": 10,
  "filters": {
    "category": "Incident",
    "department": null,
    "priority": "high",
    "language": "en"
  }
}
```

Response:

```json
{
  "query": "The account portal is offline and I cannot log in.",
  "hits": [
    {
      "ticket_id": "7294fd0a-a460-58e0-b3e2-07058d7fe95d",
      "score": 0.0328,
      "score_type": "rrf",
      "subject": "Account Disruption",
      "snippet": "The account portal is offline...",
      "resolution_snippet": "Our team is investigating the outage...",
      "category": "Incident",
      "department": "Technical Support",
      "priority": "high",
      "language": "en",
      "tags": ["Account", "Outage"],
      "source_ticket_id": "tobiasbueck/multilingual-customer-support-tickets:1",
      "source_dataset": "tobiasbueck/multilingual-customer-support-tickets",
      "source_row_index": 1,
      "source_version": "51.0",
      "pii_redactions": {"email": 0, "phone": 0, "url": 0}
    }
  ],
  "count": 1,
  "retrieval_mode": "hybrid_rrf",
  "dense_model": "sentence-transformers/all-minilm-l6-v2",
  "sparse_model": "qdrant/bm25",
  "collection": "support_tickets_v1",
  "latency_ms": 84.2,
  "query_pii_redactions": {"email": 0, "phone": 0, "url": 0},
  "warnings": []
}
```

`score` is an RRF score, not a probability or a calibrated confidence value. Generated UUIDs are
derived locators; they are not source ticket IDs.

### `GET /api/v1/retrieval/health`

Returns `status`, collection name, document count, and the configured models. A missing or
unreachable index returns HTTP 503.

## Configuration

All settings are optional and fall back to the defaults below. Put real values in `.env`; never
commit `.env`.

```env
QDRANT_URL=https://your-cluster.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=support_tickets_v1
QDRANT_DENSE_MODEL=sentence-transformers/all-minilm-l6-v2
QDRANT_SPARSE_MODEL=qdrant/bm25
QDRANT_DENSE_VECTOR=dense
QDRANT_SPARSE_VECTOR=bm25
RETRIEVAL_PREFETCH_LIMIT=50
RETRIEVAL_TIMEOUT_SECONDS=60
RETRIEVAL_BATCH_SIZE=64
```

`QUADRANT_API_KEY` is accepted as an alias for `QDRANT_API_KEY` to avoid breaking older local
files. The API key is stored as a `SecretStr` and is never logged or returned by the API.

## Free-tier limits

Qdrant's free tier provides a single node with 1 GB RAM and 4 GB disk. The 16,000-document corpus
uses a small fraction of that. The free cloud-inference models are hosted in the US and are billed
at zero cost; paid models are not used.

If the collection or inference limit is exceeded, ingestion fails with a clear error. Retry the
script; upserting the same UUID point IDs is idempotent.

## Evaluation discipline

- The retrieval index contains only the 16,000 retrieval rows.
- Dev is used for proxy checks and manual inspection.
- Eval is quarantined and never queried by the retrieval pipeline.
- `scripts/dev_retrieval_check.py` reports metadata agreement (category/department/priority match at
  k). This is a proxy, not relevance ground truth. Do not report it as a quality metric.
- Retrieval quality must be measured later against the eval split with an explicit, non-fabricated
  evaluation protocol.
