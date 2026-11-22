# Data Pipeline

## Source

- **Dataset:** `tobiasbueck/multilingual-customer-support-tickets` (Kaggle)
- **URL:** https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
- **License:** CC BY 4.0
- **Hugging Face mirror:** `miku1211/customer-support-tickets`

The raw dataset has 61,765+ records and the following relevant columns:

| Source column | Internal field | Notes |
| --- | --- | --- |
| `subject` | `subject` | Filled with `(no subject)` when missing |
| `body` | `customer_query` | Required; empty rows are dropped |
| `answer` | `agent_response` | Required for the retrieval corpus |
| `type` | `category` | Normalized to Incident / Request / Problem / Change |
| `queue` | `department` | Normalized department labels |
| `priority` | `priority` | Normalized to low / medium / high / critical / unknown |
| `language` | `language` | Normalized ISO codes; English filter by default |
| `tag_1..tag_8` | `tags` | Deduplicated JSON list |
| `version` | `metadata.source_version` | Dataset version, not ticket version |

There is no source timestamp. `created_at` is therefore null and
`metadata.has_source_timestamp` is `false`; ingestion time must not be presented as the ticket's
original creation time.

## Processing stages

1. **Acquisition** — Kaggle attachment, Kaggle API download, or Hugging Face fallback.
2. **Schema validation** — resolve expected columns and reject missing required fields.
3. **Profiling** — row count, dtypes, missingness, unique values, label distributions.
4. **Cleaning** — drop empty queries/answers, enforce query length bounds, normalize text.
5. **Language filtering** — English-only by default (`TARGET_LANGUAGES = {"en"}`).
6. **Label normalization** — category, department, priority, language.
7. **Exact deduplication** — SHA-256 fingerprint over normalized subject + query + answer.
8. **Near-duplicate clustering** — TF-IDF cosine similarity over boilerplate-stripped text,
   union-find at `NEAR_DUP_THRESHOLD` (0.88 by default).
9. **Group-aware splitting** — eval carved out first, then dev, then retrieval. Whole clusters are
   assigned to exactly one split; leftover clusters are dropped.
10. **Finalization** — deterministic UUID5 `ticket_id`, `source_ticket_id` provenance, JSON tags and
    metadata, `resolution_status`.
11. **Validation** — split-overlap and cluster-leakage assertions.
12. **Export** — Parquet splits, `data_quality_report.json`, `split_manifest.json` with SHA-256
    checksums.

## Splits

| Split | Target | Purpose | Retrieval-visible |
| --- | ---: | --- | --- |
| `retrieval` | 16,000 | Historical corpus for hybrid search | Yes |
| `dev` | 2,000 | Prompt and retrieval tuning | No |
| `eval` | 2,000 | Independent evaluation | No |

The evaluation split must never be embedded or ingested into Qdrant. Near-duplicate clusters are
kept intact within a single split to prevent leakage.

## Internal schema

| Field | Type | Notes |
| --- | --- | --- |
| `ticket_id` | UUID string | Deterministic UUID5; not a source ID |
| `source_ticket_id` | string | `{dataset}:{source_row_index}` |
| `subject` | string | |
| `customer_query` | string | |
| `agent_response` | string | Historical resolution |
| `category` | string | |
| `department` | string | |
| `priority` | string | |
| `language` | string | |
| `tags` | JSON string | List of tags |
| `resolution_status` | string | `resolved` in the current corpus |
| `source_dataset` | string | Provenance |
| `created_at` | timestamp or null | Null: source has no timestamps |
| `metadata` | JSON string | Source row, version, cluster, split |

## Outputs

Local runs write to `data/processed/`. Kaggle runs write to `/kaggle/working/processed/` and bundle
the files into `/kaggle/working/processed_tickets.zip`.
