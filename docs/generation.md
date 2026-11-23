# Phase 3: grounded response generation

Phase 3 adds an internal FastAPI response endpoint over the existing retrieval
service. It does not re-index Qdrant, resume relevance judging, or read the
reserved evaluation split. The approved 23-query retrieval pilot remains the
baseline; the full 150-query retrieval evaluation is deferred.

## API

Start the existing app with `uv run uvicorn app.main:app --reload`.

`POST /api/v1/responses/generate` accepts:

```json
{"query":"How can I resolve a browser sign-in problem?","top_k":3,"mode":"hybrid"}
```

The request also accepts the existing metadata `filters` and `prefetch_limit`.
`top_k` is limited to 1..5. Hybrid remains a provisional default. An optional
`provider` must be explicitly enabled in server configuration. Requests never
silently change provider or model after failures.

Responses contain `status` (`answered` or `insufficient_evidence`), `answer`,
structured `claims`, and bounded redacted `sources`. Each claim includes evidence
with a `source_id` and a quotation. Sources map those references to ticket IDs.
The server appends actual ticket IDs to answer claims. Other fields record the
provider, configured model, provider-reported model when available, prompt
version (`response-v1`), context SHA-256, token usage when reported, redaction
counts, and retrieval/generation/total timing.

`GET /api/v1/responses/providers` reports configuration presence and enabled
providers without making API calls or exposing credentials. Configuration
presence is not an inference-availability check.

Error contracts:

| HTTP status | Meaning |
| --- | --- |
| 422 | Invalid request or unsupported provider name |
| 429 | Local admission budget or upstream quota unavailable; see `Retry-After` |
| 502 | Provider transport, incomplete output, or upstream failure |
| 503 | Disabled/unconfigured provider or unavailable retrieval |

## Context and grounding

1. Redact the question using the existing email/URL/phone patterns.
2. Search the existing collection and fetch full payloads for the retrieved IDs.
   Only payloads marked `split=retrieval` can supply generation context.
3. Deduplicate tickets, redact again, and bound subject, issue and resolution
   text. The serialized source context is limited to 8,000 characters by default;
   sources are dropped if necessary to fit a conservative UTF-8 token estimate.
4. Send a fixed system instruction plus JSON data containing the question and
   sources. Ticket/query text is explicitly treated as untrusted data. Historical
   fixes are precedents, not evidence of current policy or performed actions.
5. Validate strict JSON. Each returned claim must cite a source in the supplied
   context and quote at least 12 characters that actually occur in that source's
   redacted resolution. The server generates ticket citation markers. Unknown
   citations, invented quotes, empty answers and malformed payloads produce a
   fixed insufficient-evidence response without returning the unverified text.
6. Redact generated claim text before returning it. Empty usable context skips
   the LLM entirely. The LLM can also abstain on unrelated or insufficient context.

These checks validate citation membership and quotation matching. They do not
prove semantic entailment, source relevance, current policy, or resistance to
every prompt-injection attempt. Regex PII redaction has the existing limitations.
No uncalibrated RRF score is presented as an answer-confidence probability.

## Provider configuration

See `.env.example`. Runtime generation settings are separate from `JUDGE_*`
settings and do not import `app.evaluation`.

| Provider | Runtime model | Default enabled |
| --- | --- | --- |
| Groq | `openai/gpt-oss-120b` | Yes |
| Gemini | `gemini-3.5-flash-lite` | No |
| Cloudflare | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | No |

To enable explicit alternatives, set
`GENERATION_ENABLED_PROVIDERS=["groq","gemini","cloudflare"]`. Select one with
the request's `provider` field or change `GENERATION_PROVIDER` server-side and
restart the app. No runtime fallback chain is enabled. Model settings are
provider-specific; arbitrary models/URLs cannot be supplied in API requests.

Gemini model migration was explicit: the inherited `gemini-2.0-flash` returned
HTTP 404 for generation. The listed `gemini-2.5-flash-lite` also returned 404,
with a provider message directing new users to `gemini-3.5-flash-lite`. Only the
new runtime-generation profile was updated. Historical judge settings and qrels
are unchanged. Model-list presence alone did not establish availability.

Cloudflare JSON mode can return a parsed object in `result.response`; the
adapter supports that as well as a JSON string. The schema is validated after
either representation. The provider-reported model is retained when present.

Defaults are temperature 0, 1,536 output tokens, a 30-second provider timeout,
and a shared process-local budget of six calls and 6,000 estimated tokens per
minute per provider. Reservations are reconciled to reported actual usage.
Admission failures return immediately. Upstream 429 responses establish a
cooldown. Timeouts are not automatically resent because remote completion may
have occurred; only one short retry is permitted for HTTP 5xx. Raw provider
error bodies and prompts are not logged. Multiple API worker processes do not
share local budgets; remote account quotas remain authoritative.

All validation uses existing free-tier credentials without plan changes or
paid tools. See primary provider documentation:
[Groq compatibility](https://console.groq.com/docs/openai),
[Gemini generateContent](https://ai.google.dev/api/generate-content),
[Gemini free-tier pricing](https://ai.google.dev/gemini-api/docs/pricing), and
[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/models/llama-3.3-70b-instruct-fp8-fast/).

## Bounded validation

```bash
uv run python scripts/evaluate_generation.py --providers groq --real-queries 3 --pilot-dir data/eval/pilot_23_20261123T090130651464Z
```

This explicitly selects at most three real queries from the approved pilot
using seed 7, reconstructs cached hybrid top-3 context, and runs two synthetic
controls per selected provider. `--providers gemini,cloudflare --real-queries 0`
checks alternatives independently. No Qdrant requests or relevance judging are
performed by this script. It permits one quota wait of at most 60 seconds per
case and stops starting cases after five minutes; there is no background job.

Timestamped `data/eval/phase3_generation_*/` directories contain sanitized
summary reports and local response-review JSONL. Review rows contain redacted
business text and should remain local. Human groundedness, usefulness and
abstention labels are blank. LLM qrel source relevance is recorded only as a
proxy; it does not measure whether a generated claim follows from its evidence.
Synthetic controls are plumbing checks rather than representative quality
metrics. The initial real-query sample produced two answers and one model
abstention. No human quality score has been fabricated.

One additional end-to-end smoke request through the actual API used live Qdrant
retrieval, full payload fetch, and Groq generation. It returned HTTP 200 with
three sources and four cited claims. Its approximately 3.05-second total timing
is a single observation, not a P50/P95 benchmark.

Response generation remains an internal development capability. Broad semantic
quality and production readiness have not been established by this small pilot.
No reranker, LangGraph agents, database migrations, or mandatory HITL are added.

All three final runtime model profiles passed the answerable and unrelated
synthetic controls (six successful controls across providers). Initial failed
checks are preserved separately instead of being overwritten. Gemini's final
profile is `gemini-3.5-flash-lite`; the failures of its earlier profiles are not
counted as successful checks. Only Groq is enabled by default in the runtime.
