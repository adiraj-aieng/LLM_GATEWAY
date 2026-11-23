# LLM Gateway — API Contract

> Wire-compatible with the OpenAI Chat Completions API, plus gateway-specific headers and an
> `/admin/*` surface for the dashboard.
>
> **Companion docs:** [`architecture.md`](./architecture.md) · [`schema.md`](./schema.md)

---

## 1. Conventions

| | |
|---|---|
| Base URL | `http://localhost:8000` (dev) |
| Versioning | `/v1` for the OpenAI-compatible surface; `/admin` for gateway management |
| Auth (public) | `Authorization: Bearer sk-gw-<key>` |
| Auth (admin) | `Authorization: Bearer <ADMIN_TOKEN>` (env var, separate from API keys) |
| Content type | `application/json` |
| Streaming | `text/event-stream` (SSE) |
| Timestamps | ISO-8601 UTC |

### Gateway response headers (attached to every response)

| Header | Example | Meaning |
|---|---|---|
| `x-request-id` | `3f2a…` | UUID; present in every log row and error body |
| `x-gateway-cache` | `EXACT_HIT` \| `SEMANTIC_HIT` \| `MISS` \| `BYPASS` | how the response was produced |
| `x-gateway-similarity` | `0.9412` | cosine similarity (semantic hit only) |
| `x-gateway-model-upstream` | `qwen2.5-coder:3b` | which upstream actually served it |
| `x-gateway-route-reason` | `rule:code_block` | why the router chose it |
| `x-gateway-latency-ms` | `18` | total gateway-side latency (excludes upstream) |
| `x-ratelimit-limit` | `100000` | token bucket capacity |
| `x-ratelimit-remaining` | `87400` | tokens left after this request |
| `retry-after` | `12` | seconds (429 only) |

### Error envelope

Must match OpenAI's shape — LangChain, the OpenAI SDKs and OpenRouter all branch on
`error.type` / `error.code`:

```json
{
  "error": {
    "message": "Rate limit exceeded for api key (tokens). Retry in 12s.",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  },
  "request_id": "3f2a9c1e-…"
}
```

| HTTP | `type` | `code` |
|---|---|---|
| 400 | `invalid_request_error` | `invalid_body`, `unsupported_parameter`, `n_greater_than_1` |
| 401 | `authentication_error` | `invalid_api_key`, `missing_api_key` |
| 403 | `permission_error` | `model_not_allowed`, `admin_token_required` |
| 404 | `not_found_error` | `model_not_found`, `route_not_found` |
| 413 | `invalid_request_error` | `prompt_too_long` |
| 429 | `rate_limit_error` | `rate_limit_exceeded`, `concurrency_limit_exceeded` |
| 500 | `server_error` | `internal_error`, `queue_overflow` |
| 502 | `api_error` | `upstream_error`, `upstream_invalid_response` |
| 503 | `server_error` | `no_provider_available`, `circuit_open` |
| 504 | `api_error` | `upstream_timeout` |

---

## 2. Public endpoints

### `GET /healthz`

Liveness. No auth. Never touches Redis or Postgres.

```json
{ "status": "ok", "version": "0.1.0", "uptime_s": 1284 }
```

---

### `GET /v1/models`

Returns the **logical** model catalog plus the upstream that currently serves each one — this is
what makes the router visible to clients.

```json
{
  "object": "list",
  "data": [
    {
      "id": "fast",
      "object": "model",
      "owned_by": "gateway",
      "created": 1767139200,
      "gateway": {
        "upstream": "ollama",
        "model": "llama3.2:1b",
        "tier": "cheap",
        "context_window": 131072,
        "available": true
      }
    },
    { "id": "smart",  "gateway": { "upstream": "ollama", "model": "llama3.2:3b",      "tier": "mid" } },
    { "id": "coder",  "gateway": { "upstream": "ollama", "model": "qwen2.5-coder:3b", "tier": "specialist" } },
    { "id": "auto",   "gateway": { "upstream": "router", "model": "resolved-at-runtime", "tier": "router" } }
  ]
}
```

Clients request `model: "auto"` and the gateway decides. That is the router's public face.

---

### `POST /v1/chat/completions`

The hot path.

**Request**

```json
{
  "model": "auto",
  "messages": [
    { "role": "system", "content": "You are a terse assistant." },
    { "role": "user",   "content": "Write a regex to match an ISO date." }
  ],
  "stream": true,
  "temperature": 0.2,
  "max_tokens": 1024
}
```

**Supported:** `model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `stop`,
`tools`, `tool_choice`, `response_format`, `user`.

**Rejected (400, explicitly):**

| Parameter | Reason |
|---|---|
| `n > 1` | most upstreams don't support it; would break cache and cost accounting |
| `logprobs: true` | not preserved through caching |
| `presence_penalty` / `frequency_penalty` | incompatible with cached responses |

**Non-streaming response** (standard OpenAI shape):

```json
{
  "id": "chatcmpl-3f2a9c1e",
  "object": "chat.completion",
  "created": 1767139200,
  "model": "auto",
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "…" },
      "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 41, "completion_tokens": 96, "total_tokens": 137 },
  "gateway": {
    "cache": "MISS",
    "similarity": null,
    "upstream_model": "qwen2.5-coder:3b",
    "route_reason": "rule:code_block",
    "route_confidence": 1.0,
    "latency_ms": 18,
    "cost_usd": 0.0,
    "cost_basis": "local_inference"
  }
}
```

**Streaming:** standard SSE events (`data: {…}\n\n`, terminated by `data: [DONE]`).

- If served from cache: **one** `data:` chunk containing the full content, then `[DONE]`.
  Documented trade-off — TTFT differs from a real stream.
- If `stream_options.include_usage = true`: a final chunk with `usage` before `[DONE]`.
  **The gateway also sets this on upstream calls**, because usage is otherwise unavailable on
  streams and cost accounting silently becomes $0 (bug #1 from the design notes).

**Cache-bypass rules** (the gateway will set `x-gateway-cache: BYPASS`):

- `temperature > 0.2` → sampling makes "correct" non-unique
- `stream: false` with `n > 1` → already rejected
- `messages` contain a `tool` result not seen before → tool state differs
- request explicitly sets `gateway.no_cache: true` (extension parameter)

---

### `POST /v1/embeddings`

Standard OpenAI shape. Also used internally to feed the semantic cache.

```json
{ "model": "text-embedding-nomic", "input": ["…", "…"] }
```

```json
{
  "object": "list",
  "data": [ { "object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "…"] } ],
  "model": "text-embedding-nomic",
  "usage": { "prompt_tokens": 18, "total_tokens": 18 }
}
```

---

### `GET /v1/usage`

Usage for the calling key. Requires API key auth.

```
GET /v1/usage?window=24h
```

```json
{
  "api_key_id": 3,
  "window": "24h",
  "requests": 1842,
  "cache": { "exact_hits": 611, "semantic_hits": 214, "misses": 1017, "hit_rate": 0.448 },
  "tokens": { "prompt": 402118, "completion": 91340, "from_cache": 244903 },
  "cost_usd": 0.0,
  "cost_basis": "local_inference",
  "latency_ms": { "p50": 14, "p95": 41, "p99": 120 },
  "top_models": [ { "model": "auto", "requests": 1842, "cache_hit_rate": 0.448 } ]
}
```

---

## 3. Admin endpoints (dashboard only)

All require `Authorization: Bearer <ADMIN_TOKEN>`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/api-keys` | create key → **plaintext returned once** |
| `GET` | `/admin/api-keys` | list keys (id, label, prefix, rate limits, last used) |
| `DELETE` | `/admin/api-keys/{id}` | revoke |
| `PATCH` | `/admin/api-keys/{id}` | update label / limits |
| `GET` | `/admin/stats?window=1h\|24h\|7d` | overview: hit rate, cost, p95, volume |
| `GET` | `/admin/stats/timeseries?window=24h&bucket=5m` | chart data |
| `GET` | `/admin/cache/entries?limit=50&sort=hits` | top cached prompts |
| `DELETE` | `/admin/cache?scope=exact\|semantic\|all` | clear |
| `POST` | `/admin/cache/threshold` | set semantic threshold at runtime |
| `GET` | `/admin/cache/threshold` | current threshold + last sweep result |
| `GET` | `/admin/router/decisions?limit=50` | recent routing decisions with reasons |
| `POST` | `/admin/router/score` | dry-run: "what would you route this prompt to?" |
| `POST` | `/admin/router/toggle` | switch between `rules` / `learned` / `off` |
| `GET` | `/admin/providers` | upstream health, circuit-breaker state |
| `POST` | `/admin/providers/{name}/reset` | close an open circuit manually |

**Create key**

```json
POST /admin/api-keys
{ "label": "my-app", "rate_limit_tokens_per_min": 100000, "allowed_models": ["auto", "fast"] }
```
```json
{
  "id": 7,
  "label": "my-app",
  "key": "sk-gw-9f3a…",          // shown once, never retrievable again
  "prefix": "sk-gw-9f3a",
  "rate_limit_tokens_per_min": 100000,
  "allowed_models": ["auto", "fast"],
  "created_at": "2026-11-13T09:14:02Z"
}
```

**Stats** (drives the dashboard)

```json
GET /admin/stats?window=24h
{
  "window": "24h",
  "requests": 12844,
  "cache": { "hit_rate": 0.462, "exact": 4011, "semantic": 1921, "miss": 6912 },
  "tokens": { "total": 3184221, "served_from_cache": 1471112 },
  "saved":  {
    "tokens": 1471112,
    "inference_seconds": 412.5,
    "modeled_usd": 0.84,
    "basis": "modeled_from_hosted_equivalent"
  },
  "latency": { "p50": 13, "p95": 38, "p99": 111, "hit_p95": 7, "miss_p95": 142 },
  "overhead_ms": { "p50": 1.2, "p95": 4.8 },
  "by_model": [
    { "model": "auto", "requests": 9120, "hit_rate": 0.51, "tokens": 2100441, "p95": 36 },
    { "model": "fast", "requests": 3724, "hit_rate": 0.34, "tokens": 1083780, "p95": 21 }
  ],
  "rate_limit": { "rejected": 118, "rejections_by_key": [ { "api_key_id": 7, "count": 96 } ] }
}
```

> Note `saved.basis` — every dollar-shaped number carries its provenance. See
> [`architecture.md` §1](./architecture.md#1-why-this-exists-and-how-it-will-be-framed).

---

## 4. Rate limiting contract

**Denominated in tokens, not requests.** Three buckets, strictest wins:

| Bucket | Key |
|---|---|
| per-user | `gw:rl:user:{user_id}` |
| per-api-key | `gw:rl:key:{api_key_id}` |
| global | `gw:rl:global` |

**Cost model**

```
cost = max(1, ceil(prompt_tokens * W_PROMPT + completion_tokens * W_COMPLETION))
```

**Pre-charge / reconcile** (required for streaming):

1. Before the upstream call, reserve `max_tokens` (or a conservative default when unset).
2. If the reservation fails → 429 immediately, with `retry-after`.
3. After the stream ends, refund `reserved − actually_used`.
4. If the stream aborts mid-way, refund what wasn't generated.

Without pre-charging, concurrent streams can blow through the budget before any usage is known.

**429 body**

```json
{
  "error": {
    "message": "Rate limit exceeded for api key (tokens). 100000 tokens/min, retry in 12s.",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  },
  "request_id": "…"
}
```

---

## 5. OpenAI compatibility notes

| Client | Works? | Note |
|---|---|---|
| `openai` Python / JS SDK | ✅ | point `base_url` at the gateway |
| LangChain `ChatOpenAI` | ✅ | checks `error.type` — envelope must match |
| curl | ✅ | |
| OpenWebUI / LobeChat | ✅ | they call `/v1/models` and `/v1/chat/completions` |
| `/v1/completions` (legacy) | ❌ Day 1 | add only if a target client needs it |
| `/v1/files`, `/v1/assistants` | ❌ out of scope | |

**Not implemented, and rejected with a clear 400 rather than silently ignored** — silent
divergence from OpenAI is the fastest way to confuse a client: `n > 1`, `logprobs`,
`presence_penalty`, `frequency_penalty`.

---

## 6. SSE details (easy to get wrong)

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no        # required behind nginx, else chunks get buffered
x-gateway-cache: MISS
```

- Flush after every chunk (`await response.drain()` / Starlette's `StreamingResponse`).
- On client disconnect, **abort the upstream request** — do not keep generating; do not retry.
  Retrying after bytes reached the client duplicates content (bug #4).
- Retry only on `429` / `5xx` / timeout, with exponential backoff **plus jitter**, and only
  *before* the first byte reaches the client.
