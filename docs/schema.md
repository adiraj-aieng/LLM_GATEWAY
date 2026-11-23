# LLM Gateway — Schema

> Database tables, Redis key inventory, cache-key construction, and configuration.
>
> **Companion docs:** [`architecture.md`](./architecture.md) · [`api-contract.md`](./api-contract.md)

---

## 1. Relational schema (Postgres; SQLite for dev)

SQLAlchemy 2.0 async models live in `gateway/db/models.py`. DDL below for review.

### `users`

```sql
CREATE TABLE users (
  id            BIGSERIAL PRIMARY KEY,
  external_id   TEXT UNIQUE,               -- SSO / provisioned id, nullable
  email         TEXT,
  display_name  TEXT,
  plan          TEXT NOT NULL DEFAULT 'free',   -- free | pro | team
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at  TIMESTAMPTZ
);
```

### `api_keys`

Keys are **hashed at rest** with blake2b. Plaintext is returned exactly once, at creation.

```sql
CREATE TABLE api_keys (
  id                    BIGSERIAL PRIMARY KEY,
  user_id               BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  label                 TEXT NOT NULL,
  key_prefix            TEXT NOT NULL,        -- 'sk-gw-9f3a'  (for display + lookup)
  key_hash              TEXT NOT NULL,        -- blake2b-256, hex
  rate_limit_tokens_min INTEGER NOT NULL DEFAULT 100000,
  daily_token_budget    INTEGER,              -- NULL = unlimited
  allowed_models        TEXT[] NOT NULL DEFAULT '{auto,fast,smart,coder}',
  is_active             BOOLEAN NOT NULL DEFAULT TRUE,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at            TIMESTAMPTZ,
  last_used_at          TIMESTAMPTZ,
  UNIQUE (user_id, key_prefix)
);
CREATE INDEX idx_api_keys_hash  ON api_keys (key_hash) WHERE is_active;
CREATE INDEX idx_api_keys_user  ON api_keys (user_id);
```

> ⚠️ The lookup is `WHERE key_hash = $1 AND is_active` — never scan on `key_prefix` alone.

### `request_logs`

Append-only, high volume. Written by the async batch writer, never by the request path.

```sql
CREATE TABLE request_logs (
  id                 BIGSERIAL PRIMARY KEY,
  request_id         UUID NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- identity
  user_id            BIGINT REFERENCES users(id),
  api_key_id         BIGINT REFERENCES api_keys(id),

  -- what was asked for vs. what served it
  model_requested    TEXT NOT NULL,        -- 'auto'
  model_served       TEXT NOT NULL,        -- 'qwen2.5-coder:3b'
  upstream_provider  TEXT NOT NULL,        -- 'ollama'
  route_reason       TEXT,                 -- 'rule:code_block' | 'classifier' | 'fallback' | 'budget'
  route_confidence   REAL,                 -- 0..1

  -- cache
  cache_status       TEXT NOT NULL,        -- exact_hit | semantic_hit | miss | bypass
  similarity         REAL,                 -- cosine, semantic_hit only

  -- volume
  prompt_tokens      INTEGER NOT NULL DEFAULT 0,
  completion_tokens  INTEGER NOT NULL DEFAULT 0,
  total_tokens       INTEGER NOT NULL DEFAULT 0,
  tokens_from_cache  INTEGER NOT NULL DEFAULT 0,   -- subset not re-inferred

  -- money (see provenance note below)
  cost_usd           NUMERIC(12,6) NOT NULL DEFAULT 0,
  cached_cost_usd    NUMERIC(12,6) NOT NULL DEFAULT 0,  -- cost avoided
  cost_basis         TEXT NOT NULL DEFAULT 'local_inference',
  price_version      TEXT NOT NULL DEFAULT 'v1',

  -- timing
  latency_ms         INTEGER NOT NULL,     -- total gateway-side
  ttfb_ms            INTEGER,              -- first byte to client
  upstream_latency_ms INTEGER,
  queue_wait_ms      INTEGER,              -- telemetry queue delay (debugging)

  -- outcome
  status_code        INTEGER NOT NULL,
  error_type         TEXT,                 -- OpenAI error.type, when failed
  cached             BOOLEAN NOT NULL DEFAULT FALSE,

  -- content (PII: preview is truncated + redacted before it ever reaches the queue)
  prompt_hash        TEXT NOT NULL,        -- blake2b of normalized prompt
  system_prompt_hash TEXT,
  prompt_preview     TEXT,                 -- first 200 chars, redacted
  metadata           JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_logs_time        ON request_logs (created_at DESC);
CREATE INDEX idx_logs_apikey_time ON request_logs (api_key_id, created_at DESC);
CREATE INDEX idx_logs_model_time  ON request_logs (model_served, created_at DESC);
CREATE INDEX idx_logs_cache_time  ON request_logs (cache_status, created_at DESC);
CREATE INDEX idx_logs_user_time   ON request_logs (user_id, created_at DESC);
```

**Why `price_version`:** provider prices change. Stamping each row keeps historical cost
reproducible instead of silently re-priced when the table is updated.

### `hourly_rollups`

Added on Day 2 — without it, the dashboard's p95 query scans the whole log table.

```sql
CREATE TABLE hourly_rollups (
  bucket_start       TIMESTAMPTZ NOT NULL,   -- truncated to hour, UTC
  api_key_id         BIGINT,
  model_served       TEXT,
  cache_status       TEXT,
  requests           BIGINT NOT NULL DEFAULT 0,
  tokens_total       BIGINT NOT NULL DEFAULT 0,
  tokens_from_cache  BIGINT NOT NULL DEFAULT 0,
  cost_usd           NUMERIC(12,6) NOT NULL DEFAULT 0,
  cached_cost_usd    NUMERIC(12,6) NOT NULL DEFAULT 0,
  -- latencies stored as arrays for exact percentile merge
  latencies_ms       INTEGER[] NOT NULL DEFAULT '{}',
  PRIMARY KEY (bucket_start, api_key_id, model_served, cache_status)
);
```

### `model_prices`

```sql
CREATE TABLE model_prices (
  version        TEXT NOT NULL,          -- 'v1'
  model_id       TEXT NOT NULL,          -- upstream model
  input_per_mtok  NUMERIC(10,6) NOT NULL, -- USD per 1M input tokens
  output_per_mtok NUMERIC(10,6) NOT NULL,
  effective_from TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (version, model_id)
);
```

> **Provenance:** with a local-only deployment these prices are *modeled equivalents of hosted
> models*, not real charges. `cost_basis` on every row is `local_inference`, and
> `cached_cost_usd` is reported as *modeled* savings. The dashboard labels them as such.

---

## 2. Cache keys

All key builders live in **one file**, `gateway/cache/keys.py`. Nothing else concatenates keys.

### 2.1 Normalization (applied before hashing)

```
input  →  NFKC unicode normalize
       →  collapse runs of whitespace to a single space
       →  strip leading/trailing whitespace
       →  DO NOT lowercase      (case is semantic in prompts)
       →  DO NOT strip punctuation
```

> NFKC is what makes `é` and `e\u0301` hash identically. Skipping it silently fragments the cache
> for any non-ASCII prompt — a real bug, and an easy one to ship.

### 2.2 Exact cache key

```
gw:v1:exact:{api_key_id}:{logical_model}:{blake2b_hex}
```

where the hash input is the **concatenation of**:

| # | Component | Why it must be included |
|---|---|---|
| 1 | normalized prompt (last user message) | the obvious part |
| 2 | **full message list roles+contents** | multi-turn context changes the answer |
| 3 | **system prompt hash** | *the* classic bug: two apps, different system prompts, shared cache → wrong answers |
| 4 | **`tools` / `functions` schema hash** | a cached `tool_calls` is wrong if tools differ |
| 5 | `temperature`, `top_p` (bucketed) | otherwise a greedy answer is served for `temp=1` |
| 6 | `max_tokens` **bucketed to nearest 256** | exact values fragment the cache for no benefit |
| 7 | `stop` sequences, `response_format` | change output shape |

**Excluded** (irrelevant to output): `user`, `n` (rejected anyway), `stream`, `stream_options`,
`gateway.*` extension params.

**Bypass (never cached):** `temperature > 0.2` — sampling means "correct" isn't unique.

**Value:** full response JSON + `usage` + `created_at`. **TTL:** `EXACT_CACHE_TTL_S`, default
86400. **Cap:** `EXACT_CACHE_MAX_ENTRIES`, default 10000, LRU-evicted.

### 2.3 Semantic cache

```
index set : gw:v1:sem:idx:{api_key_id}:{logical_model}          -> SET of entry ids
entry     : gw:v1:sem:doc:{entry_id}                            -> JSON payload
vector    : gw:v1:sem:vec:{entry_id}                            -> raw float32 bytes
```

Entry payload:

```json
{
  "entry_id": "b7c1…",
  "api_key_id": 3,
  "model": "auto",
  "prompt_hash": "…",
  "system_prompt_hash": "…",
  "tools_hash": "…",
  "response": { "…full OpenAI response…" },
  "usage": { "prompt_tokens": 41, "completion_tokens": 96 },
  "created_at": "2026-11-13T09:14:02Z",
  "hits": 12
}
```

**Hit criteria — all four must hold:**

1. same `api_key_id` and `logical_model`
2. same `system_prompt_hash` and `tools_hash`
3. `cosine(top1, query) ≥ SEMANTIC_SIM_THRESHOLD` (default **0.92**)
4. **margin guard:** `cosine(top1) − cosine(top2) ≥ 0.02`

Criterion 4 is the false-positive guard: without it, a query sitting between two near-tied
candidates silently gets whichever one was inserted first.

**Search:** in-process `numpy.float32` matrix, brute-force cosine, rebuilt from Redis on startup.
Correct to ~50k vectors. Above that → pgvector (documented path, not built Day 1).

> ⚠️ Plain `redis-server` has **no vector search** — that's Redis Stack / RediSearch, a separate
> module we are not installing. Any design claiming `FT.SEARCH` here would be wrong.

**TTL:** `SEMANTIC_CACHE_TTL_S`, default 604800 (7 days).

---

## 3. Redis key inventory

| Key | Type | Purpose |
|---|---|---|
| `gw:v1:exact:{key_id}:{model}:{hash}` | string | exact cache entry |
| `gw:v1:sem:idx:{key_id}:{model}` | set | semantic entry ids |
| `gw:v1:sem:doc:{id}` / `:vec:{id}` | string | semantic payload / vector |
| `gw:rl:user:{user_id}` | hash | token bucket: `tokens`, `refill_ts` |
| `gw:rl:key:{api_key_id}` | hash | token bucket |
| `gw:rl:global` | hash | token bucket |
| `gw:rl:conc:{api_key_id}` | string | in-flight counter, TTL = request timeout |
| `gw:cb:{provider}:{model}` | hash | circuit breaker: `state`, `failures`, `opened_at` |
| `gw:stats:{key_id}:{yyyymmddhh}` | hash | hourly counters for `/admin/stats` |

### Token bucket Lua script (`gateway/core/rate_limit.lua`)

```lua
-- KEYS[1] bucket key
-- ARGV[1] capacity, ARGV[2] refill_rate (tokens/sec),
-- ARGV[3] cost, ARGV[4] now_ms (fallback), ARGV[5] ttl_ms
local t   = redis.call('TIME')                       -- ★ Redis clock, not app clock
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local b   = redis.call('HMGET', KEYS[1], 'tokens', 'refill_ts')
local tokens  = tonumber(b[1])
local refill  = tonumber(b[2])

if tokens == nil then tokens = tonumber(ARGV[1]) ; refill = now end
local elapsed = math.max(0, now - refill)
tokens = math.min(tonumber(ARGV[1]), tokens + elapsed * tonumber(ARGV[2]))

local allowed = 0
local retry_after = 0
local cost = tonumber(ARGV[3])
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = math.ceil((cost - tokens) / tonumber(ARGV[2]))
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'refill_ts', now)
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[5]))
return {allowed, tokens, retry_after}
```

> Using `redis.call('TIME')` instead of the app clock matters: a skewed host clock produces
> phantom refills and lets clients blow past their limit. This is a genuine production bug class.

---

## 4. Configuration (`gateway/config.py`, pydantic-settings)

`.env.example` ships with every variable documented. Nothing secret is committed.

### Core

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `development` \| `production` \| `test` |
| `LOG_LEVEL` | `INFO` | |
| `ADMIN_TOKEN` | *(required in prod)* | separate from API keys |
| `DATABASE_URL` | `sqlite+aiosqlite:///./gateway.db` | prod: `postgresql+asyncpg://…` |
| `REDIS_URL` | `redis://localhost:6379/0` | must be `0.0.0.0`-bound inside WSL |

### Upstream

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | native Windows Ollama |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | fallback path only |
| `UPSTREAM_TIMEOUT_S` | `60` | stream reads use a separate idle timeout |
| `UPSTREAM_IDLE_TIMEOUT_S` | `30` | no chunk for 30s → abort |
| `CIRCUIT_FAILURE_THRESHOLD` | `5` | consecutive failures to open |
| `CIRCUIT_COOLDOWN_S` | `30` | half-open after this; **1 probe only** |
| `RETRY_MAX_ATTEMPTS` | `2` | retries only 429/5xx/timeout, with jitter |
| `MODEL_MAP` | `{"fast":"llama3.2:1b", …}` | logical → upstream |

### Caching

| Variable | Default | Notes |
|---|---|---|
| `EXACT_CACHE_ENABLED` | `true` | |
| `EXACT_CACHE_TTL_S` | `86400` | |
| `EXACT_CACHE_MAX_ENTRIES` | `10000` | LRU cap |
| `SEMANTIC_CACHE_ENABLED` | `true` | |
| `SEMANTIC_CACHE_TTL_S` | `604800` | 7 days |
| `SEMANTIC_SIM_THRESHOLD` | `0.92` | swept on Day 2 → `artifacts/threshold.json` |
| `SEMANTIC_MARGIN_GUARD` | `0.02` | top1 − top2 minimum |
| `SEMANTIC_MAX_VECTORS` | `50000` | above → warn, switch to pgvector |
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` |
| `CACHE_MAX_TEMP` | `0.2` | above → bypass |

### Rate limiting

| Variable | Default | Notes |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | |
| `RATE_LIMIT_CAPACITY` | `100000` | tokens |
| `RATE_LIMIT_REFILL_PER_S` | `1666` | ≈100k/min |
| `RATE_LIMIT_WEIGHT_PROMPT` | `1.0` | |
| `RATE_LIMIT_WEIGHT_COMPLETION` | `1.0` | |
| `CONCURRENCY_LIMIT_PER_KEY` | `8` | in-flight streams |

### Telemetry

| Variable | Default | Notes |
|---|---|---|
| `LOG_QUEUE_MAXSIZE` | `10000` | drop-on-full, counted — never blocks |
| `LOG_FLUSH_INTERVAL_S` | `1.0` | |
| `LOG_FLUSH_BATCH` | `500` | |
| `PROMPT_PREVIEW_LEN` | `200` | truncated + PII-redacted |

---

## 5. PII handling

`prompt_preview` is the only place prompt text touches the database. Before it enters the queue:

1. truncate to `PROMPT_PREVIEW_LEN`
2. redact patterns: email, phone, credit-card-like, API-key-like, IP addresses
3. store only `prompt_hash` for correlation

Raw prompts are **never** persisted. `metadata` must not be used to smuggle them in either —
there's a test for that.

---

## 6. Local dev vs. production

| | Dev | Production |
|---|---|---|
| DB | SQLite (`+aiosqlite`) | Postgres (`+asyncpg`) |
| Redis | WSL2 `redis-server` bound `0.0.0.0` | managed Redis |
| Vector index | in-process numpy | pgvector when > 50k |
| Tests | `fakeredis` + in-memory SQLite | integration job in CI |
| Telemetry flush | every 1s | every 1s + `flush_on_shutdown` |

**Compatibility caveats when writing SQL:** `JSONB` → `JSON` on SQLite; `TIMESTAMPTZ` → `DATETIME`;
`BIGSERIAL` → `INTEGER PRIMARY KEY AUTOINCREMENT`; `INTEGER[]` (rollup latencies) has no SQLite
equivalent — rollups are Postgres-only, which is why they're a Day 2 item.
