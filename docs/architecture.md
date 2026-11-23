# LLM Gateway — Architecture

> An OpenAI-compatible proxy that sits between your apps and any LLM provider, adding a
> two-stage cache, token-based rate limiting, a learned model router, and cost/latency telemetry.

**Status:** design — pending review before implementation.
**Companion docs:** [`api-contract.md`](./api-contract.md) · [`schema.md`](./schema.md)

---

## 1. Why this exists, and how it will be framed

LiteLLM, Portkey, Helicone, Cloudflare AI Gateway, Kong AI Gateway and OpenRouter already do this.
The idea is not the differentiator — **the measurements are.** This project is framed as:

> "I built an LLM gateway from scratch and measured the trade-offs."

So every subsystem below is designed to be *measurable*, not just functional:

| Subsystem | The number it produces |
|---|---|
| Semantic cache | hit rate vs. false-positive rate across similarity thresholds |
| Router | cost vs. quality curve at each operating threshold |
| Rate limiter | throughput under token-based (not request-based) pressure |
| Pipeline | p50/p95/p99 gateway overhead at N req/s |
| Telemetry | $ / tokens / latency per model, per key, per user |

### Honesty constraint (important)

Deployment is **local-only** (Ollama + RTX 4050). Local inference has **no marginal dollar cost**,
so we will not claim "cut cost by 40%" as a measured fact. Metrics are reported in three tiers:

1. **Measured** (headline): cache hit rate, tokens served from cache, p50/p95 latency, req/s, overhead.
2. **Measured → converted**: upstream inference-seconds avoided × published GPU hourly rate.
3. **Modeled** (explicitly labeled): local model → equivalent hosted model price table.

The README will state plainly which numbers are which. That honesty is the stronger answer.

---

## 2. Target hardware & process topology

Measured on the dev machine:

| | Reality | Consequence for the design |
|---|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **4 GB VRAM** | Only small models fit fully on GPU. `llama3.1:8b` (~4.7 GB) spills to CPU → avoid. |
| RAM | 15.3 GB Windows / **WSL2 capped at 7.4 GB** | Cannot run everything inside WSL. |
| CPU | Ryzen 9 8945HS, 8C / 16T | Plenty for CPU fallback and load testing. |
| Ollama | v0.40.16 installed natively, **no models pulled** | Must pull before Day 1 testing. |
| Docker | **not installed** | Dockerfiles are for CI/deploy only; nothing containerized runs locally. |
| WSL | Ubuntu 24.04.5, Python 3.12.3, no Redis/Postgres | `apt install` required. |

### Chosen topology

| Component | Runs where | Why |
|---|---|---|
| **Ollama** | **Native Windows** | Gets the RTX 4050. Running Ollama inside WSL would contend for the same 4 GB VRAM. |
| **Redis + Postgres** | **WSL2 Ubuntu** | Must bind `0.0.0.0` so Windows reaches them over `localhost`. Redis defaults to `127.0.0.1`, which **silently fails** across the WSL2 boundary. |
| **FastAPI backend** | **Windows** | Full 15.3 GB RAM, existing pyenv 3.13, existing editor tooling. |
| **React dashboard** | **Windows** | Vite dev server; no dependency on Redis. |
| **Unit tests** | `fakeredis` + in-memory SQLite | Hermetic and fast. Real Redis/Postgres reserved for integration tests. |

> ⚠️ `wsl --shutdown` takes Redis and Postgres down with it. Ship a `make infra` helper.

---

## 3. Model tiering — the router's raw material

With 4 GB VRAM, one model at a time is comfortably resident; two small models plus embeddings is
feasible on the Windows side. This yields a natural three-tier ladder:

| Tier | Model | Size | Role |
|---|---|---|---|
| cheap | `llama3.2:1b` | ~1.3 GB | lookup / simple prompts |
| mid | `llama3.2:3b` | ~2.0 GB | default; fits VRAM fully |
| specialist | `qwen2.5-coder:3b` | ~1.9 GB | code → route here |
| embed | `nomic-embed-text` | ~274 MB | semantic cache (optional path, see §6) |

This gives the router a *real* decision — "is this a coding task?" — with observable latency and
quality differences, rather than an artificial choice between two identical models.

**Ollama runtime settings (must be documented, they change p95):**

```
OLLAMA_KEEP_ALIVE=10m      # avoids cold-start skew during load tests
OLLAMA_NUM_PARALLEL=1      # stable latency profile; raise only if we want throughput numbers
```

---

## 4. Request lifecycle

```
Client (OpenAI SDK / LangChain / curl)
   │  POST /v1/chat/completions   Authorization: Bearer sk-gw-...
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. AUTH          hash key → principal (user, api_key, plan)     │
│ 2. RATE LIMIT    Redis Lua token bucket (atomic refill+consume) │
│                  pre-charge max_tokens, reconcile after stream  │
│ 3. EXACT CACHE   key = hash(norm_prompt + system + tools + sampling) │
│    │ miss                                                       │
│ 4. SEMANTIC CACHE embed → cosine → threshold AND margin guard   │
│    │ miss                                                       │
│ 5. ROUTER        rules → learned classifier → RouteDecision     │
│ 6. PROVIDER      OpenAI-compatible adapter (Ollama base_url)    │
│                  retry/backoff + circuit breaker + fallback     │
│ 7. STREAMING     SSE parse → re-emit to client                  │
│ 8. RECONCILE     actual tokens → refund unused pre-charge       │
│ 9. LOG           asyncio.Queue → batch write → Postgres         │
└─────────────────────────────────────────────────────────────────┘
   ▼
Postgres request_logs  ──►  React dashboard (hit rate, $, p95, top prompts)
```

Steps 1–4 must happen **before** any upstream call. Step 9 must **never block** the hot path.

---

## 5. Full file tree

Every file has exactly one responsibility. Files are grouped so that a reviewer can read one
directory and understand one subsystem.

```
llm-gateway/
├─ README.md                    architecture diagram, benchmark charts, run instructions
├─ docker-compose.yml           redis + postgres + api + dashboard  (CI / deploy only)
├─ Dockerfile.api
├─ Dockerfile.dashboard
├─ Makefile                     make infra | dev | test | bench | train | seed
├─ .env.example                 every env var, documented, no secrets
├─ .gitignore
├─ .wslconfig                   (Windows-side WSL memory note, optional)
├─ .github/workflows/ci.yml     lint + pytest + nightly load-test artifact upload
│
├─ gateway/
│  ├─ __init__.py
│  ├─ main.py                   app factory; lifespan = open/close pools, start queue writer
│  ├─ config.py                 pydantic-settings Settings; validates on boot, fails fast
│  ├─ logging.py                structlog JSON; request_id propagated via contextvar
│  │
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ deps.py                get_redis, get_db, require_api_key, get_principal
│  │  ├─ errors.py              OpenAI-shaped error envelope; exception handlers
│  │  └─ v1/
│  │     ├─ __init__.py
│  │     ├─ chat.py             ★ POST /v1/chat/completions  (stream + non-stream)
│  │     ├─ embeddings.py       POST /v1/embeddings  (also feeds the semantic cache)
│  │     ├─ models.py           GET  /v1/models  (logical model → upstream mapping)
│  │     └─ admin.py            /admin/* : keys, stats, cache, router inspection
│  │
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ auth.py                key create/verify (blake2b at rest), Principal resolution
│  │  ├─ rate_limit.py          token bucket; wraps the Lua script; pre-charge/reconcile
│  │  ├─ rate_limit.lua         ★ atomic refill + consume, uses Redis TIME (not app clock)
│  │  ├─ normalize.py           NFKC + whitespace collapse for cache keys
│  │  └─ hashing.py             blake2b helpers (prompt, system, tools)
│  │
│  ├─ cache/
│  │  ├─ __init__.py
│  │  ├─ keys.py                ★ ALL cache key builders — single source of truth
│  │  ├─ exact.py               get/set, TTL, LRU cap, bypass rules
│  │  ├─ semantic.py            embed → search → threshold + margin guard → store
│  │  └─ vector_index.py        numpy float32 matrix, cosine, in-process + Redis persistence
│  │
│  ├─ router/
│  │  ├─ __init__.py
│  │  ├─ base.py                Router ABC → RouteDecision(model, reason, confidence)
│  │  ├─ rules.py               length / code-block / injection / budget rules
│  │  ├─ learned.py             loads classifier → P(needs_strong)
│  │  ├─ policy.py              merge rules + learned, hysteresis, budget override
│  │  └─ explain.py             feature contributions for dashboard "why" panel
│  │
│  ├─ providers/
│  │  ├─ __init__.py
│  │  ├─ base.py                Provider ABC: chat() async generator + capabilities
│  │  ├─ registry.py            provider/model catalog, aliasing, availability probing
│  │  ├─ openai_compat.py       ★ covers Ollama, OpenAI, Groq, Together (same wire format)
│  │  ├─ anthropic.py           different wire format (Day 2, optional)
│  │  ├─ gemini.py              different wire format (Day 2, optional)
│  │  └─ resilience.py          retry w/ jitter, circuit breaker, fallback chain
│  │
│  ├─ pipeline/
│  │  ├─ __init__.py
│  │  ├─ request_pipeline.py    ★ orchestrates steps 1–9; the only place that knows order
│  │  ├─ streaming.py           SSE parse/re-emit, chunk buffering, usage extraction
│  │  └─ cost.py                token counting, versioned price table, $ saved
│  │
│  ├─ db/
│  │  ├─ __init__.py
│  │  ├─ base.py                DeclarativeBase
│  │  ├─ session.py             async engine + sessionmaker + get_session dependency
│  │  ├─ models.py              User, ApiKey, RequestLog, ModelPrice, HourlyRollup
│  │  └─ repositories.py        RequestLogRepo, ApiKeyRepo (keeps routes thin)
│  │
│  └─ telemetry/
│     ├─ __init__.py
│     ├─ queue.py               asyncio.Queue + drop-on-full counter (never blocks)
│     ├─ writer.py              batch flush on interval/size; flush_on_shutdown
│     └─ metrics.py             in-memory counters + histograms (or Prometheus)
│
├─ ml/
│  ├─ README.md                 how the router was trained + eval numbers
│  ├─ data/
│  │  ├─ build_dataset.py       heuristically label prompts → needs_strong
│  │  ├─ labeled_pairs.csv      ★ hand-labeled 500-prompt subset (the honest part)
│  │  └─ eval_set.csv           ★ held out; never touched during training
│  ├─ train_router.py           → artifacts/router_model.pkl + metrics.json
│  ├─ eval_cache_thresholds.py  → artifacts/threshold.json + hit-rate/error-rate curve
│  ├─ evaluate_router.py        → cost-vs-quality curve
│  └─ artifacts/                model, metrics, thresholds (small, committed)
│
├─ loadtest/
│  ├─ locustfile.py             cache-hit vs cache-miss scenarios
│  └─ results/                  p50/p95/p99 CSVs (gitignored, charts committed)
│
├─ tests/
│  ├─ conftest.py               fixtures: app, fakeredis, temp sqlite, FakeProvider
│  ├─ test_auth.py
│  ├─ test_rate_limit.py        ★ refill semantics, pre-charge, concurrency
│  ├─ test_exact_cache.py       ★ normalization + system/tools key inclusion
│  ├─ test_semantic_cache.py    ★ threshold, margin guard, false-positive cases
│  ├─ test_streaming.py         ★ usage extraction, partial-stream abort
│  ├─ test_router.py
│  └─ test_cost_accounting.py
│
├─ dashboard/                   React + Vite + TypeScript
│  ├─ package.json / vite.config.ts / tsconfig.json / index.html
│  └─ src/
│     ├─ main.tsx / App.tsx / router.tsx
│     ├─ api/client.ts          typed fetch wrapper, auth header injection
│     ├─ api/types.ts           TS types mirroring the Pydantic schemas
│     ├─ pages/  Overview · Cost · Cache · Latency · Prompts
│     └─ components/  Chart · StatCard · Table · TimeRangePicker
│
└─ docs/
   ├─ architecture.md           ← this file
   ├─ api-contract.md           ← endpoint-by-endpoint contract
   ├─ schema.md                 ← DB, cache keys, config
   ├─ benchmark.md              ← Day 2: the numbers
   └─ decisions.md              ← ADRs
```

---

## 6. Technology choices (ADR summary)

| Decision | Choice | Rationale | Alternative rejected |
|---|---|---|---|
| Language | **Python 3.13** | Matches your existing stack; FastAPI + async is the natural fit | Go — faster, but you'd restart your learning curve |
| Web framework | **FastAPI + uvicorn[standard]** | Native SSE streaming, Pydantic validation, async | Flask (no native async) |
| HTTP client | **httpx.AsyncClient, created once at startup** | Connection reuse; per-request clients destroy p99 | requests (sync, no streaming) |
| Redis client | **redis-py asyncio** | Lua `EVALSHA` for atomic token bucket | — |
| Vector search | **numpy float32, in-process cosine** | Plain `redis-server` has **no vector search** (that's Redis Stack/RediSearch). Numpy is correct to ~50k vectors. | `FT.SEARCH` — needs a module we don't have |
| Scale path | **pgvector** (documented, not built Day 1) | Postgres already present | — |
| Embeddings | **fastembed (ONNX, in-process)** — primary; Ollama `nomic-embed-text` — fallback | In-process avoids a network hop on the latency-critical cache path; **no torch dependency** (torch is ~2.5 GB) | sentence-transformers — 15× memory for no benefit here |
| DB | **Postgres (prod) / SQLite (dev)** via SQLAlchemy 2.0 async | Async logging + `jsonb` in prod; zero-install locally | — |
| Auth | **API keys, blake2b at rest, shown once** | Stateless, matches OpenAI conventions | JWT — unnecessary for machine-to-machine keys |
| Rate limiting | **Redis Lua token bucket, token-denominated** | Atomic; counts tokens not requests, as specified | sliding window — more complex, no benefit here |
| Telemetry | **in-process queue → batch writer** | Never blocks the hot path; drop-on-full is explicit | sync insert per request — adds latency to every call |
| Frontend | **React + Vite + TS** | Per your spec | Angular (reserved for the other project) |
| Load test | **Locust** (Python, reuse stack) | Scenario scripting is easier than k6 for this | k6 |

---

## 7. Known trade-offs (deliberate, documented)

1. **Streaming + exact cache.** Replaying a cached response as one chunk changes TTFT vs. the
   original stream. *Decision:* cache non-streaming fully; for `stream: true`, store the assembled
   text and replay as a single chunk. Documented in `decisions.md`.
2. **In-process vector index.** Lost on restart unless persisted to Redis (we do persist vectors;
   the matrix rebuilds on startup). At >50k vectors we move to pgvector.
3. **Semantic cache and `temperature > 0`.** Sampling means the "correct" answer isn't unique.
   *Decision:* bypass the semantic cache when `temperature > 0.2`.
4. **Local-only cost.** See §1 honesty constraint.
5. **WSL2 dependency.** Redis/Postgres die on `wsl --shutdown` — mitigated by `make infra`.

---

## 8. Scope

### Day 1 — working vertical slice (8–10 focused hours)

- [ ] Ollama models pulled; Redis + Postgres running in WSL
- [ ] OpenAI-compatible `/v1/chat/completions` (stream + non-stream), `/v1/models`
- [ ] API-key auth (hashed at rest)
- [ ] Token-bucket rate limit (per-user + per-key, pre-charge `max_tokens`)
- [ ] Exact cache with correct normalization + system-prompt and `tools` hashes
- [ ] Semantic cache (numpy cosine, threshold **+ margin guard**)
- [ ] Ollama adapter via OpenAI-compatible wire format, retry + circuit breaker
- [ ] Async batch logging to Postgres
- [ ] React dashboard: 4 stat cards + 2 charts
- [ ] pytest suite + GitHub Actions

### Day 2 — the numbers (4–6 hours)

- [ ] Cache threshold sweep → hit-rate vs. error-rate curve
- [ ] Router: rules → trained classifier, cost/quality curve
- [ ] Locust load test → p50/p95/p99 + gateway overhead
- [ ] Benchmark charts + ADRs into README

**Honest estimate: 1.5–2 days total, not 1.** Day 1 gets a demo; Day 2 gets the story.

---

## 9. Measurement plan

| Claim | How we prove it | Tool |
|---|---|---|
| "< X ms p95 overhead" | load test with cache disabled vs. enabled, difference in p95 | Locust |
| "hit rate Y% at threshold T" | sweep T ∈ [0.80, 0.97] on `eval_set.csv` | `eval_cache_thresholds.py` |
| "false-positive rate Z%" | hand-labeled pairs: did the cached answer fit the new prompt? | manual pass on 200 pairs |
| "router saves N% at <3% quality drop" | cost/quality curve over held-out eval set | `evaluate_router.py` |
| "token-based limiting holds at R req/s" | Locust with token-denominated budgets | Locust |
