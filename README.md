# LLM Gateway

An OpenAI-compatible gateway that sits between your applications and any LLM provider,
adding a **two-stage cache** (exact → semantic), **token-based rate limiting** (atomic,
in Redis Lua), a **rules → learned model router**, and **async cost/latency telemetry**
to a React dashboard.

> **Framing:** the idea is not the differentiator — LiteLLM, Portkey, Helicone,
> Cloudflare AI Gateway and OpenRouter all exist. The point of this project is that
> every subsystem here is *measured*, and that the numbers are reported honestly.

## Status

| | |
|---|---|
| Day 1 — vertical slice | ✅ gateway, caches, rate limiter, router, telemetry, dashboard, tests, CI |
| Day 2 — the numbers | ⏳ threshold sweep, router training, load test → `docs/benchmark.md` |

## Architecture

```
Client (OpenAI SDK / LangChain / curl)
   │  POST /v1/chat/completions   Authorization: Bearer sk-gw-…
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. AUTH          peppered blake2b key lookup → principal        │
│ 2. RATE LIMIT    Redis Lua token bucket, 3 scopes, all-or-      │
│                  nothing; pre-charge max_tokens, refund later   │
│ 3. EXACT CACHE   hash(messages + system + tools + sampling)     │
│    │ miss                                                       │
│ 4. SEMANTIC CACHE embed → cosine → threshold AND margin guard   │
│    │ miss                                                       │
│ 5. ROUTER        rules → (Day 2: classifier) → RouteDecision    │
│ 6. PROVIDER      OpenAI-compatible adapter (Ollama), retry +    │
│                  jitter + circuit breaker                       │
│ 7. STREAMING     SSE parse → re-emit; usage always captured     │
│ 8. RECONCILE     refund unused rate-limit reservation           │
│ 9. LOG           bounded queue → batch writer → SQLite/Postgres │
└─────────────────────────────────────────────────────────────────┘
   ▼
request_logs  ──►  /admin/stats  ──►  React dashboard
```

Full detail: [`docs/architecture.md`](docs/architecture.md) ·
[`docs/api-contract.md`](docs/api-contract.md) · [`docs/schema.md`](docs/schema.md)

## Honesty about cost

This deployment runs **locally** (Ollama on an RTX 4050). Local inference has **no
marginal dollar cost**, so the gateway reports cost in labelled tiers:

| Tier | Field | Meaning |
|---|---|---|
| Measured | `cost_usd` | Real spend — **`0.0`**, because it is |
| Measured | tokens, hit rate, p50/p95 latency, req/s | The headline numbers |
| Modelled | `modeled_usd` + `basis: local_inference` | Hosted-equivalent estimate, versioned per row (`price_version`) |

Every dollar-shaped number carries its `basis`. No flat "cut cost 40%" claim appears
anywhere in this repo.

## Quickstart (Windows dev topology)

Prereqs: Python 3.13, Node 22, Ollama, Redis (in WSL — see below).

```powershell
# 1. Python deps
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt

# 2. Redis inside WSL (once) — must bind 0.0.0.0 or Windows can't reach it
wsl -d Ubuntu -- sudo apt-get install -y redis-server
wsl -d Ubuntu -- sudo sed -i 's/^bind 127.0.0.1 ::1/bind 0.0.0.0/' /etc/redis/redis.conf
wsl -d Ubuntu -- sudo service redis-server start

# 3. Ollama (native Windows, for GPU access)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 4. Configure + run
copy .env.example .env       # edit ADMIN_TOKEN, API_KEY_PEPPER
.\.venv\Scripts\uvicorn gateway.main:app --reload --port 8000

# 5. Dashboard (second terminal)
cd dashboard && npm install && npm run dev   # http://localhost:5173
```

Create a key and make your first call:

```powershell
curl -X POST localhost:8000/admin/api-keys `
  -H "Authorization: Bearer <ADMIN_TOKEN>" -H "Content-Type: application/json" `
  -d '{\"label\":\"demo\",\"email\":\"me@example.com\"}'

curl localhost:8000/v1/chat/completions `
  -H "Authorization: Bearer sk-gw-…" -H "Content-Type: application/json" `
  -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'
```

Second identical call returns `x-gateway-cache: EXACT_HIT`; a paraphrase returns
`SEMANTIC_HIT` with `x-gateway-similarity`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Hermetic: `fakeredis` (with Lua, so the real token-bucket script executes), in-memory
SQLite, fake provider, deterministic fake embedder. No Redis, no Ollama, no model
downloads — CI runs the same suite on a cold machine (`.github/workflows/ci.yml`).

## What's deliberately rejected with a 400

`n > 1`, `logprobs`, `presence_penalty`, `frequency_penalty` — silent divergence from
the OpenAI contract is how clients end up debugging the proxy instead of their code.

## Layout

```
gateway/     app: api, core (auth/rate-limit/normalize), cache, router,
             providers, pipeline (orchestration), db, telemetry
ml/          Day 2: dataset, training, threshold sweep, artifacts
loadtest/    Locust scenarios → p50/p95/p99 + gateway overhead
dashboard/   React + Vite + TS (4 stat cards, 2 charts, top prompts)
tests/       pytest suite (auth, caches, streaming, router, rate limit, cost)
docs/        architecture · api contract · schema · benchmark (Day 2) · decisions
```
