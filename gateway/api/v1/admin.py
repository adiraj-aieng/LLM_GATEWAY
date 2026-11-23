"""``/admin/*`` — the dashboard's API surface.

Gated by a single ``ADMIN_TOKEN`` rather than by API keys: these endpoints expose other
people's usage data and can mutate gateway state (clearing the cache, changing the
semantic threshold, revoking keys).

Every dollar-shaped number returned here carries its ``basis`` field. See
``gateway/pipeline/cost.py`` for why.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api import deps
from gateway.core.auth import generate_api_key, hash_api_key, key_prefix_of
from gateway.db.models import ApiKey, RequestLog, User
from gateway.db.session import get_session

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(deps.require_admin)])


# =======================================================================
# schemas
# =======================================================================
class CreateKeyBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1, max_length=128)
    user_id: int | None = None
    email: str | None = None
    rate_limit_tokens_per_min: int = Field(default=100000, gt=0)
    allowed_models: list[str] = Field(default_factory=lambda: ["auto", "fast", "smart", "coder"])
    expires_in_days: int | None = Field(default=None, gt=0)


class ThresholdBody(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


class RouterModeBody(BaseModel):
    mode: Literal["rules", "learned"]


# =======================================================================
# api keys
# =======================================================================
@router.post("/api-keys")
async def create_api_key(
    body: CreateKeyBody,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
):
    """Issue a key. The plaintext is returned **once** and is unrecoverable afterwards."""
    user: User | None = None
    if body.user_id is not None:
        user = await db.get(User, body.user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Unknown user_id")
    else:
        if not body.email:
            raise HTTPException(
                status_code=400,
                detail="Provide either user_id or email to attach the key to a user",
            )
        result = await db.execute(select(User).where(User.email == body.email))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(email=body.email, display_name=body.email.split("@")[0])
            db.add(user)
            await db.flush()

    plaintext = generate_api_key()
    key = ApiKey(
        user_id=user.id,
        label=body.label,
        key_prefix=key_prefix_of(plaintext),
        key_hash=hash_api_key(plaintext, request.app.state.settings.api_key_pepper),
        rate_limit_tokens_min=body.rate_limit_tokens_per_min,
        allowed_models=body.allowed_models,
        expires_at=(
            datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
            if body.expires_in_days
            else None
        ),
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)

    return {
        "id": key.id,
        "label": key.label,
        "key": plaintext,  # shown once
        "prefix": key.key_prefix,
        "rate_limit_tokens_per_min": key.rate_limit_tokens_min,
        "allowed_models": key.allowed_models,
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "created_at": key.created_at.isoformat(),
        "user_id": user.id,
    }


@router.get("/api-keys")
async def list_api_keys(db: Annotated[AsyncSession, Depends(get_session)]):
    rows = (await db.execute(select(ApiKey).order_by(desc(ApiKey.created_at)))).scalars().all()
    return {
        "items": [
            {
                "id": k.id,
                "label": k.label,
                "prefix": k.key_prefix,
                "user_id": k.user_id,
                "is_active": k.is_active,
                "rate_limit_tokens_per_min": k.rate_limit_tokens_min,
                "allowed_models": k.allowed_models,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "created_at": k.created_at.isoformat(),
            }
            for k in rows
        ]
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
):
    key = await db.get(ApiKey, key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="Unknown API key")
    key.is_active = False
    await db.commit()
    # Revocation should take effect immediately, not on next bucket refill.
    try:
        await request.app.state.ratelimit.set_capacity(key_id, 1)
    except Exception:
        pass  # a Redis hiccup must not fail an otherwise-committed revocation
    return {"id": key_id, "is_active": False}


# =======================================================================
# stats
# =======================================================================
_WINDOWS = {"1h": 1, "24h": 24, "7d": 24 * 7}


@router.get("/stats")
async def stats(
    db: Annotated[AsyncSession, Depends(get_session)],
    window: Literal["1h", "24h", "7d"] = Query(default="24h"),
):
    since = datetime.now(timezone.utc) - timedelta(hours=_WINDOWS[window])

    base = select(RequestLog).where(RequestLog.created_at >= since)
    rows = (await db.execute(base)).scalars().all()

    total = len(rows)
    by_cache: dict[str, int] = {}
    tokens_total = 0
    tokens_from_cache = 0
    latencies: list[int] = []
    hit_latencies: list[int] = []
    miss_latencies: list[int] = []
    by_model: dict[str, dict[str, Any]] = {}
    by_key: dict[str, int] = {}
    rejections = 0

    for r in rows:
        by_cache[r.cache_status] = by_cache.get(r.cache_status, 0) + 1
        tokens_total += int(r.total_tokens or 0)
        tokens_from_cache += int(r.tokens_from_cache or 0)
        latencies.append(int(r.latency_ms or 0))
        if r.cache_status in ("exact_hit", "semantic_hit"):
            hit_latencies.append(int(r.latency_ms or 0))
        elif r.cache_status == "miss":
            miss_latencies.append(int(r.latency_ms or 0))
        if r.status_code == 429:
            rejections += 1

        m = by_model.setdefault(
            r.model_served, {"requests": 0, "tokens": 0, "latencies": [], "cache": {}}
        )
        m["requests"] += 1
        m["tokens"] += int(r.total_tokens or 0)
        m["latencies"].append(int(r.latency_ms or 0))
        m["cache"][r.cache_status] = m["cache"].get(r.cache_status, 0) + 1

        if r.api_key_id is not None and r.status_code == 429:
            by_key[str(r.api_key_id)] = by_key.get(str(r.api_key_id), 0) + 1

    hits = by_cache.get("exact_hit", 0) + by_cache.get("semantic_hit", 0)
    miss = by_cache.get("miss", 0)

    model_out = []
    for name, m in sorted(by_model.items(), key=lambda kv: -kv[1]["requests"]):
        m_hits = m["cache"].get("exact_hit", 0) + m["cache"].get("semantic_hit", 0)
        model_out.append(
            {
                "model": name,
                "requests": m["requests"],
                "tokens": m["tokens"],
                "cache_hit_rate": round(m_hits / m["requests"], 4) if m["requests"] else 0.0,
                "p95": _pct(m["latencies"], 95),
            }
        )

    return {
        "window": window,
        "requests": total,
        "cache": {
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "exact": by_cache.get("exact_hit", 0),
            "semantic": by_cache.get("semantic_hit", 0),
            "misses": miss,
            "bypassed": by_cache.get("bypass", 0),
        },
        "tokens": {"total": tokens_total, "served_from_cache": tokens_from_cache},
        "saved": {
            "tokens": tokens_from_cache,
            "basis": "local_inference",
            "note": "Local inference has no marginal $ cost; token and latency savings "
                    "are the measured figures.",
        },
        "latency": {
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
            "hit_p95": _pct(hit_latencies, 95),
            "miss_p95": _pct(miss_latencies, 95),
        },
        "rate_limit": {
            "rejected": rejections,
            "rejections_by_key": [
                {"api_key_id": int(k), "count": v}
                for k, v in sorted(by_key.items(), key=lambda kv: -kv[1])
            ],
        },
        "by_model": model_out,
    }


@router.get("/stats/timeseries")
async def timeseries(
    db: Annotated[AsyncSession, Depends(get_session)],
    window: Literal["1h", "24h", "7d"] = Query(default="24h"),
    bucket_minutes: int = Query(default=5, ge=1, le=1440),
):
    """Bucketed chart data. Buckets are computed in UTC; the dashboard renders local."""
    since = datetime.now(timezone.utc) - timedelta(hours=_WINDOWS[window])
    rows = (
        await db.execute(
            select(RequestLog.created_at, RequestLog.cache_status, RequestLog.latency_ms,
                   RequestLog.total_tokens).where(RequestLog.created_at >= since)
        )
    ).all()

    buckets: dict[str, dict[str, Any]] = {}
    step = timedelta(minutes=bucket_minutes)
    for created, cache_status, latency, tokens in rows:
        if created is None:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        idx = int((created - since) // step)
        ts = (since + idx * step).isoformat()
        b = buckets.setdefault(
            ts,
            {"ts": ts, "requests": 0, "hits": 0, "tokens": 0, "latency_sum": 0,
             "latencies": []},
        )
        b["requests"] += 1
        if cache_status in ("exact_hit", "semantic_hit"):
            b["hits"] += 1
        b["tokens"] += int(tokens or 0)
        b["latency_sum"] += int(latency or 0)
        b["latencies"].append(int(latency or 0))

    out = []
    for ts in sorted(buckets):
        b = buckets[ts]
        out.append(
            {
                "ts": b["ts"],
                "requests": b["requests"],
                "hits": b["hits"],
                "hit_rate": round(b["hits"] / b["requests"], 4) if b["requests"] else 0.0,
                "tokens": b["tokens"],
                "avg_latency": round(b["latency_sum"] / b["requests"], 2)
                if b["requests"]
                else 0.0,
                "p95": _pct(b["latencies"], 95),
            }
        )
    return {"window": window, "bucket_minutes": bucket_minutes, "series": out}


# =======================================================================
# cache
# =======================================================================
@router.get("/cache/entries")
async def cache_entries(
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(default=50, ge=1, le=500),
):
    """Top cached prompts, ranked by how often they were served from cache."""
    since = datetime.now(timezone.utc) - timedelta(days=7)
    rows = (
        await db.execute(
            select(
                RequestLog.prompt_hash,
                RequestLog.prompt_preview,
                RequestLog.model_served,
                func.count().label("hits"),
                func.max(RequestLog.similarity).label("best_similarity"),
            )
            .where(RequestLog.created_at >= since, RequestLog.cached.is_(True))
            .group_by(RequestLog.prompt_hash, RequestLog.prompt_preview, RequestLog.model_served)
            .order_by(desc("hits"))
            .limit(limit)
        )
    ).all()
    return {
        "items": [
            {
                "prompt_hash": h,
                "preview": preview,
                "model": model,
                "hits": int(hits),
                "best_similarity": float(sim) if sim is not None else None,
            }
            for h, preview, model, hits, sim in rows
        ]
    }


@router.delete("/cache")
async def clear_cache(
    request: Request,
    scope: Literal["exact", "semantic", "all"] = Query(default="all"),
):
    state = request.app.state
    removed: dict[str, int] = {}
    if scope in ("exact", "all"):
        removed["exact"] = await state.exact_cache.clear()
    if scope in ("semantic", "all"):
        removed["semantic"] = await state.semantic_cache.clear()
    return {"cleared": removed, "scope": scope}


@router.get("/cache/threshold")
async def get_threshold(request: Request):
    settings = request.app.state.settings
    from pathlib import Path

    artifact = Path(__file__).resolve().parents[3] / "ml" / "artifacts" / "threshold.json"
    sweep = None
    if artifact.exists():
        import json

        try:
            sweep = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            sweep = None
    return {
        "threshold": settings.semantic_sim_threshold,
        "margin_guard": settings.semantic_margin_guard,
        "last_sweep": sweep,
    }


@router.post("/cache/threshold")
async def set_threshold(body: ThresholdBody, request: Request):
    """Tune at runtime — the sweep on Day 2 writes a value, this lets us verify it live."""
    request.app.state.settings.semantic_sim_threshold = body.threshold
    return {
        "threshold": request.app.state.settings.semantic_sim_threshold,
        "margin_guard": request.app.state.settings.semantic_margin_guard,
    }


# =======================================================================
# router + providers
# =======================================================================
@router.get("/router/decisions")
async def router_decisions(
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(default=50, ge=1, le=500),
):
    rows = (
        await db.execute(
            select(
                RequestLog.created_at, RequestLog.model_requested, RequestLog.model_served,
                RequestLog.route_reason, RequestLog.route_confidence, RequestLog.cache_status,
                RequestLog.latency_ms,
            )
            .order_by(desc(RequestLog.created_at))
            .limit(limit)
        )
    ).all()
    return {
        "items": [
            {
                "created_at": c.isoformat() if c else None,
                "requested": req,
                "served": served,
                "reason": reason,
                "confidence": float(conf) if conf is not None else None,
                "cache_status": cache,
                "latency_ms": lat,
            }
            for c, req, served, reason, conf, cache, lat in rows
        ]
    }


@router.post("/router/score")
async def router_score(body: dict[str, Any], request: Request):
    """Dry run: what would the router decide for this prompt, and why?"""
    from gateway.router.base import RouteContext
    from gateway.router.features import build_context

    messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
    ctx = build_context(
        requested_model=str(body.get("model", "auto")),
        messages=messages,
        prompt_tokens=0,
    )
    decision = request.app.state.router.route(
        RouteContext(
            requested_model=ctx["requested_model"],
            prompt_text=ctx["prompt_text"],
            system_text=ctx["system_text"],
            prompt_tokens=ctx["prompt_tokens"],
            has_code=ctx["has_code"],
            wants_json=ctx["wants_json"],
            is_stream=False,
            features=ctx["features"],
        )
    )
    return {"decision": decision.as_dict(), "features": _public_features(ctx["features"])}


@router.get("/router")
async def router_status(request: Request):
    return request.app.state.router.snapshot()


@router.post("/router/toggle")
async def router_toggle(body: RouterModeBody, request: Request):
    request.app.state.router.set_mode(body.mode)
    return request.app.state.router.snapshot()


@router.get("/providers")
async def providers(request: Request):
    return await request.app.state.registry.health()


@router.get("/telemetry")
async def telemetry_stats(request: Request):
    return request.app.state.telemetry.stats()


# =======================================================================
# helpers
# =======================================================================
def _pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return int(ordered[idx])


def _public_features(feats: dict[str, Any]) -> dict[str, Any]:
    """Drop the raw text-bearing fields before returning features to a client."""
    return {k: v for k, v in feats.items() if k != "vector"}
