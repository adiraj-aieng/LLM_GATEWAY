"""``GET /healthz`` — liveness, mounted at the application root.

Deliberately touches **nothing**: no Redis, no Postgres, no upstream. A health probe that
fails because the cache is briefly unavailable will get the service restarted by an
orchestrator for the wrong reason. Readiness (which *does* check dependencies) is a
separate endpoint, ``/readyz``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from gateway.api import deps

router = APIRouter(tags=["meta"])


@router.get("/healthz")
async def healthz(request: Request):
    state = request.app.state
    started = getattr(state, "started_at", time.time())
    return {
        "status": "ok",
        "version": getattr(state, "version", "0.1.0"),
        "uptime_s": int(time.time() - started),
    }


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: checks Redis and the upstream. Used for deploy gates, not liveness."""
    from redis.asyncio import Redis

    checks: dict[str, bool] = {}

    redis: Redis | None = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            checks["redis"] = bool(await redis.ping())
        except Exception as exc:
            checks["redis"] = False
            request.app.state.last_redis_error = str(exc)
    else:
        checks["redis"] = False

    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        health = await registry.health()
        checks["upstream"] = any(bool(v.get("available")) for v in health.values())
        checks["providers"] = health  # type: ignore[assignment]
    else:
        checks["upstream"] = False

    ready = checks.get("redis", False) and checks.get("upstream", False)
    return {"status": "ready" if ready else "degraded", "checks": checks}
