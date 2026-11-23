"""Exact (byte-identical) cache.

Two things make this correct rather than merely fast:

1. **The key** — built by :mod:`gateway.cache.keys`, it folds in messages, system prompt,
   tools schema and sampling params. Anything that changes an answer is in the key.
2. **The ``max_tokens`` guard** — keys bucket ``max_tokens`` *upward* for hit rate, so a
   hit must be checked against what the caller actually allowed before it is served.

Eviction is a per-scope ZSET scored by last access (true LRU) with an explicit entry cap,
*plus* a TTL. Relying only on Redis' ``maxmemory-policy`` would make behaviour depend on
deployment config rather than code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from gateway.cache.keys import fits_within_max_tokens
from gateway.config import Settings
from gateway.logging import get_logger

log = get_logger("gateway.cache.exact")

VALUE_TTL_PREFIX = "gw:v1:exact:lru"
MISSES = "miss"


@dataclass(frozen=True)
class ExactHit:
    response: dict[str, Any]
    usage: dict[str, int]
    created_at: str
    age_s: float


class ExactCache:
    def __init__(self, redis: Redis, settings: Settings):
        self._redis = redis
        self._settings = settings

    # ------------------------------------------------------------------
    @staticmethod
    def _lru_key(scope: str) -> str:
        return f"{VALUE_TTL_PREFIX}:{scope}"

    async def get(
        self,
        key: str,
        *,
        scope: str,
        requested_max_tokens: int | None = None,
    ) -> ExactHit | None:
        """Return a hit, or ``None``. Never raises — a cache failure must not fail a request."""
        if not self._settings.exact_cache_enabled:
            return None
        try:
            raw = await self._redis.get(key)
            if not raw:
                return None

            payload = json.loads(raw)
            entry = payload.get("entry", payload)  # tolerate both shapes

            # Guard: bucketed max_tokens must not return more than was asked for.
            if not _fits(entry.get("usage", {}), requested_max_tokens):
                log.debug("exact_cache_over_max_tokens", key=key)
                return None

            # Bump LRU position (best effort).
            lru = self._lru_key(scope)
            await self._redis.zadd(lru, {key: _now_ts()})
            await self._redis.pexpire(
                lru, self._settings.exact_cache_ttl_s * 1000
            )

            created = entry.get("created_at", "")
            age = _age_seconds(created)
            return ExactHit(
                response=entry.get("response", {}),
                usage=entry.get("usage", {}) or {},
                created_at=created,
                age_s=age,
            )
        except Exception as exc:  # pragma: no cover - cache must never break the path
            log.warning("exact_cache_get_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    async def set(
        self,
        key: str,
        *,
        scope: str,
        response: dict[str, Any],
        usage: dict[str, int],
    ) -> None:
        if not self._settings.exact_cache_enabled:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            payload = {
                "entry": {
                    "response": response,
                    "usage": usage,
                    "created_at": now,
                }
            }
            ttl = self._settings.exact_cache_ttl_s
            pipe = self._redis.pipeline(transaction=True)
            pipe.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)
            lru = self._lru_key(scope)
            pipe.zadd(lru, {key: _now_ts()})
            pipe.pexpire(lru, ttl * 1000)
            # trim to the configured cap, oldest-first
            pipe.zremrangebyrank(lru, 0, -self._settings.exact_cache_max_entries - 1)
            await pipe.execute()
        except Exception as exc:  # pragma: no cover
            log.warning("exact_cache_set_failed", error=str(exc))

    # ------------------------------------------------------------------
    async def clear(self, scope: str | None = None) -> int:
        """Delete entries. With a scope, only that scope's keys; otherwise scan-and-delete."""
        removed = 0
        try:
            if scope:
                lru = self._lru_key(scope)
                keys = await self._redis.zrange(lru, 0, -1)
                if keys:
                    removed = await self._redis.delete(*keys)
                    await self._redis.delete(lru)
                return int(removed)

            async for key in self._redis.scan_iter(match="gw:v1:exact:*", count=500):
                if key.startswith(VALUE_TTL_PREFIX):
                    continue
                removed += await self._redis.delete(key)
        except Exception as exc:  # pragma: no cover
            log.warning("exact_cache_clear_failed", error=str(exc))
        return int(removed)

    async def size(self, scope: str) -> int:
        try:
            return int(await self._redis.zcard(self._lru_key(scope)))
        except Exception:  # pragma: no cover
            return 0


def _fits(usage: dict[str, Any], requested_max: int | None) -> bool:
    """Mirror of ``keys.fits_within_max_tokens`` for a raw usage dict."""
    if requested_max is None:
        return True
    produced = int(usage.get("completion_tokens", 0) or 0)
    return produced <= requested_max


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _age_seconds(created: str) -> float:
    if not created:
        return 0.0
    try:
        dt = datetime.fromisoformat(created)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except ValueError:  # pragma: no cover
        return 0.0
