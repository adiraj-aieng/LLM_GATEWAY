"""Token-bucket rate limiting, denominated in **tokens rather than requests**.

Three buckets are enforced together and atomically by a single Lua script:

=====================  ==========================================
bucket                 source of capacity
=====================  ==========================================
``gw:rl:user:{id}``    plan default / settings
``gw:rl:key:{id}``     the API key's ``rate_limit_tokens_min``
``gw:rl:global``       ``RATE_LIMIT_CAPACITY``
=====================  ==========================================

Pre-charge / reconcile
----------------------
For non-streaming requests the token cost is known after the fact. For **streaming** it
isn't — usage arrives (if at all) in a final SSE chunk, and may never arrive. So we
reserve ``max_tokens`` up front and refund the unused portion afterwards. Without this,
concurrent streams can exhaust a bucket before any usage is observed.

The script is loaded once via ``SCRIPT LOAD`` and reused with ``EVALSHA``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from redis.asyncio import Redis
from redis.exceptions import NoScriptError

from gateway.config import Settings
from gateway.core.auth import Principal
from gateway.logging import get_logger

log = get_logger("gateway.ratelimit")

_LUA_PATH = Path(__file__).with_name("rate_limit.lua")


@dataclass(frozen=True)
class BucketSpec:
    key: str
    capacity: float
    refill_rate: float
    enabled: bool = True


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: int
    limit: int


class RateLimiter:
    """Thin wrapper around the Lua token bucket."""

    def __init__(self, redis: Redis, settings: Settings):
        self._redis = redis
        self._settings = settings
        self._sha: str | None = None

    # ------------------------------------------------------------------
    async def _ensure_script(self) -> str:
        if self._sha is None:
            source = _LUA_PATH.read_text(encoding="utf-8")
            self._sha = await self._redis.script_load(source)
        return self._sha

    async def _eval(self, keys: list[str], args: list[float | int]):
        sha = await self._ensure_script()
        try:
            return await self._redis.evalsha(sha, len(keys), *keys, *args)
        except NoScriptError:
            # Redis restarted or SCRIPT FLUSH — reload and retry once.
            self._sha = None
            sha = await self._ensure_script()
            return await self._redis.evalsha(sha, len(keys), *keys, *args)

    # ------------------------------------------------------------------
    def _buckets(self, principal: Principal) -> list[BucketSpec]:
        s = self._settings
        return [
            BucketSpec(
                key=f"gw:rl:user:{principal.user_id}",
                capacity=float(s.rate_limit_capacity),
                refill_rate=float(s.rate_limit_refill_per_s),
            ),
            BucketSpec(
                key=f"gw:rl:key:{principal.api_key_id}",
                capacity=float(principal.rate_limit_tokens_min),
                refill_rate=float(principal.rate_limit_tokens_min) / 60.0,
            ),
            BucketSpec(
                key="gw:rl:global",
                capacity=float(s.rate_limit_capacity) * 10,
                refill_rate=float(s.rate_limit_refill_per_s) * 10,
            ),
        ]

    # ------------------------------------------------------------------
    async def check(
        self, principal: Principal, cost: float
    ) -> RateLimitResult:
        """All-or-nothing consume across the three buckets."""
        if not self._settings.rate_limit_enabled:
            return RateLimitResult(True, 0, 0, principal.rate_limit_tokens_min)

        buckets = self._buckets(principal)
        keys = [b.key for b in buckets]
        args: list[float | int] = [float(cost), 60_000]
        for b in buckets:
            args.extend(
                [
                    b.capacity,
                    b.refill_rate,
                    1 if b.enabled else 0,
                ]
            )

        raw = await self._eval(keys, args)
        allowed = int(raw[0]) == 1
        remaining = int(raw[1])
        retry_after = int(raw[2])

        if not allowed:
            log.info(
                "rate_limited",
                api_key_id=principal.api_key_id,
                user_id=principal.user_id,
                cost=cost,
                retry_after=retry_after,
            )
        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, remaining),
            retry_after=max(1, retry_after),
            limit=principal.rate_limit_tokens_min,
        )

    # ------------------------------------------------------------------
    async def refund(
        self, principal: Principal, tokens: float
    ) -> None:
        """Give back an unused pre-charge. Best-effort: failures must not fail the request."""
        if tokens <= 0 or not self._settings.rate_limit_enabled:
            return
        try:
            buckets = self._buckets(principal)
            for b in buckets:
                # HMINCRBY-style: read, add, cap at capacity, write.
                await self._redis.hincrbyfloat(b.key, "tokens", tokens)
                await self._redis.hset(
                    b.key,
                    mapping={"refill_ts": _now_ms()},
                )
                # cap at capacity so a refund can't overfill the bucket
                current = float(await self._redis.hget(b.key, "tokens") or 0)
                if current > b.capacity:
                    await self._redis.hset(
                        b.key, mapping={"tokens": str(b.capacity)}
                    )
        except Exception as exc:  # pragma: no cover - never fail the hot path
            log.warning("rate_limit_refund_failed", error=str(exc))

    # ------------------------------------------------------------------
    async def set_capacity(self, key_id: int, tokens_per_min: int) -> None:
        """Runtime adjustment for an API key (admin endpoint)."""
        await self._redis.hset(
            f"gw:rl:key:{key_id}",
            mapping={
                "capacity": str(tokens_per_min),
                "refill_ts": str(_now_ms()),
            },
        )


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
