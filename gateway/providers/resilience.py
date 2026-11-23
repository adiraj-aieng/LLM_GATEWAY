"""Reliability: retry with jitter, and a circuit breaker.

Retry policy
------------
Retry **only** ``429``, ``5xx`` and timeouts. A ``400`` is our fault (bad request) and
retrying it just burns quota. Crucially, once any bytes have reached the client we must
**not** retry — the client would see duplicated content. Callers enforce that by not
retrying mid-stream; this module only retries whole-request attempts.

Backoff is exponential **with full jitter** (``random(0, min(cap, base * 2^attempt))``).
Without jitter, concurrent failures retry in lockstep and stampede the upstream — the
classic thundering-herd.

Circuit breaker
--------------
``closed → open → half-open → closed``. In half-open state **exactly one** probe is
allowed through; if it succeeds we close, if it fails we reopen. Letting N probes through
re-floods an upstream that is already struggling.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from gateway.config import Settings
from gateway.logging import get_logger

log = get_logger("gateway.resilience")

T = TypeVar("T")

#: Only these upstream statuses are worth retrying.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    cooldown_s: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float = 0.0
    probing: bool = False
    last_error: str = ""

    def allow(self) -> bool:
        """True if a request may proceed right now."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.cooldown_s:
                self.state = CircuitState.HALF_OPEN
                self.probing = True
                log.info("circuit_half_open")
                return True
            return False
        # HALF_OPEN: only the single probe proceeds.
        if self.probing:
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.probing = False
        self.last_error = ""
        if self.state != CircuitState.CLOSED:
            log.info("circuit_closed")
        self.state = CircuitState.CLOSED

    def record_failure(self, error: str = "") -> None:
        self.failures += 1
        self.last_error = error
        if self.state == CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                log.warning(
                    "circuit_opened",
                    failures=self.failures,
                    cooldown_s=self.cooldown_s,
                )
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            self.probing = False

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.probing = False
        self.last_error = ""

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "failures": self.failures,
            "last_error": self.last_error,
            "cooldown_s": self.cooldown_s,
        }


@dataclass
class BreakerRegistry:
    """One breaker per ``provider:model`` — a bad model shouldn't take down the rest."""

    settings: Settings
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=self.settings.circuit_failure_threshold,
                cooldown_s=self.settings.circuit_cooldown_s,
            )
        return self._breakers[name]

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {k: v.snapshot() for k, v in self._breakers.items()}


async def with_retry(
    op: Callable[[], Awaitable[T]],
    *,
    settings: Settings,
    label: str,
    is_retryable: Callable[[Exception], bool] | None = None,
) -> T:
    """Run ``op`` with exponential backoff + full jitter."""
    attempts = max(1, settings.retry_max_attempts)
    last_exc: Exception | None = None

    for attempt in range(attempts):
        try:
            return await op()
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable(exc) if is_retryable else _default_retryable(exc)
            if not retryable or attempt == attempts - 1:
                raise
            base = min(2.0 * (2**attempt), 8.0)
            delay = random.uniform(0, base)  # full jitter
            log.warning(
                "upstream_retry",
                label=label,
                attempt=attempt + 1,
                max_attempts=attempts,
                delay_s=round(delay, 3),
                error=str(exc)[:200],
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _default_retryable(exc: Exception) -> bool:
    from gateway.providers.base import ProviderError

    if isinstance(exc, ProviderError):
        if exc.upstream_status is not None:
            return exc.upstream_status in RETRYABLE_STATUSES
        return exc.retryable
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return False
