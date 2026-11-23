"""In-process telemetry queue.

The single hard rule here: **the hot path must never block on telemetry.**

``emit()`` does a ``put_nowait``. If the queue is full the record is dropped and the drop
is counted — visible as ``dropped`` rather than hidden behind an ever-growing backlog that
eventually OOMs the process under load. A gateway that dies at 1000 req/s because it was
writing logs is worse than one that loses 0.1% of log rows at 1000 req/s.

Drops are exported as a metric precisely because a silently lossy log is the dangerous
kind.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from gateway.config import Settings
from gateway.logging import get_logger

log = get_logger("gateway.telemetry")

#: Hard ceiling on undrained records. The queue is bounded anyway; this is the belt to
#: the queue's braces, and the point where we start warning loudly.
_DROP_WARN_EVERY = 1000


class LogQueue:
    """Bounded, non-blocking queue of ready-to-insert log records."""

    def __init__(self, settings: Settings):
        self._maxsize = settings.log_queue_maxsize
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=self._maxsize
        )
        self.dropped = 0
        self.emitted = 0
        self._dropped_since_warn = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def emit(self, record: dict[str, Any]) -> bool:
        """Enqueue without ever waiting. Returns False if the record was dropped."""
        self.emitted += 1
        try:
            self._queue.put_nowait(record)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            self._dropped_since_warn += 1
            if self._dropped_since_warn >= _DROP_WARN_EVERY:
                self._dropped_since_warn = 0
                log.warning(
                    "telemetry_dropped",
                    dropped_total=self.dropped,
                    emitted_total=self.emitted,
                    queue_size=self._queue.qsize(),
                )
            return False

    def close(self) -> None:
        """Sentinel telling the writer to drain and exit."""
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            # Can't signal: the writer's shutdown timeout will handle it.
            pass

    async def get(self) -> dict[str, Any] | None:
        return await self._queue.get()

    def get_nowait(self) -> dict[str, Any] | None:
        """Non-blocking drain step, used during shutdown."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def stats(self) -> dict[str, int]:
        return {
            "emitted": self.emitted,
            "dropped": self.dropped,
            "queued": self._queue.qsize(),
            "capacity": self._maxsize,
        }


class RollingWindow:
    """Fixed-size ring of recent latencies — cheap percentiles without a database query."""

    def __init__(self, capacity: int = 2048):
        self._capacity = capacity
        self._items: deque[float] = deque(maxlen=capacity)

    def add(self, value: float) -> None:
        self._items.append(value)

    def percentile(self, p: float) -> float:
        if not self._items:
            return 0.0
        data = sorted(self._items)
        idx = min(len(data) - 1, max(0, int(round((p / 100.0) * (len(data) - 1)))))
        return data[idx]

    def count(self) -> int:
        return len(self._items)

    def mean(self) -> float:
        if not self._items:
            return 0.0
        return sum(self._items) / len(self._items)

    def clear(self) -> None:
        self._items.clear()
