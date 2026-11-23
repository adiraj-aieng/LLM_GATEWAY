"""Batch writer: drains the telemetry queue into the database.

Flushes when either

* ``LOG_FLUSH_INTERVAL_S`` seconds have elapsed, **or**
* ``LOG_FLUSH_BATCH`` records have accumulated,

whichever comes first. On shutdown it drains everything still queued — otherwise the last
second of traffic, i.e. often the most interesting part of a load test, disappears.

Inserts are ``executemany``-style bulk inserts; one round trip per record would make the
writer the bottleneck and back up the queue.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from gateway.config import Settings
from gateway.db.models import ApiKey, RequestLog
from gateway.logging import get_logger
from gateway.telemetry.queue import LogQueue

log = get_logger("gateway.telemetry.writer")


class BatchWriter:
    def __init__(
        self,
        queue: LogQueue,
        session_factory: async_sessionmaker,
        settings: Settings,
    ):
        self._queue = queue
        self._factory = session_factory
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._stopping = False
        self.flushed = 0
        self.failed_batches = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="telemetry-writer")

    async def stop(self) -> None:
        """Signal shutdown, wait, then drain anything left."""
        self._stopping = True
        self._queue.close()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):  # pragma: no cover
                self._task.cancel()
            self._task = None
        # Final drain — don't lose the tail of a run.
        await self._flush_remaining()

    # ------------------------------------------------------------------
    async def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        interval = self._settings.log_flush_interval_s
        max_batch = self._settings.log_flush_batch

        while True:
            timeout = max(0.05, interval - (time.monotonic() - last_flush))
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                item = None
                if batch:
                    await self._write(batch)
                    batch = []
                    last_flush = time.monotonic()
                continue

            if item is None:  # shutdown sentinel
                if batch:
                    await self._write(batch)
                    batch = []
                return

            batch.append(item)
            if len(batch) >= max_batch or time.monotonic() - last_flush >= interval:
                await self._write(batch)
                batch = []
                last_flush = time.monotonic()

    # ------------------------------------------------------------------
    async def _flush_remaining(self) -> None:
        leftover: list[dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except Exception:
                break
            if item is None:
                continue
            leftover.append(item)
        if leftover:
            await self._write(leftover)

    # ------------------------------------------------------------------
    async def _write(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            async with self._factory() as session:
                await session.execute(insert(RequestLog), batch)
                await session.commit()
            self.flushed += len(batch)
        except Exception as exc:
            self.failed_batches += 1
            log.error(
                "telemetry_write_failed",
                batch_size=len(batch),
                failed_batches=self.failed_batches,
                error=str(exc)[:300],
            )


async def touch_api_keys(
    session_factory: async_sessionmaker, key_ids: set[int]
) -> None:
    """Batch-update ``last_used_at`` for keys seen during a flush window.

    Done here rather than on the request path so authentication stays a single read.
    """
    if not key_ids:
        return
    try:
        from sqlalchemy import update

        async with session_factory() as session:
            await session.execute(
                update(ApiKey)
                .where(ApiKey.id.in_(key_ids))
                .values(last_used_at=datetime.now(timezone.utc))
            )
            await session.commit()
    except Exception as exc:  # pragma: no cover - never fail the writer
        log.warning("touch_api_keys_failed", error=str(exc)[:200])
