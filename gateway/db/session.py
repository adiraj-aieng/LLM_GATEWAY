"""Async engine + session management.

The engine is created **once per process** and shared. Creating an engine per request
leaks connection pools and is one of the classic ways to destroy p99 latency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from gateway.config import get_settings
from gateway.db.base import Base

_engine: AsyncEngine | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def _sqlite_pragmas(engine: AsyncEngine) -> None:
    """WAL + busy_timeout so concurrent telemetry writes don't immediately hit locks."""
    if engine.url.get_backend_name() != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pragma: no cover - driver detail
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine, _factory
    if _engine is None:
        settings = get_settings()
        kwargs: dict = {"echo": False, "future": True}
        if settings.database_url.startswith("sqlite"):
            # StaticPool keeps a single connection, which is what in-memory SQLite needs
            # for tests and avoids WAL file churn in dev.
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_async_engine(settings.database_url, **kwargs)
        _sqlite_pragmas(_engine)
        _factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _factory is not None
    return _factory


async def init_db() -> None:
    """Create tables if they don't exist. Idempotent."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    global _engine, _factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session, committed by the caller."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
