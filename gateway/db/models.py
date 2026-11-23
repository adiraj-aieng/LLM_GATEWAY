"""ORM models.

Portability notes (dev = SQLite, prod = Postgres):

* ``BigInteger`` PKs are downgraded to ``Integer`` on SQLite — SQLite only supports
  autoincrement on ``INTEGER PRIMARY KEY``, so a straight ``BIGSERIAL`` silently breaks.
* ``JSON`` rather than ``JSONB`` so the same model works on both. Postgres loses GIN
  indexing on the column; if we ever query inside ``metadata`` we'll add a variant.
* ``DateTime(timezone=True)`` yields ``TIMESTAMPTZ`` on Postgres and aware datetimes
  locally. Everything is stored in **UTC**; the dashboard renders local time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.db.base import Base

# --- type helpers ----------------------------------------------------------
PK = BigInteger().with_variant(Integer, "sqlite")
FK = BigInteger().with_variant(Integer, "sqlite")

#: Cache outcome — mirrors `x-gateway-cache`.
CACHE_STATUSES = ("exact_hit", "semantic_hit", "miss", "bypass")


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use naive `datetime.utcnow()`."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


# =======================================================================
# identity
# =======================================================================
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(PK, primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    email: Mapped[str | None] = mapped_column(String(255), index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    plan: Mapped[str] = mapped_column(String(32), default="free", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(PK, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        FK, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    #: First characters of the key, for display and logs. Never used alone for lookup.
    key_prefix: Mapped[str] = mapped_column(String(24), nullable=False)
    #: blake2b-256 hex of the full key. Lookup is always on this, plus `is_active`.
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rate_limit_tokens_min: Mapped[int] = mapped_column(
        Integer, default=100000, nullable=False
    )
    daily_token_budget: Mapped[int | None] = mapped_column(Integer)
    allowed_models: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: ["auto", "fast", "smart", "coder"], nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="api_keys")

    @property
    def masked(self) -> str:
        """Safe-to-display form, e.g. `sk-gw-9f3a…`."""
        return f"{self.key_prefix}…"


# =======================================================================
# telemetry
# =======================================================================
class RequestLog(Base):
    """Append-only. Written by the async batch writer, never by the request path."""

    __tablename__ = "request_logs"
    __table_args__ = (
        Index("idx_logs_time", "created_at"),
        Index("idx_logs_apikey_time", "api_key_id", "created_at"),
        Index("idx_logs_model_time", "model_served", "created_at"),
        Index("idx_logs_cache_time", "cache_status", "created_at"),
        Index("idx_logs_user_time", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(PK, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    user_id: Mapped[int | None] = mapped_column(FK, ForeignKey("users.id"))
    api_key_id: Mapped[int | None] = mapped_column(FK, ForeignKey("api_keys.id"))

    # what was asked for vs. what served it
    model_requested: Mapped[str] = mapped_column(String(128), nullable=False)
    model_served: Mapped[str] = mapped_column(String(128), nullable=False)
    upstream_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    route_reason: Mapped[str | None] = mapped_column(String(64))
    route_confidence: Mapped[float | None] = mapped_column(Float)

    # cache
    cache_status: Mapped[str] = mapped_column(String(24), nullable=False)
    similarity: Mapped[float | None] = mapped_column(Float)

    # volume
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_from_cache: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # money (provenance tracked explicitly)
    cost_usd: Mapped[float] = mapped_column(
        Numeric(12, 6), default=0, nullable=False
    )
    cached_cost_usd: Mapped[float] = mapped_column(
        Numeric(12, 6), default=0, nullable=False
    )
    cost_basis: Mapped[str] = mapped_column(
        String(32), default="local_inference", nullable=False
    )
    price_version: Mapped[str] = mapped_column(
        String(16), default="v1", nullable=False
    )

    # timing
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    ttfb_ms: Mapped[int | None] = mapped_column(Integer)
    upstream_latency_ms: Mapped[int | None] = mapped_column(Integer)
    queue_wait_ms: Mapped[int | None] = mapped_column(Integer)

    # outcome
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64))
    cached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # content — preview is truncated + redacted before it reaches this row
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    system_prompt_hash: Mapped[str | None] = mapped_column(String(64))
    prompt_preview: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )


class ModelPrice(Base):
    """Versioned price table so historical cost stays reproducible."""

    __tablename__ = "model_prices"
    __table_args__ = (UniqueConstraint("version", "model_id", name="uq_price_model"),)

    id: Mapped[int] = mapped_column(PK, primary_key=True)
    version: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_per_mtok: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    output_per_mtok: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class HourlyRollup(Base):
    """Pre-aggregated stats (Day 2). Without it the dashboard's p95 scan is O(all rows)."""

    __tablename__ = "hourly_rollups"
    __table_args__ = (
        UniqueConstraint(
            "bucket_start", "api_key_id", "model_served", "cache_status",
            name="uq_rollup_dims",
        ),
    )

    id: Mapped[int] = mapped_column(PK, primary_key=True)
    bucket_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    api_key_id: Mapped[int | None] = mapped_column(FK, ForeignKey("api_keys.id"))
    model_served: Mapped[str] = mapped_column(String(128), nullable=False)
    cache_status: Mapped[str] = mapped_column(String(24), nullable=False)

    requests: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    tokens_total: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    tokens_from_cache: Mapped[int] = mapped_column(
        BigInteger, default=0, nullable=False
    )
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    cached_cost_usd: Mapped[float] = mapped_column(
        Numeric(12, 6), default=0, nullable=False
    )
    #: Latencies kept as a list so percentiles merge exactly across buckets.
    latencies_ms: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
