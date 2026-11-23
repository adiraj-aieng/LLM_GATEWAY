"""API-key issue, hashing and verification.

Design notes:

* Keys are ``sk-gw-`` + 48 hex chars (192 bits). Shown **once** at creation, never again.
* Stored as blake2b over ``pepper:key``. Because keys are high-entropy random strings,
  a slow KDF (bcrypt/argon2) buys nothing — an attacker can't dictionary-attack 192 bits.
  The *pepper* is what protects a database-only leak, and it lives in env, not the DB.
* Lookup is always ``WHERE key_hash = ? AND is_active`` — never on ``key_prefix`` alone.
* ``last_used_at`` is deliberately **not** written here. This is the hot path; a write per
  request would serialize every request on the DB. Telemetry updates it in batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.errors import AuthenticationError, PermissionError_
from gateway.config import Settings, get_settings
from gateway.core.hashing import blake2b_hex
from gateway.db.models import ApiKey, User

KEY_PREFIX = "sk-gw-"
_KEY_BYTES = 24  # → 48 hex chars


@dataclass(frozen=True)
class Principal:
    """The authenticated identity attached to a request."""

    user_id: int
    api_key_id: int
    label: str
    plan: str
    rate_limit_tokens_min: int
    daily_token_budget: int | None
    allowed_models: tuple[str, ...]

    def allows_model(self, logical: str) -> bool:
        """`auto` is always permitted — it's the router's own entry point."""
        return logical == "auto" or logical in self.allowed_models

    def require_model(self, logical: str) -> None:
        if not self.allows_model(logical):
            raise PermissionError_(
                f"API key is not permitted to use model `{logical}`.",
                code="model_not_allowed",
                param="model",
            )


# =======================================================================
# key lifecycle
# =======================================================================
def generate_api_key() -> str:
    """New random key. Returned to the caller exactly once."""
    import secrets

    return KEY_PREFIX + secrets.token_hex(_KEY_BYTES)


def hash_api_key(key: str, pepper: str) -> str:
    """Peppered blake2b hex digest of a full key."""
    return blake2b_hex(f"{pepper}:{key}", size=32)


def key_prefix_of(key: str) -> str:
    """Display prefix, e.g. ``sk-gw-9f3a``. Safe to log and show in the UI."""
    return key[: len(KEY_PREFIX) + 4]


def looks_like_gateway_key(value: str) -> bool:
    """Cheap pre-check so a malformed header doesn't hit the database."""
    return value.startswith(KEY_PREFIX) and len(value) >= len(KEY_PREFIX) + 32


# =======================================================================
# verification
# =======================================================================
def extract_bearer(header: str | None) -> str | None:
    """Pull the token out of ``Authorization: Bearer <token>`` (case-insensitive)."""
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def authenticate(
    db: AsyncSession,
    presented: str | None,
    settings: Settings | None = None,
) -> Principal:
    """Resolve a presented key to a :class:`Principal`, or raise 401/403."""
    settings = settings or get_settings()

    if not presented:
        raise AuthenticationError(
            "Missing API key. Pass `Authorization: Bearer sk-gw-...`.",
            code="missing_api_key",
        )
    if not looks_like_gateway_key(presented):
        # Deliberately identical message to a wrong key: don't leak which check failed.
        raise AuthenticationError("Invalid API key.", code="invalid_api_key")

    digest = hash_api_key(presented, settings.api_key_pepper)

    stmt = (
        select(ApiKey, User)
        .join(User, ApiKey.user_id == User.id)
        .where(ApiKey.key_hash == digest, ApiKey.is_active.is_(True))
    )
    row = (await db.execute(stmt)).first()

    if row is None:
        raise AuthenticationError("Invalid API key.", code="invalid_api_key")

    key, user = row

    if not user.is_active:
        raise AuthenticationError("Invalid API key.", code="invalid_api_key")

    if key.expires_at is not None:
        expires = key.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            raise AuthenticationError("API key has expired.", code="expired_api_key")

    return Principal(
        user_id=user.id,
        api_key_id=key.id,
        label=key.label,
        plan=user.plan,
        rate_limit_tokens_min=key.rate_limit_tokens_min,
        daily_token_budget=key.daily_token_budget,
        allowed_models=tuple(key.allowed_models or ("auto",)),
    )
