"""FastAPI dependencies.

Everything is pulled from ``app.state`` (built once in the lifespan) rather than being
constructed per-request — a new Redis client or httpx client per request is the standard
way to turn a working proxy into a slow one.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.errors import AuthenticationError, PermissionError_
from gateway.cache.exact import ExactCache
from gateway.cache.semantic import SemanticCache
from gateway.config import Settings
from gateway.core.auth import Principal, authenticate, extract_bearer
from gateway.core.rate_limit import RateLimiter
from gateway.db.session import get_session
from gateway.pipeline.request_pipeline import RequestPipeline
from gateway.providers.registry import ProviderRegistry
from gateway.router.policy import PolicyRouter
from gateway.telemetry.metrics import MetricsRegistry
from gateway.telemetry.queue import LogQueue

# --- typed aliases ---------------------------------------------------------
RequestDep = Annotated[Request, None]


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_pipeline(request: Request) -> RequestPipeline:
    return request.app.state.pipeline


def get_metrics(request: Request) -> MetricsRegistry:
    return request.app.state.metrics


def get_telemetry(request: Request) -> LogQueue:
    return request.app.state.telemetry


def get_router(request: Request) -> PolicyRouter:
    return request.app.state.router


def get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.registry


def get_exact(request: Request) -> ExactCache:
    return request.app.state.exact_cache


def get_semantic(request: Request) -> SemanticCache:
    return request.app.state.semantic_cache


def get_ratelimit(request: Request) -> RateLimiter:
    return request.app.state.ratelimit


# =======================================================================
# auth
# =======================================================================
async def require_api_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Principal:
    """Resolve ``Authorization: Bearer sk-gw-…`` to a :class:`Principal`.

    Raises 401 for a missing/unknown/expired key and 403 when the key may not use the
    requested model. The resolved principal is stashed on ``request.state`` so middleware
    and telemetry can see who called without resolving it twice.
    """
    header = request.headers.get("authorization")
    token = extract_bearer(header)
    settings = request.app.state.settings
    principal = await authenticate(db, token, settings)
    request.state.principal = principal
    return principal


async def require_admin(request: Request) -> None:
    """Gate ``/admin/*``. Separate from API keys — different audience, different secret."""
    settings: Settings = request.app.state.settings
    header = request.headers.get("authorization")
    token = extract_bearer(header)
    if not token:
        raise PermissionError_("Admin token required.", code="admin_token_required")
    if token != settings.admin_token:
        raise PermissionError_("Admin token required.", code="admin_token_required")


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "")


# Re-export the session generator so routes need only import from here.
__all__ = [
    "get_settings",
    "get_redis",
    "get_pipeline",
    "get_metrics",
    "get_telemetry",
    "get_router",
    "get_registry",
    "get_exact",
    "get_semantic",
    "get_ratelimit",
    "require_api_key",
    "require_admin",
    "request_id_of",
    "get_session",
]
