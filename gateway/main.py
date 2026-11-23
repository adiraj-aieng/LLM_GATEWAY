"""FastAPI application factory.

The lifespan builds every long-lived object **once** — Redis client, httpx client (inside
the provider), caches, router, registry, telemetry queue and writer — and tears them down
in reverse. Constructing any of these per request would leak sockets and pools and show up
directly in p99.

Ordering in shutdown matters: the telemetry writer is stopped *before* the database is
disposed, otherwise queued records are written into a closed engine.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from redis.asyncio import Redis

from gateway.api.errors import install_error_handlers
from gateway.api.health import router as health_router
from gateway.api.v1 import admin as admin_v1
from gateway.api.v1 import chat as chat_v1
from gateway.api.v1 import embeddings as embed_v1
from gateway.api.v1 import models as models_v1
from gateway.cache.exact import ExactCache
from gateway.cache.semantic import SemanticCache
from gateway.config import get_settings
from gateway.core.rate_limit import RateLimiter
from gateway.db.session import dispose_db, init_db
from gateway.logging import configure_logging, get_logger, request_id_ctx
from gateway.pipeline.request_pipeline import RequestPipeline
from gateway.providers.embedder import build_embedder
from gateway.providers.registry import ProviderRegistry
from gateway.router.learned import LearnedRouter
from gateway.router.policy import PolicyRouter
from gateway.router.rules import RuleRouter
from gateway.telemetry.metrics import MetricsRegistry
from gateway.telemetry.queue import LogQueue
from gateway.telemetry.writer import BatchWriter

log = get_logger("gateway.app")

VERSION = "0.1.0"

#: CORS origins for the dashboard. In production, set this explicitly — a gateway with
#: `*` is a gateway that will happily serve any website's cross-origin traffic.
DEFAULT_CORS = ["http://localhost:5173", "http://127.0.0.1:5173"]


# =======================================================================
# lifespan
# =======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=not settings.is_test)
    log.info("starting", env=settings.app_env, version=VERSION)

    # --- database ----------------------------------------------------------
    await init_db()

    # --- redis -------------------------------------------------------------
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        max_connections=50,
        socket_connect_timeout=5.0,
        socket_timeout=5.0,
    )
    try:
        await redis.ping()
        log.info("redis_connected", url=settings.redis_url)
    except Exception as exc:
        # Fail loudly at boot rather than on the first request — a gateway that
        # "works" while silently bypassing its cache and rate limiter is worse.
        log.error(
            "redis_unavailable",
            url=settings.redis_url,
            error=str(exc),
            hint="In WSL, Redis must bind 0.0.0.0 (not 127.0.0.1) to be reachable "
                 "from Windows. Run: make infra",
        )
        raise

    # --- ml / routing ------------------------------------------------------
    embedder = build_embedder(settings)
    semantic = SemanticCache(redis, settings, embedder)
    exact = ExactCache(redis, settings)
    learned = LearnedRouter(settings)
    loaded = learned.load()
    router = PolicyRouter(
        settings,
        rules=RuleRouter(settings),
        learned=learned,
    )
    registry = ProviderRegistry(settings)
    ratelimit = RateLimiter(redis, settings)

    # --- telemetry ---------------------------------------------------------
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from gateway.db.session import get_session_factory

    factory = get_session_factory()
    telemetry = LogQueue(settings)
    writer = BatchWriter(telemetry, factory, settings)
    metrics = MetricsRegistry()

    # --- pipeline ----------------------------------------------------------
    pipeline = RequestPipeline(
        settings=settings,
        exact=exact,
        semantic=semantic,
        ratelimit=ratelimit,
        registry=registry,
        router=router,
        telemetry=telemetry,
        metrics=metrics,
    )

    # Populate in-process vector indexes from Redis so the cache survives restarts.
    await semantic.warmup()

    writer.start()

    app.state.settings = settings
    app.state.redis = redis
    app.state.embedder = embedder
    app.state.exact_cache = exact
    app.state.semantic_cache = semantic
    app.state.router = router
    app.state.learned_router = learned
    app.state.registry = registry
    app.state.ratelimit = ratelimit
    app.state.pipeline = pipeline
    app.state.telemetry = telemetry
    app.state.writer = writer
    app.state.metrics = metrics
    app.state.version = VERSION
    app.state.started_at = time.time()

    log.info(
        "started_ok",
        router_mode=router.mode,
        learned_artifact=("loaded" if loaded else learned.error),
        models=settings.logical_models(),
        embedder=embedder.name,
    )

    try:
        yield
    finally:
        log.info("shutting_down")
        # Writer first: queued telemetry must be flushed before the engine closes.
        try:
            await writer.stop()
            log.info("telemetry_flushed", **writer_stats(writer, telemetry))
        except Exception as exc:  # pragma: no cover
            log.warning("telemetry_flush_failed", error=str(exc))
        try:
            await registry.aclose()
        except Exception:  # pragma: no cover
            pass
        try:
            await redis.aclose()
        except Exception:  # pragma: no cover
            pass
        await dispose_db()
        log.info("stopped")


def writer_stats(writer: Any, telemetry: LogQueue) -> dict[str, int]:
    stats = telemetry.stats()
    stats["flushed"] = writer.flushed
    stats["failed_batches"] = writer.failed_batches
    return stats


# =======================================================================
# middleware
# =======================================================================
async def request_context_middleware(request: Request, call_next):  # noqa: ARG001
    """Assign a request id, put it in the log context, and time the request.

    Registered with ``@app.middleware("http")`` semantics: signature is
    ``(request, call_next)``, *not* the ``(app)`` factory that
    ``app.add_middleware(...)`` expects.
    """
    incoming = request.headers.get("x-request-id")
    rid = incoming if incoming and len(incoming) < 64 else str(uuid.uuid4())
    token = request_id_ctx.set(rid)
    request.state.request_id = rid

    started = time.perf_counter()
    try:
        response: Response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        request_id_ctx.reset(token)

    response.headers["x-request-id"] = rid
    response.headers["x-gateway-latency-ms"] = str(int(round(elapsed_ms)))

    # Per-route timing metric (skips the streaming body itself — see note below).
    metrics: MetricsRegistry | None = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.observe("http_request", elapsed_ms, route=_route_label(request))

    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round(elapsed_ms, 2),
    )
    return response


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


# =======================================================================
# app
# =======================================================================
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="LLM Gateway",
        version=VERSION,
        description=(
            "OpenAI-compatible gateway with a two-stage cache, token-based rate "
            "limiting, a learned model router and cost/latency telemetry."
        ),
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    app.middleware("http")(request_context_middleware)

    # Dashboard origin only; never `*` in production.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEFAULT_CORS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "x-request-id",
            "x-gateway-cache",
            "x-gateway-similarity",
            "x-gateway-model-upstream",
            "x-gateway-route-reason",
            "x-gateway-latency-ms",
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "retry-after",
        ],
    )

    install_error_handlers(app)

    app.include_router(chat_v1.router, prefix="/v1", tags=["chat"])
    app.include_router(models_v1.router, prefix="/v1", tags=["models"])
    app.include_router(embed_v1.router, prefix="/v1", tags=["embeddings"])
    app.include_router(admin_v1.router, tags=["admin"])
    app.include_router(health_router)

    return app


#: Uvicorn entrypoint: ``uvicorn gateway.main:app``.
app = create_app()
