"""The request pipeline: the one place that knows the order of operations.

    auth → rate limit → exact cache → semantic cache → route → provider → cache write
                                                                          ↓
                                                  refund reservation → telemetry

Everything else in the codebase is a building block; this module decides when each one
runs. Keeping the sequence in a single file is what makes the behaviour reviewable — and
what makes it possible to assert, in tests, that a cache hit never reaches an upstream.

Two invariants the tests enforce:

* **The rate-limit reservation is always settled.** Refunded on success, on cache hit, and
  in the failure path — a leak here silently eats a client's budget.
* **Telemetry is emitted exactly once per request**, including error paths.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from gateway.api.errors import (
    GatewayError,
    NoProviderAvailableError,
    RateLimitError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gateway.cache.exact import ExactCache, ExactHit
from gateway.cache.keys import (
    ExactKeyInput,
    SamplingParams,
    build_exact_key,
    bypass_reason,
    system_prompt_hash,
    tools_hash,
)
from gateway.cache.semantic import SemanticCache, SemanticHit
from gateway.config import Settings
from gateway.core.auth import Principal
from gateway.core.hashing import blake2b_hex
from gateway.core.normalize import normalize_messages, normalize_text
from gateway.core.rate_limit import RateLimiter
from gateway.logging import get_logger
from gateway.pipeline import cost as cost_mod
from gateway.pipeline.streaming import (
    StreamAccumulator,
    assemble_response,
    make_chunk,
    sse_done,
    sse_event,
)
from gateway.providers.base import ProviderError
from gateway.providers.registry import ProviderRegistry, Route
from gateway.router.base import RouteContext, RouteDecision
from gateway.router.features import build_context
from gateway.router.policy import PolicyRouter
from gateway.telemetry.metrics import MetricsRegistry
from gateway.telemetry.queue import LogQueue

log = get_logger("gateway.pipeline")


# =======================================================================
# request / result types
# =======================================================================
@dataclass
class ChatRequest:
    model: str = "auto"
    messages: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    user: str | None = None
    gateway_no_cache: bool = False

    def sampling(self, default_temperature: float) -> SamplingParams:
        return SamplingParams(
            temperature=(
                self.temperature if self.temperature is not None else default_temperature
            ),
            top_p=self.top_p if self.top_p is not None else 1.0,
            max_tokens=self.max_tokens,
            stop=tuple(self.stop or ()),
            response_format=self.response_format,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )


@dataclass
class Prepared:
    """Everything computed before any upstream call happens."""

    logical_model: str
    params: SamplingParams
    bypass: str | None
    precharge: int
    est_prompt_tokens: int
    exact_key: str
    scope: str
    system_hash: str
    tools_hash: str
    prompt_hash: str
    prompt_text: str
    request_ctx: dict[str, Any]
    exact_hit: ExactHit | None = None
    semantic_hit: SemanticHit | None = None
    similarity: float = 0.0

    @property
    def cache_status(self) -> str:
        if self.bypass:
            return "bypass"
        if self.exact_hit:
            return "exact_hit"
        if self.semantic_hit:
            return "semantic_hit"
        return "miss"


@dataclass
class PipelineResult:
    body: dict[str, Any]
    headers: dict[str, str]
    request_id: str


# =======================================================================
# pipeline
# =======================================================================
class RequestPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        exact: ExactCache,
        semantic: SemanticCache,
        ratelimit: RateLimiter,
        registry: ProviderRegistry,
        router: PolicyRouter,
        telemetry: LogQueue,
        metrics: MetricsRegistry,
        default_temperature: float = 0.2,
    ):
        self.settings = settings
        self.exact = exact
        self.semantic = semantic
        self.ratelimit = ratelimit
        self.registry = registry
        self.router = router
        self.telemetry = telemetry
        self.metrics = metrics
        self.default_temperature = default_temperature

    # ------------------------------------------------------------------
    # preparation (shared by streaming and non-streaming)
    # ------------------------------------------------------------------
    async def _prepare(
        self,
        req: ChatRequest,
        principal: Principal,
        request_id: str,
    ) -> Prepared:
        logical = (req.model or "auto").strip()
        principal.require_model(logical)

        params = req.sampling(self.default_temperature)
        messages = req.messages
        normalized = normalize_messages(messages)
        prompt_text = " ".join(m["content"] for m in normalized if m["role"] != "system")
        prompt_text = normalize_text(prompt_text)

        est_prompt = cost_mod.estimate_message_tokens(messages)
        precharge = cost_mod.precharge_tokens(
            estimated_prompt_tokens=est_prompt, max_tokens=req.max_tokens
        )

        # --- rate limit (pre-charge; refunded in every terminal path) ---------
        decision = await self.ratelimit.check(principal, precharge)
        if not decision.allowed:
            raise RateLimitError(
                f"Rate limit exceeded for api key (tokens). "
                f"Retry in {decision.retry_after}s.",
                retry_after=decision.retry_after,
            )

        inp = ExactKeyInput(messages=messages, sampling=params)
        exact_key = build_exact_key(principal.api_key_id, logical, inp)
        scope = f"{principal.api_key_id}:{logical}"

        prepared = Prepared(
            logical_model=logical,
            params=params,
            bypass=self._bypass_reason(req, params),
            precharge=precharge,
            est_prompt_tokens=est_prompt,
            exact_key=exact_key,
            scope=scope,
            system_hash=system_prompt_hash(messages),
            tools_hash=tools_hash(req.tools),
            prompt_hash=blake2b_hex(prompt_text),
            prompt_text=prompt_text,
            request_ctx=build_context(
                requested_model=logical,
                messages=messages,
                prompt_tokens=est_prompt,
                wants_json=bool(req.response_format),
                is_stream=req.stream,
            ),
        )

        if prepared.bypass:
            return prepared

        # --- exact cache -----------------------------------------------------
        prepared.exact_hit = await self.exact.get(
            exact_key, scope=scope, requested_max_tokens=req.max_tokens
        )
        if prepared.exact_hit:
            return prepared

        # --- semantic cache --------------------------------------------------
        prepared.semantic_hit, prepared.similarity = await self.semantic.get(
            prompt_text,
            api_key_id=principal.api_key_id,
            model=logical,
            system_hash=prepared.system_hash,
            tools_hash=prepared.tools_hash,
            requested_max_tokens=req.max_tokens,
        )
        return prepared

    def _bypass_reason(self, req: ChatRequest, params: SamplingParams) -> str | None:
        if req.gateway_no_cache:
            return "client_request"
        return bypass_reason(params, self.settings)

    # ------------------------------------------------------------------
    async def _settle(
        self,
        prepared: Prepared,
        principal: Principal,
        *,
        used_tokens: int,
    ) -> None:
        """Refund whatever the reservation didn't cover. Never raises."""
        refund = prepared.precharge - used_tokens
        if refund > 0:
            await self.ratelimit.refund(principal, refund)

    # ------------------------------------------------------------------
    def _route(self, prepared: Prepared) -> RouteDecision:
        ctx = prepared.request_ctx
        return self.router.route(
            RouteContext(
                requested_model=ctx["requested_model"],
                prompt_text=ctx["prompt_text"],
                system_text=ctx["system_text"],
                prompt_tokens=ctx["prompt_tokens"],
                has_code=ctx["has_code"],
                wants_json=ctx["wants_json"],
                is_stream=ctx["is_stream"],
                features=ctx["features"],
            )
        )

    def _resolve_route(self, decision: RouteDecision) -> Route:
        try:
            return self.registry.resolve(decision.logical_model)
        except KeyError:
            # Router asked for a logical model we don't have — fall back to `auto`.
            return self.registry.resolve("auto")

    # ------------------------------------------------------------------
    # cache-hit path
    # ------------------------------------------------------------------
    def _cached_body(
        self,
        prepared: Prepared,
        request_id: str,
        source: str,
        *,
        similarity: float | None = None,
    ) -> dict[str, Any]:
        if prepared.exact_hit:
            stored = dict(prepared.exact_hit.response)
        else:
            assert prepared.semantic_hit is not None
            stored = dict(prepared.semantic_hit.entry.response)

        stored["id"] = f"chatcmpl-{request_id}"
        stored["model"] = prepared.logical_model
        stored["gateway"] = {
            "cache": source,
            "similarity": similarity,
            "upstream_model": stored.get("gateway", {}).get("upstream_model"),
            "route_reason": "cache",
            "latency_ms": 0,
        }
        return stored

    # ------------------------------------------------------------------
    # non-streaming
    # ------------------------------------------------------------------
    async def complete(
        self,
        req: ChatRequest,
        principal: Principal,
        request_id: str,
    ) -> PipelineResult:
        started = time.perf_counter()
        prepared = await self._prepare(req, principal, request_id)
        status = prepared.cache_status

        try:
            if prepared.cache_status == "exact_hit" and prepared.exact_hit:
                body = self._cached_body(prepared, request_id, "EXACT_HIT")
                await self._settle(prepared, principal, used_tokens=0)
                latency = (time.perf_counter() - started) * 1000
                self._emit(prepared, principal, request_id,
                           status="exact_hit", model_served=prepared.logical_model,
                           provider="cache", usage=prepared.exact_hit.usage,
                           latency_ms=latency, http_status=200, cached=True,
                           cost=cost_mod.compute_cost(prepared.logical_model, 0, 0))
                return PipelineResult(body, self._headers("EXACT_HIT", None, latency, None), request_id)

            if prepared.cache_status == "semantic_hit" and prepared.semantic_hit:
                entry = prepared.semantic_hit.entry
                body = self._cached_body(
                    prepared, request_id, "SEMANTIC_HIT",
                    similarity=prepared.semantic_hit.similarity,
                )
                await self._settle(prepared, principal, used_tokens=0)
                latency = (time.perf_counter() - started) * 1000
                usage = entry.usage
                self._emit(prepared, principal, request_id,
                           status="semantic_hit", model_served=entry.model,
                           provider="cache", usage=usage, latency_ms=latency,
                           http_status=200, cached=True,
                           similarity=prepared.semantic_hit.similarity,
                           cost=cost_mod.compute_cost(
                               entry.model, int(usage.get("prompt_tokens", 0)),
                               int(usage.get("completion_tokens", 0))))
                return PipelineResult(
                    body,
                    self._headers("SEMANTIC_HIT", prepared.semantic_hit.similarity,
                                  latency, None),
                    request_id,
                )

            # --- miss: go to the upstream ------------------------------------
            decision = self._route(prepared)
            route = self._resolve_route(decision)
            breaker = self.registry.breakers.get(route.breaker_key)
            if not breaker.allow():
                await self._settle(prepared, principal, used_tokens=0)
                raise NoProviderAvailableError(
                    f"Upstream {route.breaker_key} is temporarily unavailable (circuit open)."
                )

            from gateway.providers.resilience import with_retry

            upstream_started = time.perf_counter()
            try:
                result = await with_retry(
                    lambda: route.provider.chat(
                        model=route.upstream_model,
                        messages=req.messages,
                        params=prepared.params,
                        timeout=self.settings.upstream_timeout_s,
                    ),
                    settings=self.settings,
                    label=route.breaker_key,
                )
                breaker.record_success()
            except ProviderError as exc:
                breaker.record_failure(exc.message)
                await self._settle(prepared, principal, used_tokens=0)
                self._emit_error(prepared, principal, request_id, decision, route, exc,
                                 (time.perf_counter() - started) * 1000)
                raise _translate(exc) from exc

            upstream_ms = (time.perf_counter() - upstream_started) * 1000
            latency = (time.perf_counter() - started) * 1000

            usage = {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            }
            if usage["total_tokens"] == 0:
                usage = _estimate_usage(result.text, prepared)

            await self._write_caches(prepared, req, principal, result.raw or {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": 0,
                "model": prepared.logical_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }],
                "usage": usage,
            }, usage)

            await self._settle(prepared, principal,
                               used_tokens=usage["prompt_tokens"] + usage["completion_tokens"])

            cost = cost_mod.compute_cost(
                route.upstream_model, usage["prompt_tokens"], usage["completion_tokens"]
            )
            body = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": prepared.logical_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text,
                                **({"tool_calls": result.tool_calls} if result.tool_calls else {})},
                    "finish_reason": result.finish_reason,
                }],
                "usage": usage,
                "gateway": {
                    "cache": "MISS",
                    "similarity": prepared.similarity or None,
                    "upstream_model": result.model or route.upstream_model,
                    "route_reason": decision.reason,
                    "route_confidence": decision.confidence,
                    "latency_ms": round(latency, 2),
                    **cost.as_dict(),
                },
            }

            self._emit(prepared, principal, request_id, status="miss",
                       model_served=result.model or route.upstream_model,
                       provider=route.provider_name, usage=usage, latency_ms=latency,
                       upstream_ms=upstream_ms, http_status=200, cached=False,
                       cost=cost, decision=decision)
            return PipelineResult(
                body,
                self._headers("MISS", None, latency,
                              result.model or route.upstream_model),
                request_id,
            )

        except GatewayError:
            raise
        except Exception as exc:  # pragma: no cover - safety net
            await self._settle(prepared, principal, used_tokens=0)
            log.exception("pipeline_unexpected_error", error=str(exc))
            raise UpstreamError(f"Gateway failed to complete the request: {exc}") from exc

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------
    async def stream(
        self,
        req: ChatRequest,
        principal: Principal,
        request_id: str,
    ) -> AsyncIterator[bytes]:
        started = time.perf_counter()
        prepared = await self._prepare(req, principal, request_id)

        # --- cache hit: replay as a single chunk ------------------------------
        if prepared.cache_status == "exact_hit" and prepared.exact_hit:
            body = self._cached_body(prepared, request_id, "EXACT_HIT")
            content = _content_of(body)
            latency = (time.perf_counter() - started) * 1000
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model, delta=content))
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model,
                                       finish_reason="stop"))
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model, include_usage=True,
                                       usage=_usage_of(body)))
            yield sse_done()
            await self._settle(prepared, principal, used_tokens=0)
            self._emit(prepared, principal, request_id, status="exact_hit",
                       model_served=prepared.logical_model, provider="cache",
                       usage=_usage_of(body), latency_ms=latency, http_status=200,
                       cached=True, cost=cost_mod.compute_cost(prepared.logical_model, 0, 0))
            return

        if prepared.cache_status == "semantic_hit" and prepared.semantic_hit:
            entry = prepared.semantic_hit.entry
            sim = prepared.semantic_hit.similarity
            body = self._cached_body(prepared, request_id, "SEMANTIC_HIT", similarity=sim)
            content = _content_of(body)
            latency = (time.perf_counter() - started) * 1000
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model, delta=content))
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model,
                                       finish_reason="stop"))
            yield sse_event(make_chunk(request_id=request_id,
                                       model=prepared.logical_model, include_usage=True,
                                       usage=entry.usage or _usage_of(body)))
            yield sse_done()
            await self._settle(prepared, principal, used_tokens=0)
            self._emit(prepared, principal, request_id, status="semantic_hit",
                       model_served=entry.model, provider="cache",
                       usage=entry.usage or _usage_of(body), latency_ms=latency,
                       http_status=200, cached=True, similarity=sim,
                       cost=cost_mod.compute_cost(entry.model, 0, 0))
            return

        # --- miss -------------------------------------------------------------
        decision = self._route(prepared)
        route = self._resolve_route(decision)
        breaker = self.registry.breakers.get(route.breaker_key)
        if not breaker.allow():
            await self._settle(prepared, principal, used_tokens=0)
            raise NoProviderAvailableError(
                f"Upstream {route.breaker_key} is temporarily unavailable (circuit open)."
            )

        acc = StreamAccumulator(model=route.upstream_model)
        upstream_started = time.perf_counter()
        first_byte_ms: float | None = None

        try:
            async for chunk in route.provider.stream(
                model=route.upstream_model,
                messages=req.messages,
                params=prepared.params,
                timeout=self.settings.upstream_timeout_s,
                idle_timeout=self.settings.upstream_idle_timeout_s,
            ):
                if first_byte_ms is None and chunk.delta:
                    first_byte_ms = (time.perf_counter() - started) * 1000
                acc.add(chunk.delta, chunk.finish_reason, chunk.usage, chunk.tool_calls)

                if chunk.delta:
                    yield sse_event(make_chunk(request_id=request_id,
                                               model=prepared.logical_model,
                                               delta=chunk.delta))
                if chunk.finish_reason:
                    yield sse_event(make_chunk(request_id=request_id,
                                               model=prepared.logical_model,
                                               finish_reason=chunk.finish_reason))
            breaker.record_success()
        except ProviderError as exc:
            breaker.record_failure(exc.message)
            await self._settle(prepared, principal, used_tokens=0)
            latency = (time.perf_counter() - started) * 1000
            self._emit_error(prepared, principal, request_id, decision, route, exc, latency)
            # NEVER retry here: bytes already reached the client and a second pass would
            # duplicate content. Emit the error as an SSE event instead of raising.
            yield sse_event({"error": _error_payload(exc, request_id)})
            yield sse_done()
            return

        upstream_ms = (time.perf_counter() - upstream_started) * 1000
        latency = (time.perf_counter() - started) * 1000

        usage = acc.resolve_usage(
            estimated_prompt=prepared.est_prompt_tokens, messages=req.messages
        )

        assembled = assemble_response(acc, request_id=request_id,
                                      logical_model=prepared.logical_model)
        await self._write_caches(prepared, req, principal, assembled, usage)
        await self._settle(prepared, principal,
                           used_tokens=usage["prompt_tokens"] + usage["completion_tokens"])

        # Final chunk carrying usage — clients (and our own accounting) need it.
        yield sse_event(make_chunk(request_id=request_id, model=prepared.logical_model,
                                   include_usage=True, usage=usage))
        yield sse_done()

        cost = cost_mod.compute_cost(route.upstream_model, usage["prompt_tokens"],
                                     usage["completion_tokens"])
        self._emit(prepared, principal, request_id, status="miss",
                   model_served=acc.model or route.upstream_model,
                   provider=route.provider_name, usage=usage, latency_ms=latency,
                   upstream_ms=upstream_ms, ttfb_ms=first_byte_ms, http_status=200,
                   cached=False, cost=cost, decision=decision)

    # ------------------------------------------------------------------
    # cache writes
    # ------------------------------------------------------------------
    async def _write_caches(
        self,
        prepared: Prepared,
        req: ChatRequest,
        principal: Principal,
        body: dict[str, Any],
        usage: dict[str, int],
    ) -> None:
        if prepared.bypass:
            return
        annotated = dict(body)
        annotated["gateway"] = {
            **(annotated.get("gateway") or {}),
            "upstream_model": annotated.get("model", prepared.logical_model),
        }
        await self.exact.set(
            prepared.exact_key,
            scope=prepared.scope,
            response=annotated,
            usage=usage,
        )
        await self.semantic.set(
            prepared.prompt_text,
            api_key_id=principal.api_key_id,
            model=prepared.logical_model,
            system_hash=prepared.system_hash,
            tools_hash=prepared.tools_hash,
            prompt_hash=prepared.prompt_hash,
            response=annotated,
            usage=usage,
            max_tokens=req.max_tokens,
        )

    # ------------------------------------------------------------------
    # headers + telemetry
    # ------------------------------------------------------------------
    @staticmethod
    def _headers(
        cache: str,
        similarity: float | None,
        latency_ms: float,
        upstream: str | None,
    ) -> dict[str, str]:
        headers = {
            "x-request-id-echo": "",  # replaced by the route
            "x-gateway-cache": cache,
            "x-gateway-latency-ms": str(int(round(latency_ms))),
            "Cache-Control": "no-store",
        }
        if similarity is not None:
            headers["x-gateway-similarity"] = f"{similarity:.4f}"
        if upstream:
            headers["x-gateway-model-upstream"] = upstream
        return headers

    def _emit(
        self,
        prepared: Prepared,
        principal: Principal,
        request_id: str,
        *,
        status: str,
        model_served: str,
        provider: str,
        usage: dict[str, Any],
        latency_ms: float,
        http_status: int,
        cached: bool,
        cost: cost_mod.CostResult,
        decision: RouteDecision | None = None,
        upstream_ms: float | None = None,
        ttfb_ms: float | None = None,
        similarity: float | None = None,
    ) -> None:
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        self.telemetry.emit({
            "request_id": request_id,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "model_requested": prepared.logical_model,
            "model_served": model_served,
            "upstream_provider": provider,
            "route_reason": decision.reason if decision else "cache",
            "route_confidence": decision.confidence if decision else None,
            "cache_status": status,
            "similarity": similarity,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": prompt_t + completion_t,
            "tokens_from_cache": (prompt_t + completion_t) if cached else 0,
            "cost_usd": cost.cost_usd,
            "cached_cost_usd": 0.0,
            "cost_basis": cost.basis,
            "price_version": cost.price_version,
            "latency_ms": int(round(latency_ms)),
            "ttfb_ms": int(round(ttfb_ms)) if ttfb_ms else None,
            "upstream_latency_ms": int(round(upstream_ms)) if upstream_ms else None,
            "queue_wait_ms": None,
            "status_code": http_status,
            "error_type": None,
            "cached": cached,
            "prompt_hash": prepared.prompt_hash,
            "system_prompt_hash": prepared.system_hash,
            "prompt_preview": prepared.prompt_text[: self.settings.prompt_preview_len],
            "metadata_": {
                "bypass": prepared.bypass,
                "router": decision.router if decision else "cache",
                "route_detail": decision.detail if decision else "",
            },
        })
        self.metrics.incr("requests", status=status)
        self.metrics.observe("gateway_latency", latency_ms, status=status)
        if upstream_ms is not None:
            self.metrics.observe("upstream_latency", upstream_ms)
        if ttfb_ms is not None:
            self.metrics.observe("ttfb", ttfb_ms)

    def _emit_error(
        self,
        prepared: Prepared,
        principal: Principal,
        request_id: str,
        decision: RouteDecision | None,
        route: Route,
        exc: ProviderError,
        latency_ms: float,
    ) -> None:
        self.telemetry.emit({
            "request_id": request_id,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "model_requested": prepared.logical_model,
            "model_served": route.upstream_model,
            "upstream_provider": route.provider_name,
            "route_reason": decision.reason if decision else "error",
            "route_confidence": decision.confidence if decision else None,
            "cache_status": "miss",
            "similarity": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_from_cache": 0,
            "cost_usd": 0.0,
            "cached_cost_usd": 0.0,
            "cost_basis": "local_inference",
            "price_version": cost_mod.PRICE_VERSION,
            "latency_ms": int(round(latency_ms)),
            "ttfb_ms": None,
            "upstream_latency_ms": None,
            "queue_wait_ms": None,
            "status_code": exc.status_code,
            "error_type": "upstream_error",
            "cached": False,
            "prompt_hash": prepared.prompt_hash,
            "system_prompt_hash": prepared.system_hash,
            "prompt_preview": prepared.prompt_text[: self.settings.prompt_preview_len],
            "metadata_": {"upstream_status": exc.upstream_status,
                          "provider": exc.provider},
        })
        self.metrics.incr("requests", status="error")
        self.metrics.observe("gateway_latency", latency_ms, status="error")


# =======================================================================
# helpers
# =======================================================================
def _translate(exc: ProviderError) -> GatewayError:
    from gateway.api.errors import (
        AuthenticationError,
        InvalidRequestError,
        RateLimitError,
        UpstreamError,
        UpstreamTimeoutError,
    )

    if exc.status_code == 429:
        return RateLimitError(exc.message, retry_after=1)
    if exc.status_code == 401:
        return AuthenticationError(exc.message, code="upstream_auth_error")
    if exc.status_code in (400, 404, 413):
        return InvalidRequestError(exc.message, code="upstream_rejected_request")
    if exc.status_code == 504:
        return UpstreamTimeoutError(exc.message)
    return UpstreamError(exc.message)


def _estimate_usage(text: str, prepared: Prepared) -> dict[str, int]:
    completion = cost_mod.estimate_tokens(text)
    prompt = prepared.est_prompt_tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _content_of(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    return ((choices[0].get("message") or {}).get("content")) or ""


def _usage_of(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _error_payload(exc: ProviderError, request_id: str) -> dict[str, Any]:
    return {
        "message": exc.message,
        "type": "api_error",
        "param": None,
        "code": "upstream_error",
        "request_id": request_id,
    }


def new_request_id() -> str:
    return str(uuid.uuid4())
