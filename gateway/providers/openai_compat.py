"""OpenAI-compatible upstream adapter.

One class covers **Ollama, OpenAI, Groq and Together** — same wire format, different
``base_url`` and model names. That is why this is the default adapter and why adding a
hosted provider later is a config change rather than a new file.

The streaming part has two traps this module handles explicitly:

1. **Usage is not in an SSE stream** unless ``stream_options.include_usage`` is set, and
   not every server accepts that field. We request it, and fall back to omitting it if the
   upstream rejects it with a 400 — otherwise every streamed request logs ``0`` tokens and
   cost accounting silently becomes fiction.
2. **Idle vs total timeout.** A generation may legitimately run for minutes while a dead
   connection produces nothing, so the read timeout is an *idle* timeout on each chunk,
   not a cap on the whole stream.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from gateway.cache.keys import SamplingParams
from gateway.config import Settings
from gateway.logging import get_logger
from gateway.providers.base import (
    ChatResult,
    Provider,
    ProviderError,
    StreamChunk,
)
from gateway.providers.resilience import RETRYABLE_STATUSES

log = get_logger("gateway.provider.openai_compat")

_DONE = "[DONE]"


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, settings: Settings, *, base_url: str, name: str | None = None):
        self._settings = settings
        self._base = base_url.rstrip("/")
        if name:
            self.name = name
        # ONE client for the process's lifetime. A client per request leaks sockets and
        # is the most common cause of bad p99 in Python HTTP proxies.
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(
                settings.upstream_timeout_s,
                connect=10.0,
                # read = per-chunk idle timeout while streaming
                read=settings.upstream_idle_timeout_s,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
            ),
            headers={"Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------
    def _headers(self, params: SamplingParams) -> dict[str, str]:
        # `params` reserved for per-request upstream headers (e.g. provider API keys).
        return {}

    def _build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        stream: bool,
        include_usage: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.max_tokens is not None:
            body["max_tokens"] = params.max_tokens
        if params.stop:
            body["stop"] = list(params.stop)
        if params.response_format:
            body["response_format"] = params.response_format
        if params.tools:
            body["tools"] = params.tools
        if params.tool_choice is not None:
            body["tool_choice"] = params.tool_choice
        if stream:
            body["stream"] = True
            if include_usage:
                body["stream_options"] = {"include_usage": True}
        return body

    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        timeout: float,
    ) -> ChatResult:
        body = self._build_body(
            model=model,
            messages=messages,
            params=params,
            stream=False,
            include_usage=False,
        )
        try:
            resp = await self._client.post(
                "/v1/chat/completions",
                json=body,
                timeout=timeout,
                headers=self._headers(params),
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Upstream timed out after {timeout}s.",
                status_code=504,
                retryable=True,
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Could not reach upstream: {exc}",
                status_code=502,
                retryable=True,
                provider=self.name,
            ) from exc

        if resp.status_code >= 400:
            raise self._to_error(resp)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ChatResult(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            model=data.get("model", model),
            usage=data.get("usage") or {},
            tool_calls=message.get("tool_calls") or [],
            raw=data,
        )

    # ------------------------------------------------------------------
    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[StreamChunk]:
        """Yield decoded SSE events.

        Attempts to request usage in the stream first; if the upstream answers 400 because
        it doesn't understand ``stream_options``, retries once without it.
        """
        include_usage = True
        attempt = 0
        while True:
            try:
                async for chunk in self._stream_once(
                    model=model,
                    messages=messages,
                    params=params,
                    timeout=timeout,
                    idle_timeout=idle_timeout,
                    include_usage=include_usage,
                ):
                    yield chunk
                return
            except ProviderError as exc:
                if (
                    include_usage
                    and exc.upstream_status == 400
                    and attempt == 0
                ):
                    # Upstream doesn't accept `stream_options` — retry without it.
                    log.debug("stream_usage_fallback", provider=self.name)
                    include_usage = False
                    attempt += 1
                    continue
                raise

    async def _stream_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        timeout: float,
        idle_timeout: float,
        include_usage: bool,
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(
            model=model,
            messages=messages,
            params=params,
            stream=True,
            include_usage=include_usage,
        )
        read_timeout = min(idle_timeout, timeout)

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                timeout=httpx.Timeout(
                    timeout, connect=10.0, read=read_timeout, write=30.0, pool=10.0
                ),
                headers=self._headers(params),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._to_error(resp)

                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    line = raw.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == _DONE:
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        log.debug("stream_bad_json", provider=self.name)
                        continue

                    choice = (data.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    usage = data.get("usage")

                    yield StreamChunk(
                        delta=delta.get("content") or "",
                        finish_reason=choice.get("finish_reason"),
                        usage=usage if usage else None,
                        tool_calls=delta.get("tool_calls") or [],
                        model=data.get("model", model),
                    )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Upstream stream idle for {read_timeout}s.",
                status_code=504,
                retryable=True,
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Upstream stream error: {exc}",
                status_code=502,
                retryable=True,
                provider=self.name,
            ) from exc

    # ------------------------------------------------------------------
    def _to_error(self, resp: httpx.Response) -> ProviderError:
        status = resp.status_code
        try:
            body = resp.json()
            message = (body.get("error") or {}).get("message") or resp.text[:500]
        except Exception:
            message = resp.text[:500]
        return ProviderError(
            f"[{self.name}] {message}",
            status_code=502 if status >= 500 else (429 if status == 429 else status),
            retryable=status in RETRYABLE_STATUSES,
            provider=self.name,
            upstream_status=status,
        )

    # ------------------------------------------------------------------
    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            return resp.status_code < 500
        except Exception:  # pragma: no cover
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            if resp.status_code >= 400:
                return []
            return [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
        except Exception:  # pragma: no cover
            return []

    async def aclose(self) -> None:
        await self._client.aclose()
