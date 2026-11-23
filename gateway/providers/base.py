"""Provider abstraction.

One interface, several wire formats. :class:`OpenAICompatProvider` covers Ollama, OpenAI,
Groq and Together — they all speak the same protocol and differ only in ``base_url`` and
model names, so they share a single adapter. Anthropic and Gemini need their own files
because their request/response shapes differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from gateway.cache.keys import SamplingParams


@dataclass
class ChatResult:
    """A completed, non-streaming response reduced to what the gateway needs."""

    text: str
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0) or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0) or 0)


@dataclass
class StreamChunk:
    """One SSE event, already decoded."""

    delta: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""


class ProviderError(Exception):
    """Raised by providers; translated to the OpenAI envelope upstream."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        retryable: bool = False,
        provider: str = "",
        upstream_status: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.provider = provider
        self.upstream_status = upstream_status


class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        timeout: float,
    ) -> Any:
        """Return an awaitable producing :class:`ChatResult`."""

    @abstractmethod
    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[StreamChunk]:
        """Async iterator of :class:`StreamChunk`."""

    @abstractmethod
    async def health(self) -> bool: ...

    async def list_models(self) -> list[str]:
        return []

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
