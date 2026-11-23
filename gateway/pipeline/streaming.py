"""SSE handling: re-emit chunks to the client and capture usage.

Two things this module is responsible for:

* **Correct framing.** Each event must be ``data: {...}\\n\\n`` and the stream must end
  with ``data: [DONE]\\n\\n``, or clients hang waiting for more.
* **Usage capture.** Usage arrives in a final chunk — and only if the upstream was asked
  for ``stream_options.include_usage``. If it never arrives we must fall back to
  estimating rather than logging zeros, because zero-token telemetry makes every cost and
  savings figure a lie.

``assemble_response`` converts a completed stream into the single payload the caches store,
so a streamed answer is cacheable exactly like a non-streamed one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gateway.pipeline.cost import estimate_tokens

DONE_SENTINEL = "data: [DONE]\n\n"


def sse_event(payload: dict[str, Any]) -> bytes:
    """Encode one SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return DONE_SENTINEL.encode("utf-8")


@dataclass
class StreamAccumulator:
    """Collects a stream's chunks so it can be stored, cached and logged."""

    content: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    chunks: int = 0

    def add(self, delta: str, finish_reason: str | None, usage: dict | None,
            tool_calls: list[dict] | None = None) -> None:
        if delta:
            self.content += delta
        if finish_reason:
            self.finish_reason = finish_reason
        if usage:
            self.usage.update(usage)
        if tool_calls:
            self.tool_calls.extend(tool_calls)
        self.chunks += 1

    @property
    def has_usage(self) -> bool:
        return bool(self.usage.get("prompt_tokens") or self.usage.get("completion_tokens"))

    def resolve_usage(self, *, estimated_prompt: int, messages: list[dict]) -> dict[str, int]:
        """Use upstream usage if present, otherwise estimate.

        Estimating is the honest fallback: better a labelled approximation than a
        zero that corrupts every downstream statistic.
        """
        if self.has_usage:
            return {
                "prompt_tokens": int(self.usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(self.usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(self.usage.get("total_tokens", 0) or 0),
            }
        completion = estimate_tokens(self.content)
        prompt = estimated_prompt or estimate_tokens(
            " ".join(str(m.get("content") or "") for m in messages)
        )
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "estimated": 1,  # type: ignore[dict-item]
        }


def make_chunk(
    *,
    request_id: str,
    model: str,
    delta: str = "",
    finish_reason: str | None = None,
    include_usage: bool = False,
    usage: dict[str, int] | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Build an OpenAI-shaped streaming chunk."""
    chunk: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        chunk["usage"] = usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return chunk


def assemble_response(
    acc: StreamAccumulator,
    *,
    request_id: str,
    logical_model: str,
) -> dict[str, Any]:
    """Turn a completed stream into the same payload a non-streaming call returns.

    This is what makes streamed responses cacheable.
    """
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": 0,
        "model": logical_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": acc.content,
                    **({"tool_calls": acc.tool_calls} if acc.tool_calls else {}),
                },
                "finish_reason": acc.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(acc.usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(acc.usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(acc.usage.get("total_tokens", 0) or 0),
        },
    }
