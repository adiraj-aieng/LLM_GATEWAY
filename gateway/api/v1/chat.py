"""``POST /v1/chat/completions`` — the hot path.

OpenAI compatibility is enforced at the boundary, and anything we *don't* support is
rejected with an explicit 400 rather than silently dropped. Silent divergence from the
OpenAI contract is how clients end up debugging a proxy instead of their own code.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api import deps
from gateway.api.errors import InvalidRequestError
from gateway.core.auth import Principal
from gateway.db.session import get_session
from gateway.pipeline.request_pipeline import ChatRequest, RequestPipeline

router = APIRouter(tags=["chat"])


# =======================================================================
# schema
# =======================================================================
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None


class GatewayOptions(BaseModel):
    """Gateway-specific extension. Ignored by stock OpenAI clients."""

    model_config = ConfigDict(extra="ignore")

    no_cache: bool = False


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "auto"
    messages: list[Message] = Field(min_length=1)
    stream: bool = False

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None
    gateway: GatewayOptions | None = None

    # --- explicitly rejected: better a clear 400 than silent divergence ---
    n: int | None = Field(default=None, ge=1)
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _reject_unsupported(self) -> "ChatBody":
        if self.n is not None and self.n != 1:
            raise ValueError(
                "`n` must be 1: the gateway caches responses and accounts cost "
                "per completion, so n>1 is not supported."
            )
        if self.logprobs or self.top_logprobs is not None:
            raise ValueError(
                "`logprobs` is not supported: cached responses do not preserve logprobs."
            )
        if self.presence_penalty is not None or self.frequency_penalty is not None:
            raise ValueError(
                "`presence_penalty`/`frequency_penalty` are not supported: they are "
                "incompatible with cached responses."
            )
        if not self.messages:
            raise ValueError("`messages` must contain at least one message")
        if self.temperature is not None and self.temperature > 2.0:
            raise ValueError("`temperature` must be <= 2")
        return self

    @field_validator("model")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("`model` must be a non-empty string")
        return v.strip()

    # ------------------------------------------------------------------
    def to_pipeline(self) -> ChatRequest:
        stop: tuple[str, ...] = ()
        if isinstance(self.stop, str):
            stop = (self.stop,)
        elif isinstance(self.stop, list):
            stop = tuple(s for s in self.stop if isinstance(s, str))

        return ChatRequest(
            model=self.model,
            messages=[m.model_dump(exclude_none=True) for m in self.messages],
            stream=self.stream,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop,
            response_format=self.response_format,
            tools=self.tools,
            tool_choice=self.tool_choice,
            user=self.user,
            gateway_no_cache=bool(self.gateway and self.gateway.no_cache),
        )


# =======================================================================
# endpoint
# =======================================================================
@router.post("/chat/completions")
async def chat_completions(
    body: ChatBody,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(deps.require_api_key)],
    pipeline: Annotated[RequestPipeline, Depends(deps.get_pipeline)],
):
    req = body.to_pipeline()
    request_id = deps.request_id_of(request)

    if body.stream:
        return StreamingResponse(
            pipeline.stream(req, principal, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Required behind nginx, otherwise chunks are buffered and TTFT looks
                # perfect while the user stares at a blank screen.
                "X-Accel-Buffering": "no",
            },
        )

    result = await pipeline.complete(req, principal, request_id)
    headers = dict(result.headers)
    headers.pop("x-request-id-echo", None)
    headers["x-request-id"] = request_id
    if principal.rate_limit_tokens_min:
        headers["x-ratelimit-limit"] = str(principal.rate_limit_tokens_min)
    return JSONResponse(content=result.body, headers=headers)
