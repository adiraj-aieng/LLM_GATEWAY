"""``POST /v1/embeddings``.

Same interface as OpenAI's, backed by whichever embedder the semantic cache uses — so a
client can generate vectors for its own use without a second model dependency, and the
gateway's cache and its public embeddings endpoint are guaranteed to agree.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from gateway.api import deps
from gateway.core.auth import Principal

router = APIRouter(tags=["embeddings"])


class EmbedBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "text-embedding-nomic"
    input: str | list[str] = Field(min_length=1)
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None


@router.post("/embeddings")
async def create_embeddings(
    body: EmbedBody,
    request: Request,
    principal: Annotated[Principal, Depends(deps.require_api_key)],
):
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    if not texts:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "`input` must contain at least one string.",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "invalid_body",
                }
            },
        )
    # Guard against an accidental OOM: each vector is dim × 4 bytes, and the model call
    # is synchronous work we're pushing to a thread.
    if len(texts) > 256:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Batch size limited to 256 inputs per request.",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "batch_too_large",
                }
            },
        )

    embedder = request.app.state.embedder
    vectors = await embedder.embed(texts)

    data = [
        {"object": "embedding", "index": i, "embedding": vec.astype(float).tolist()}
        for i, vec in enumerate(vectors)
    ]
    usage = {
        "prompt_tokens": int(sum(len(t) for t in texts) / 4) or 1,
        "total_tokens": int(sum(len(t) for t in texts) / 4) or 1,
    }
    return {
        "object": "list",
        "data": data,
        "model": body.model,
        "usage": usage,
        "gateway": {"embedder": embedder.name, "dim": embedder.dim},
    }
