"""Shared test fixtures.

Design goals:

* **Hermetic.** No Redis, no Postgres, no Ollama, no model downloads. Tests must run in
  CI on a cold machine.
* **Fast.** An in-memory SQLite DB, `fakeredis` (with Lua, so the real token-bucket script
  executes), and a fake embedder instead of ONNX.
* **Real code paths.** The app is built by the production `create_app()` and run through
  its lifespan — only the three external boundaries (Redis, upstream provider, embedder)
  are substituted.

The fake embedder is deterministic and word-overlap based, so cosine similarity behaves
predictably: identical prompts score 1.0, disjoint prompts score 0.0.
"""

from __future__ import annotations

import os
import re
import zlib
from typing import Any, AsyncIterator

import fakeredis.aioredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Must be set before anything imports gateway.config.
os.environ["APP_ENV"] = "test"
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("API_KEY_PEPPER", "test-pepper")

from gateway.config import clear_settings_cache  # noqa: E402
from gateway.providers.base import (  # noqa: E402
    ChatResult,
    Provider,
    ProviderError,
    StreamChunk,
)

ADMIN_HEADERS = {"Authorization": "Bearer test-admin-token"}


# =======================================================================
# fakes
# =======================================================================
class FakeEmbedder:
    """Deterministic, word-overlap embedder. No network, no model file."""

    name = "fake"
    dim = 32

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    async def health(self) -> bool:
        return True

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[zlib.crc32(token.encode("utf-8")) % 32] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


class FakeProvider(Provider):
    """Canned upstream. Records every call so tests can assert cache-hit behaviour."""

    name = "fake"
    #: Shared across instances so tests can inspect after the fact.
    calls: list[dict[str, Any]] = []
    #: Optional per-test override: list of responses to pop, else a default echo.
    script: list[str] = []
    fail_with: ProviderError | None = None

    def __init__(self, settings: Any = None, *, base_url: str = "", name: str = "fake", **_: Any):
        self.name = name

    # -- helpers -------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.script = []
        cls.fail_with = None

    def _reply(self, messages: list[dict[str, Any]]) -> str:
        if FakeProvider.script:
            return FakeProvider.script.pop(0)
        last = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                last = str(m.get("content") or "")
                break
        return f"reply to: {last}"

    # -- Provider API --------------------------------------------------
    async def chat(self, *, model, messages, params, timeout) -> ChatResult:
        if FakeProvider.fail_with is not None:
            raise FakeProvider.fail_with
        text = self._reply(messages)
        FakeProvider.calls.append(
            {"model": model, "messages": messages, "stream": False}
        )
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        return ChatResult(
            text=text,
            finish_reason="stop",
            model=model,
            usage=usage,
            raw={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}
                ],
                "usage": usage,
            },
        )

    async def stream(self, *, model, messages, params, timeout, idle_timeout) -> AsyncIterator[StreamChunk]:
        if FakeProvider.fail_with is not None:
            raise FakeProvider.fail_with
        text = self._reply(messages)
        FakeProvider.calls.append({"model": model, "messages": messages, "stream": True})
        # Emit the real-world shape: content chunks, then a usage-only chunk.
        for i in range(0, len(text), 5):
            yield StreamChunk(delta=text[i : i + 5], model=model)
        yield StreamChunk(finish_reason="stop", model=model)
        yield StreamChunk(
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            model=model,
        )

    async def health(self) -> bool:
        return True


# =======================================================================
# fixtures
# =======================================================================
@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeProvider.reset()
    yield
    FakeProvider.reset()


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


@pytest.fixture
def app(monkeypatch, fake_redis):
    """Production app factory with the three external boundaries substituted."""
    clear_settings_cache()

    monkeypatch.setattr(
        "gateway.main.Redis.from_url", lambda *a, **k: fake_redis
    )
    monkeypatch.setattr(
        "gateway.main.build_embedder", lambda settings: FakeEmbedder()
    )
    monkeypatch.setattr(
        "gateway.providers.registry.OpenAICompatProvider", FakeProvider
    )

    from gateway.main import create_app

    application = create_app()
    yield application
    clear_settings_cache()


@pytest.fixture
def client(app):
    """TestClient that runs the lifespan (so caches, DB and writer start and stop)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_key(client) -> str:
    """A freshly minted API key, created through the real admin endpoint."""
    resp = client.post(
        "/admin/api-keys",
        json={"label": "test-key", "email": "tester@example.com"},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["key"]


@pytest.fixture
def auth_headers(api_key) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


# =======================================================================
# helpers
# =======================================================================
def chat_payload(content: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "auto",
        "messages": [{"role": "user", "content": content}],
    }
    body.update(extra)
    return body


def cache_header(response) -> str:
    return response.headers.get("x-gateway-cache", "")
