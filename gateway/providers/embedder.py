"""Embedding providers for the semantic cache.

Two implementations behind one interface:

``FastEmbedEmbedder`` (default)
    ONNX Runtime, **in-process, no PyTorch**. This matters because the semantic cache sits
    on the latency-critical path: an in-process model costs ~1-3 ms, whereas round-tripping
    to Ollama for every lookup would make cache hits slower than they should be, and
    would drag in a ~2.5 GB torch dependency for no benefit.

``OllamaEmbedder``
    Fallback / cross-check. Same interface, one HTTP hop.

Both return ``float32`` row vectors already shaped ``(n, dim)``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
import numpy as np

from gateway.config import Settings
from gateway.logging import get_logger

log = get_logger("gateway.embed")

#: ONNX model used by fastembed. 384-dim, ~90 MB, no torch.
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array."""

    async def embed_one(self, text: str) -> np.ndarray:
        arr = await self.embed([text])
        return arr[0]

    async def health(self) -> bool:
        return True


class FastEmbedEmbedder(Embedder):
    """In-process ONNX embedder. Blocking calls are pushed to a worker thread."""

    name = "fastembed"
    dim = 0

    def __init__(self, model_name: str = FASTEMBED_MODEL):
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                def _load():
                    from fastembed import TextEmbedding  # lazy: ~1s + model download

                    return TextEmbedding(model_name=self._model_name)

                log.info("embedder_loading", model=self._model_name)
                self._model = await asyncio.to_thread(_load)
                # Probe once so `dim` is known before the first cache write.
                probe = await asyncio.to_thread(
                    lambda: list(self._model.embed(["dimension probe"]))[0]
                )
                self.dim = int(np.asarray(probe).shape[-1])
                log.info("embedder_ready", model=self._model_name, dim=self.dim)
        return self._model

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        model = await self._ensure_loaded()

        def _run() -> np.ndarray:
            vectors = list(model.embed(texts))
            return np.asarray(vectors, dtype=np.float32)

        return await asyncio.to_thread(_run)

    async def health(self) -> bool:
        try:
            await self._ensure_loaded()
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("embedder_unhealthy", error=str(exc))
            return False


class OllamaEmbedder(Embedder):
    """Embed via Ollama's HTTP API. Slower (network hop) but no local model deps."""

    name = "ollama"
    dim = 384

    def __init__(self, settings: Settings, *, model: str | None = None, dim: int = 384):
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = model or settings.ollama_embed_model
        self.dim = dim
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for text in texts:
            resp = await self._client.post(
                f"{self._base}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape[-1] != self.dim:
            self.dim = int(arr.shape[-1])
        return arr

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base}/api/tags")
            return resp.status_code == 200
        except Exception:  # pragma: no cover
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider == "ollama":
        return OllamaEmbedder(settings)
    return FastEmbedEmbedder()
