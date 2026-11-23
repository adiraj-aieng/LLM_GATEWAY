"""Semantic cache: near-duplicate prompts served from a stored response.

A hit requires **all four** of:

1. same ``api_key_id`` and logical model (separate index per scope)
2. same ``system_prompt_hash`` and ``tools_hash`` — a differing system prompt is a
   *different request*, and sharing it across apps is the classic semantic-cache bug
3. ``cosine(top1, query) >= SEMANTIC_SIM_THRESHOLD``
4. ``cosine(top1) - cosine(top2) >= SEMANTIC_MARGIN_GUARD``

Criterion 4 is the false-positive guard. Without it, a query landing between two
near-tied candidates silently gets whichever was inserted first — which is exactly the
failure mode that makes semantic caching feel unreliable.

Vectors live in Redis (survive restart) and in an in-process index (fast search); the
index is rebuilt from Redis on startup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from redis.asyncio import Redis

from gateway.cache.keys import (
    SemanticEntry,
    build_semantic_doc_key,
    build_semantic_index_key,
    build_semantic_vec_key,
)
from gateway.cache.vector_index import VectorIndex
from gateway.config import Settings
from gateway.logging import get_logger
from gateway.providers.embedder import Embedder

log = get_logger("gateway.cache.semantic")


@dataclass(frozen=True)
class SemanticHit:
    entry: SemanticEntry
    similarity: float
    margin: float
    age_s: float


class SemanticCache:
    def __init__(self, redis: Redis, settings: Settings, embedder: Embedder):
        self._redis = redis
        self._settings = settings
        self._embedder = embedder
        self._indexes: dict[str, VectorIndex] = {}
        self._warmed = False

    # ------------------------------------------------------------------
    def _index(self, scope: str) -> VectorIndex:
        idx = self._indexes.get(scope)
        if idx is None:
            idx = VectorIndex(self._embedder.dim or 384)
            self._indexes[scope] = idx
        elif idx.dim != (self._embedder.dim or 384):
            # Embedder model changed underneath us — start clean rather than mix dims.
            idx = VectorIndex(self._embedder.dim or 384)
            self._indexes[scope] = idx
        return idx

    # ------------------------------------------------------------------
    async def warmup(self) -> int:
        """Rebuild in-process indexes from Redis. Called once at startup."""
        if self._warmed:
            return 0
        loaded = 0
        try:
            async for key in self._redis.scan_iter(
                match="gw:v1:sem:idx:*", count=100
            ):
                scope = key.removeprefix("gw:v1:sem:idx:")
                entry_ids = await self._redis.smembers(key)
                idx = self._index(scope)
                for entry_id in entry_ids:
                    if isinstance(entry_id, bytes):
                        entry_id = entry_id.decode()
                    raw = await self._redis.get(build_semantic_vec_key(entry_id))
                    if not raw:
                        await self._redis.srem(key, entry_id)
                        continue
                    vec = np.frombuffer(raw, dtype=np.float32)
                    idx.add(entry_id, vec)
                    loaded += 1
            self._warmed = True
            log.info("semantic_cache_warmed", vectors=loaded)
        except Exception as exc:  # pragma: no cover
            log.warning("semantic_cache_warmup_failed", error=str(exc))
        return loaded

    # ------------------------------------------------------------------
    async def get(
        self,
        text: str,
        *,
        api_key_id: int,
        model: str,
        system_hash: str,
        tools_hash: str,
        requested_max_tokens: int | None = None,
    ) -> tuple[SemanticHit | None, float]:
        """Look up `text`. Returns ``(hit_or_none, top1_similarity)``.

        The similarity is returned even on a miss so callers can log it.
        """
        if not self._settings.semantic_cache_enabled:
            return None, 0.0

        scope = f"{api_key_id}:{model}"
        try:
            vector = await self._embedder.embed_one(text)
            idx = self._index(scope)
            matches = idx.search(vector, k=2)
            if not matches:
                return None, 0.0

            top1 = matches[0]
            top2 = matches[1] if len(matches) > 1 else None
            margin = top1.score - (top2.score if top2 else 0.0)

            if top1.score < self._settings.semantic_sim_threshold:
                return None, top1.score
            if margin < self._settings.semantic_margin_guard:
                # Ambiguous: two candidates nearly tied. Refuse rather than guess.
                log.debug(
                    "semantic_cache_margin_rejected",
                    score=round(top1.score, 4),
                    margin=round(margin, 4),
                )
                return None, top1.score

            doc = await self._redis.get(build_semantic_doc_key(top1.entry_id))
            if not doc:
                # Index and store disagree — drop the stale vector.
                idx.remove(top1.entry_id)
                return None, top1.score

            import json

            entry = SemanticEntry.from_dict(json.loads(doc))

            # Guard rails: a differing system prompt or tools schema is a different request.
            if entry.system_prompt_hash != system_hash:
                return None, top1.score
            if entry.tools_hash != tools_hash:
                return None, top1.score
            if not _fits(entry, requested_max_tokens):
                return None, top1.score

            age = _age_seconds(entry.created_at)
            return (
                SemanticHit(
                    entry=entry,
                    similarity=round(top1.score, 6),
                    margin=round(margin, 6),
                    age_s=age,
                ),
                top1.score,
            )
        except Exception as exc:  # pragma: no cover - cache must never fail a request
            log.warning("semantic_cache_get_failed", error=str(exc))
            return None, 0.0

    # ------------------------------------------------------------------
    async def set(
        self,
        text: str,
        *,
        api_key_id: int,
        model: str,
        system_hash: str,
        tools_hash: str,
        prompt_hash: str,
        response: dict[str, Any],
        usage: dict[str, int],
        max_tokens: int | None = None,
    ) -> None:
        if not self._settings.semantic_cache_enabled:
            return
        try:
            idx_len = len(self._index(f"{api_key_id}:{model}"))
            if idx_len >= self._settings.semantic_max_vectors:
                log.warning(
                    "semantic_cache_capacity_reached",
                    size=idx_len,
                    limit=self._settings.semantic_max_vectors,
                )
                return

            vector = await self._embedder.embed_one(text)
            entry_id = uuid.uuid4().hex
            scope = f"{api_key_id}:{model}"
            entry = SemanticEntry(
                entry_id=entry_id,
                api_key_id=api_key_id,
                model=model,
                prompt_hash=prompt_hash,
                system_prompt_hash=system_hash,
                tools_hash=tools_hash,
                response=response,
                usage=usage,
                max_tokens=max_tokens,
                created_at=datetime.now(timezone.utc).isoformat(),
                hits=0,
            )

            import json

            ttl = self._settings.semantic_cache_ttl_s
            pipe = self._redis.pipeline(transaction=True)
            pipe.set(
                build_semantic_doc_key(entry_id),
                json.dumps(entry.to_dict(), ensure_ascii=False),
                ex=ttl,
            )
            pipe.set(
                build_semantic_vec_key(entry_id),
                np.asarray(vector, dtype=np.float32).tobytes(),
                ex=ttl,
            )
            pipe.sadd(build_semantic_index_key(api_key_id, model), entry_id)
            pipe.expire(build_semantic_index_key(api_key_id, model), ttl)
            await pipe.execute()

            self._index(scope).add(entry_id, np.asarray(vector, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            log.warning("semantic_cache_set_failed", error=str(exc))

    # ------------------------------------------------------------------
    async def clear(self, *, api_key_id: int | None = None) -> int:
        removed = 0
        try:
            pattern = "gw:v1:sem:idx:*"
            if api_key_id is not None:
                pattern = f"gw:v1:sem:idx:{api_key_id}:*"
            async for key in self._redis.scan_iter(match=pattern, count=100):
                scope = key.removeprefix("gw:v1:sem:idx:")
                entry_ids = await self._redis.smembers(key)
                for entry_id in entry_ids:
                    if isinstance(entry_id, bytes):
                        entry_id = entry_id.decode()
                    await self._redis.delete(
                        build_semantic_doc_key(entry_id),
                        build_semantic_vec_key(entry_id),
                    )
                    removed += 1
                await self._redis.delete(key)
                if scope in self._indexes:
                    self._indexes[scope].clear()
        except Exception as exc:  # pragma: no cover
            log.warning("semantic_cache_clear_failed", error=str(exc))
        return removed

    def stats(self) -> dict[str, int]:
        return {scope: len(idx) for scope, idx in self._indexes.items()}


def _fits(entry: SemanticEntry, requested_max: int | None) -> bool:
    if requested_max is None:
        return True
    produced = int(entry.usage.get("completion_tokens", 0) or 0)
    return produced <= requested_max


def _age_seconds(created: str) -> float:
    if not created:
        return 0.0
    try:
        dt = datetime.fromisoformat(created)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except ValueError:  # pragma: no cover
        return 0.0
