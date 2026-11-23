"""In-process vector index for the semantic cache.

Why numpy and not Redis?
    Plain ``redis-server`` has **no vector search**. ``FT.SEARCH`` with ``VECTOR`` is
    RediSearch, a separate module that is not installed here. Claiming otherwise in a
    design doc would be wrong.

Why not pgvector on Day 1?
    An extra round trip on the hottest path, and Postgres isn't running in dev.

Scaling path (documented, not built): persist to pgvector once we pass
``SEMANTIC_MAX_VECTORS``. This class is correct to roughly 50k vectors — brute-force
cosine over 50k × 384 float32 is ~19 MB and a few milliseconds.

Implementation notes
--------------------
* Vectors are **L2-normalised at insert**, so cosine similarity reduces to a single
  matrix-vector product (``M @ q``) — no division at query time.
* Removal leaves a tombstone rather than reshuffling rows; tombstoned rows are masked to
  ``-inf`` during search and their slots recycled on the next insert. This keeps the hot
  path allocation-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gateway.logging import get_logger

log = get_logger("gateway.cache.vector")

_TOMBSTONE = "\x00tombstone"
_GROWTH_FACTOR = 1.5
_MIN_CAPACITY = 64


@dataclass(frozen=True)
class Match:
    entry_id: str
    score: float  # cosine similarity in [-1, 1]
    position: int


class VectorIndex:
    """Cosine-similarity index over a fixed dimensionality."""

    def __init__(self, dim: int, *, normalize_eps: float = 1e-9):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._eps = normalize_eps
        self._capacity = _MIN_CAPACITY
        self._matrix = np.zeros((self._capacity, dim), dtype=np.float32)
        self._ids: list[str | None] = [None] * self._capacity
        self._pos: dict[str, int] = {}
        self._free: list[int] = []
        self._alive = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._alive

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    def add(self, entry_id: str, vector: np.ndarray) -> int:
        """Insert or replace. Returns the row position."""
        if entry_id in self._pos:
            self.remove(entry_id)

        vec = self._normalize(vector)
        if self._free:
            row = self._free.pop()
        else:
            # `_capacity` is the first row beyond the live range; grow first so it exists.
            row = self._capacity
            self._grow()

        self._matrix[row] = vec
        self._ids[row] = entry_id
        self._pos[entry_id] = row
        self._alive += 1
        return row

    def remove(self, entry_id: str) -> bool:
        row = self._pos.pop(entry_id, None)
        if row is None:
            return False
        self._ids[row] = _TOMBSTONE
        self._matrix[row] = 0.0
        self._free.append(row)
        self._alive -= 1
        return True

    def contains(self, entry_id: str) -> bool:
        return entry_id in self._pos

    def clear(self) -> None:
        self._matrix[:] = 0.0
        self._ids = [None] * self._capacity
        self._pos.clear()
        self._free.clear()
        self._alive = 0

    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, *, k: int = 2) -> list[Match]:
        """Top-``k`` most similar live entries, descending by score.

        ``k=2`` is the normal call: the semantic cache needs the runner-up to compute the
        ambiguity margin (``top1 - top2``) that guards against false positives.
        """
        if self._alive == 0 or k <= 0:
            return []

        q = self._normalize(query)
        scores = self._matrix @ q  # cosine, since both sides are unit-length

        # Mask dead rows so they can never win.
        dead = np.fromiter(
            (i is None or i == _TOMBSTONE for i in self._ids),
            dtype=bool,
            count=self._capacity,
        )
        if dead.any():
            scores = np.where(dead, -np.inf, scores)

        k = min(k, self._alive)
        if k >= self._capacity:
            order = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n) vs O(n log n) for full sort — matters at 50k rows.
            part = np.argpartition(scores, -k)[-k:]
            order = part[np.argsort(scores[part])[::-1]]

        out: list[Match] = []
        for row in order[:k]:
            score = float(scores[row])
            if not math.isfinite(score):
                break
            entry_id = self._ids[row]
            if entry_id is None or entry_id == _TOMBSTONE:
                continue
            out.append(Match(entry_id=entry_id, score=score, position=int(row)))
        return out

    # ------------------------------------------------------------------
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(
                f"vector dim {arr.shape[0]} does not match index dim {self.dim}"
            )
        norm = float(np.linalg.norm(arr))
        if norm < self._eps:
            # A zero vector has no direction; returning it would make every cosine 0.
            return arr
        return arr / norm

    def _grow(self) -> None:
        new_capacity = max(_MIN_CAPACITY, int(self._capacity * _GROWTH_FACTOR))
        grown = np.zeros((new_capacity, self.dim), dtype=np.float32)
        grown[: self._capacity] = self._matrix
        self._matrix = grown
        self._ids.extend([None] * (new_capacity - self._capacity))
        self._capacity = new_capacity
        log.debug("vector_index_grew", capacity=new_capacity)
