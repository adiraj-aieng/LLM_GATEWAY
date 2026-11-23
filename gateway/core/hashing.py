"""Hashing helpers.

blake2b is used throughout (fast, in the stdlib, no extra dependency). Digests are
returned hex-encoded so they can go straight into Redis keys and Postgres text columns.

Two rules that matter:

* `hash_json` produces a **canonical** serialization (sorted keys, no whitespace) so that
  two semantically identical objects hash identically regardless of construction order.
* We never hash a secret for storage comparison of API keys by hand — `auth.py` owns that
  and adds a per-key salt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def blake2b_hex(data: str | bytes, *, size: int = 32) -> str:
    """Hex digest of `data`. `size` is the digest length in bytes (16..64)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def hash_json(obj: Any, *, size: int = 32) -> str:
    """Canonical hash of a JSON-serializable object.

    `sort_keys` + compact separators make this order- and whitespace-insensitive.
    `default=str` keeps datetimes and similar from raising.
    """
    canonical = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return blake2b_hex(canonical, size=size)


def short_hash(obj: Any, *, length: int = 16) -> str:
    """First `length` hex chars — for log previews and request ids."""
    return hash_json(obj)[:length]
