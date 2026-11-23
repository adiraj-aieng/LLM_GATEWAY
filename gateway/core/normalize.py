"""Prompt normalization — the input to every cache key.

Rules (deliberate, and worth defending in review):

* **NFKC** so that `é` and `e\u0301` (and full-width/half-width forms) hash identically.
  Skipping this silently fragments the cache for any non-ASCII prompt.
* **Collapse whitespace** so re-indenting a prompt doesn't miss the cache.
* **Do NOT lowercase** — case carries meaning in prompts ("Python" vs "python" is rarely
  decisive, but "US" vs "us" is, and we would rather miss than return a wrong answer).
* **Do NOT strip punctuation** — same reason: conservative beats clever here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Canonical form of a single string. Idempotent."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def normalize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Canonical form of a chat message list.

    Only `role` and `content` participate in the key — metadata like `name` or
    provider-specific extras must not fragment the cache. Ordering is preserved
    because conversation order is semantic.
    """
    if not messages:
        return []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        out.append(
            {
                "role": str(msg.get("role", "")),
                "content": normalize_text(_content_as_text(msg.get("content"))),
            }
        )
    return out


def _content_as_text(content: Any) -> str:
    """Flatten multimodal content (str or list of parts) into one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def extract_system_prompt(messages: list[dict[str, Any]] | None) -> str:
    """Concatenate every `system` message. Used for the system-prompt cache guard."""
    if not messages:
        return ""
    return normalize_text(
        " ".join(
            _content_as_text(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "system"
        )
    )
