"""Feature extraction — shared by the rule router, the learned router and the explainer.

Keeping extraction in one place matters: if rules and the classifier disagree about what
"has code" means, the router's behaviour becomes unexplainable, which defeats the point of
shipping a router at all.

Features are deliberately **cheap and deterministic** — no model calls, no I/O — because
they run on the hot path before the upstream is even chosen.
"""

from __future__ import annotations

import re
from typing import Any

from gateway.core.normalize import extract_system_prompt, normalize_messages, normalize_text

# --- patterns ---------------------------------------------------------------
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INJECTION = re.compile(
    r"\b(ignore (?:all )?(?:previous|above|earlier) instructions|"
    r"disregard the (?:system|previous) prompt|"
    r"you are now|new system prompt|override (?:your|the) instructions)\b",
    re.IGNORECASE,
)
_JSON_HINT = re.compile(
    r"\b(json|yaml|csv|schema|marshal|seriali[sz]e|response_format|api response)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(
    r"\b(what|why|how|when|where|which|who|explain|describe|compare|"
    r"list|define|summarize|translate|prove|derive)\b",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"\b(write|implement|build|create|refactor|debug|fix|optimi[sz]e|"
    r"generate|compose|draft|design)\b",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*([-*•]|\d+\.)\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+")

#: Lines that look like code rather than prose.
_CODE_LINE = re.compile(
    r"^\s*(?:"
    r"def |class |import |from .+ import |return |if __name__|"
    r"for .+ in .+:|while .+:|try:|except|with .+:|"
    r"(?:const|let|var|function|public|private|void|int|string)\s+|"
    r".*[{};]\s*$|.*=\s*.+"
    r")",
    re.MULTILINE,
)

#: Strong-model territory: reasoning a small model typically fumbles.
_REASONING = re.compile(
    r"\b(prove|deri[vs]e|theorem|lemma|optimi[sz]e|concurren(t|cy)|"
    r"race condition|deadlock|distributed consensus|raft|paxos|"
    r"asymptotic|big[- ]o|regression proof|legal advice|medical)\b",
    re.IGNORECASE,
)


def _content_text(messages: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for msg in normalize_messages(messages or []):
        parts.append(msg.get("content", ""))
    return "\n".join(p for p in parts if p)


def extract_features(
    messages: list[dict[str, Any]] | None,
    *,
    prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Compute the router's feature set from a chat message list."""
    body = _content_text(messages)
    system = extract_system_prompt(messages)

    lines = body.splitlines() or ([body] if body else [])
    line_count = len(lines)
    code_matches = _CODE_LINE.findall(body)
    code_lines = len(code_matches)
    fence_count = len(_CODE_FENCE.findall(body))

    non_ascii = sum(1 for c in body if ord(c) > 127)
    char_len = len(body)

    feats: dict[str, Any] = {
        # volume
        "prompt_tokens": prompt_tokens,
        "char_len": char_len,
        "line_count": line_count,
        "word_count": len(body.split()),
        "system_len": len(system),
        # shape
        "has_code": fence_count > 0 or code_lines >= 3,
        "code_fence_count": fence_count,
        "code_line_ratio": round(code_lines / line_count, 4) if line_count else 0.0,
        "bullet_count": len(_BULLET.findall(body)),
        "number_count": sum(c.isdigit() for c in body),
        # intent
        "wants_json": bool(_JSON_HINT.search(body)),
        "question_count": len(_QUESTION_WORDS.findall(body)),
        "imperative_count": len(_IMPERATIVE.findall(body)),
        "needs_reasoning": bool(_REASONING.search(body)),
        "has_url": bool(_URL.search(body)),
        # safety
        "injection_marker": bool(_INJECTION.search(body)),
        # language
        "non_ascii_ratio": round(non_ascii / char_len, 4) if char_len else 0.0,
        "is_empty": char_len == 0,
    }
    # Compact numeric list, in a fixed order — this is what the classifier consumes.
    feats["vector"] = feature_vector(feats)
    return feats


#: Order is part of the model contract: `train_router.py` writes this list into the
#: artifact and `learned.py` reads it back, so reordering can't silently corrupt inference.
FEATURE_ORDER: list[str] = [
    "prompt_tokens",
    "char_len",
    "line_count",
    "word_count",
    "system_len",
    "has_code",
    "code_fence_count",
    "code_line_ratio",
    "bullet_count",
    "number_count",
    "wants_json",
    "question_count",
    "imperative_count",
    "needs_reasoning",
    "has_url",
    "injection_marker",
    "non_ascii_ratio",
    "is_empty",
]


def feature_vector(feats: dict[str, Any]) -> list[float]:
    """Fixed-order numeric vector for the classifier."""
    return [float(feats.get(name, 0)) for name in FEATURE_ORDER]


def build_context(
    *,
    requested_model: str,
    messages: list[dict[str, Any]] | None,
    prompt_tokens: int = 0,
    wants_json: bool = False,
    is_stream: bool = False,
) -> dict[str, Any]:
    """Convenience bundle used by the pipeline."""
    feats = extract_features(messages, prompt_tokens=prompt_tokens)
    return {
        "requested_model": requested_model,
        "prompt_text": _content_text(messages),
        "system_text": extract_system_prompt(messages),
        "prompt_tokens": prompt_tokens,
        "has_code": bool(feats["has_code"]),
        "wants_json": wants_json or bool(feats["wants_json"]),
        "is_stream": is_stream,
        "features": feats,
    }
