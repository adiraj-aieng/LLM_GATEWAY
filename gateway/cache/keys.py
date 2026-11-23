"""All cache key builders. **Single source of truth** — nothing else in the codebase
is allowed to concatenate a Redis key.

The exact-cache key is a hash over everything that can change a model's answer:

    messages (incl. system) + tools schema + sampling params + response_format + stop

Omit any one of these and you get the classic bugs:

* no system-prompt hash  → two apps with different system prompts share a cache,
                            each receives the other's answers
* no tools hash          → a cached ``tool_calls`` is returned for a different tool schema
* no temperature in key  → a greedy (temp=0) answer is served for a temp=1 request

``max_tokens`` deserves its own note. It is bucketed **upward** to a power of two so that
``max_tokens=1000`` and ``max_tokens=1024`` share an entry (hit rate), but on retrieval we
check ``cached.completion_tokens <= caller.max_tokens`` and treat a violation as a miss.
That gives the hit rate of bucketing without ever returning more tokens than the caller
allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gateway.config import Settings, get_settings
from gateway.core.hashing import blake2b_hex, hash_json
from gateway.core.normalize import extract_system_prompt, normalize_messages, normalize_text

# --- key prefixes -----------------------------------------------------------
EXACT_PREFIX = "gw:v1:exact"
SEM_IDX_PREFIX = "gw:v1:sem:idx"
SEM_DOC_PREFIX = "gw:v1:sem:doc"
SEM_VEC_PREFIX = "gw:v1:sem:vec"

# Minimum bucket so tiny requests don't all collapse into one entry.
_MIN_TOKEN_BUCKET = 256
_MAX_TOKEN_BUCKET = 8192


# =======================================================================
# sampling params
# =======================================================================
@dataclass(frozen=True)
class SamplingParams:
    """The subset of a chat request that affects the output."""

    temperature: float
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None

    def to_fingerprint(self) -> dict[str, Any]:
        """Canonical dict used for hashing. Order-independent (hash_json sorts keys)."""
        return {
            "temperature": round(self.temperature, 3),
            "top_p": round(self.top_p, 3),
            "max_tokens": bucket_max_tokens(self.max_tokens),
            "stop": sorted(self.stop),
            "response_format": self.response_format,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }


def bucket_max_tokens(n: int | None) -> int | None:
    """Round up to a power of two in [256, 8192]. `None` stays `None` (unbounded)."""
    if n is None:
        return None
    if n <= _MIN_TOKEN_BUCKET:
        return _MIN_TOKEN_BUCKET
    value = _MIN_TOKEN_BUCKET
    while value < n and value < _MAX_TOKEN_BUCKET:
        value *= 2
    return value


# =======================================================================
# bypass rules
# =======================================================================
def bypass_reason(
    params: SamplingParams, settings: Settings | None = None
) -> str | None:
    """Return why this request must not be cached, or `None` if it may be.

    High temperature means the caller *wants* diversity. Returning one stored sample for
    every identical prompt would silently destroy that, so we bypass rather than lie.
    """
    settings = settings or get_settings()
    if params.temperature > settings.cache_max_temp:
        return "temperature"
    if params.tools or params.tool_choice:
        # Tool state is not fully captured by the schema hash (the client's local tool
        # results are not), so tool-calling requests bypass the exact cache in v1.
        return "tools"
    return None


# =======================================================================
# exact cache
# =======================================================================
@dataclass(frozen=True)
class ExactKeyInput:
    """Everything folded into the exact-cache fingerprint."""

    messages: list[dict[str, Any]]
    sampling: SamplingParams

    def fingerprint(self) -> dict[str, Any]:
        return {
            "messages": normalize_messages(self.messages),
            "sampling": self.sampling.to_fingerprint(),
        }


def exact_fingerprint(inp: ExactKeyInput) -> str:
    """Hex hash of the canonical request. Same input → same hash, always."""
    return hash_json(inp.fingerprint())


def build_exact_key(
    api_key_id: int,
    logical_model: str,
    inp: ExactKeyInput,
) -> str:
    return f"{EXACT_PREFIX}:{api_key_id}:{logical_model}:{exact_fingerprint(inp)}"


def system_prompt_hash(messages: list[dict[str, Any]] | None) -> str:
    """Stored alongside semantic entries so a differing system prompt never hits."""
    return blake2b_hex(extract_system_prompt(messages))


def tools_hash(tools: list[dict[str, Any]] | None) -> str:
    return hash_json(tools) if tools else "none"


# =======================================================================
# semantic cache
# =======================================================================
def build_semantic_index_key(api_key_id: int, logical_model: str) -> str:
    return f"{SEM_IDX_PREFIX}:{api_key_id}:{logical_model}"


def build_semantic_doc_key(entry_id: str) -> str:
    return f"{SEM_DOC_PREFIX}:{entry_id}"


def build_semantic_vec_key(entry_id: str) -> str:
    return f"{SEM_VEC_PREFIX}:{entry_id}"


@dataclass
class SemanticEntry:
    """Persisted payload for one semantic-cache entry."""

    entry_id: str
    api_key_id: int
    model: str
    prompt_hash: str
    system_prompt_hash: str
    tools_hash: str
    response: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)
    max_tokens: int | None = None
    created_at: str = ""
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "api_key_id": self.api_key_id,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "system_prompt_hash": self.system_prompt_hash,
            "tools_hash": self.tools_hash,
            "response": self.response,
            "usage": self.usage,
            "max_tokens": self.max_tokens,
            "created_at": self.created_at,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticEntry":
        return cls(
            entry_id=d["entry_id"],
            api_key_id=d["api_key_id"],
            model=d["model"],
            prompt_hash=d.get("prompt_hash", ""),
            system_prompt_hash=d.get("system_prompt_hash", ""),
            tools_hash=d.get("tools_hash", "none"),
            response=d.get("response", {}),
            usage=d.get("usage", {}) or {},
            max_tokens=d.get("max_tokens"),
            created_at=d.get("created_at", ""),
            hits=int(d.get("hits", 0)),
        )


def fits_within_max_tokens(entry: SemanticEntry, requested_max: int | None) -> bool:
    """Guard against returning more tokens than the caller allowed.

    Paired with upward `max_tokens` bucketing: bucketing buys hit rate, this buys safety.
    """
    if requested_max is None:
        return True
    produced = int(entry.usage.get("completion_tokens", 0) or 0)
    return produced <= requested_max
