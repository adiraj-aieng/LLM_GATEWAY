"""Router interface and result type.

The router answers one question: **which upstream model should serve this request?**

Everything downstream consumes :class:`RouteDecision`. Keeping the decision, the reason and
the confidence together is what lets the dashboard explain *why* a request went where it
did — and it is the difference between "I added a router" and "I can show you what it did."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    logical_model: str
    upstream_model: str
    reason: str  # 'rule:code_block' | 'classifier' | 'fallback' | 'budget' | 'requested'
    confidence: float = 1.0
    #: Human-readable explanation for the dashboard "why" panel.
    detail: str = ""
    #: Feature contributions when the decision came from the learned model.
    contributions: list[dict[str, Any]] = field(default_factory=list)
    router: str = "rules"

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_model": self.logical_model,
            "upstream_model": self.upstream_model,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "contributions": self.contributions,
            "router": self.router,
        }


@dataclass(frozen=True)
class RouteContext:
    """Everything the router is allowed to look at.

    Deliberately *not* the raw messages: routers should work off extracted features so
    prompt content isn't accidentally persisted in decision logs.
    """

    requested_model: str
    prompt_text: str
    system_text: str = ""
    prompt_tokens: int = 0
    has_code: bool = False
    wants_json: bool = False
    is_stream: bool = False
    features: dict[str, Any] = field(default_factory=dict)


class Router(ABC):
    name: str = "base"

    @abstractmethod
    def route(self, ctx: RouteContext) -> RouteDecision: ...

    async def health(self) -> bool:
        return True
