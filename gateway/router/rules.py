"""Rule-based router — the Day 1 default.

Rules are ordered and **first match wins**, so behaviour is deterministic and the
``reason`` field is always truthful. Explicit client intent outranks every inference:
if a caller names a model, we honour it.

Order matters:

1. ``model != "auto"``  → do what was asked. Never second-guess the caller.
2. injection marker     → strongest tier; treat untrusted instructions as a hard case.
3. explicit reasoning   → strongest tier.
4. code                 → specialist tier.
5. long prompt          → mid tier (long context needs more headroom).
6. otherwise            → cheapest tier.
"""

from __future__ import annotations

from gateway.router.base import RouteContext, RouteDecision, Router

#: A prompt longer than this (tokens) gets a bigger model regardless of content.
LONG_PROMPT_TOKENS = 400
#: …or this many characters, when token counting wasn't available.
LONG_PROMPT_CHARS = 2000


class RuleRouter(Router):
    name = "rules"

    def __init__(self, settings):  # settings unused today; kept for symmetry
        self._settings = settings

    def route(self, ctx: RouteContext) -> RouteDecision:
        requested = (ctx.requested_model or "auto").strip()

        if requested and requested != "auto":
            return RouteDecision(
                logical_model=requested,
                upstream_model=self._settings.resolve_model(requested),
                reason="requested",
                confidence=1.0,
                detail=f"client explicitly requested `{requested}`",
                router=self.name,
            )

        feats = ctx.features
        model_map = self._settings.model_map

        if feats.get("injection_marker"):
            return self._decide("smart", "rule:injection", 1.0,
                                "prompt contains instruction-override language")

        if feats.get("needs_reasoning"):
            return self._decide("smart", "rule:reasoning", 0.9,
                                "prompt asks for proof, derivation or concurrency reasoning")

        if feats.get("has_code"):
            detail = (
                "code fence present"
                if feats.get("code_fence_count")
                else f"{feats.get('code_line_ratio', 0)} of lines look like code"
            )
            # `coder` only routes somewhere different if it's actually mapped.
            target = "coder" if "coder" in model_map else "smart"
            return self._decide(target, "rule:code_block", 0.9, detail)

        if ctx.prompt_tokens >= LONG_PROMPT_TOKENS or feats.get("char_len", 0) >= LONG_PROMPT_CHARS:
            return self._decide(
                "smart",
                "rule:length",
                0.8,
                f"{ctx.prompt_tokens or feats.get('char_len')} tokens",
            )

        return self._decide("fast", "rule:default", 0.7, "short, non-code prompt")

    # ------------------------------------------------------------------
    def _decide(self, logical: str, reason: str, confidence: float, detail: str) -> RouteDecision:
        return RouteDecision(
            logical_model=logical,
            upstream_model=self._settings.resolve_model(logical),
            reason=reason,
            confidence=confidence,
            detail=detail,
            router=self.name,
        )
