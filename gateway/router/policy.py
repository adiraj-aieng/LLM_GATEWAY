"""Policy router: decides between the rule engine and the classifier.

Precedence, highest first:

1. **Client intent** — ``model != "auto"`` is never overridden.
2. **Safety rules** — an instruction-override marker always goes to the strongest tier.
   A classifier trained on ordinary prompts has no business downgrading that.
3. **Classifier** — when available, it makes the call and the rule's opinion is recorded
   alongside it so the dashboard can show *both*.
4. **Rules** — the fallback whenever the artifact is missing or inference failed.

Recording ``rule_reason`` next to ``reason`` is deliberate: when the two disagree you can
see it, and disagreement is exactly what you want to inspect when tuning the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from gateway.logging import get_logger
from gateway.router.base import RouteContext, RouteDecision, Router
from gateway.router.learned import LearnedRouter
from gateway.router.rules import RuleRouter

log = get_logger("gateway.router")

#: Rule reasons the classifier is not allowed to override.
HARD_RULE_REASONS = {"requested", "rule:injection"}


@dataclass
class RouterConfig:
    use_learned: bool = True


class PolicyRouter(Router):
    name = "policy"

    def __init__(self, settings, *, rules: RuleRouter | None = None,
                 learned: LearnedRouter | None = None):
        self._settings = settings
        self.rules = rules or RuleRouter(settings)
        self.learned = learned if learned is not None else LearnedRouter(settings)
        self.use_learned = True
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.learned.available

    @property
    def mode(self) -> str:
        if not self.use_learned:
            return "rules"
        if self.learned.available:
            return "learned"
        return "rules"

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "learned_available": self.learned.available,
            "learned_threshold": self.learned.threshold,
            "learned_metrics": self.learned.metrics,
            "learned_error": self.learned.error,
            "last_error": self._last_error,
        }

    def set_mode(self, mode: str) -> None:
        """Runtime switch: ``rules`` | ``learned``."""
        if mode not in {"rules", "learned"}:
            raise ValueError("mode must be 'rules' or 'learned'")
        self.use_learned = mode == "learned"

    # ------------------------------------------------------------------
    def route(self, ctx: RouteContext) -> RouteDecision:
        rule_decision = self.rules.route(ctx)

        if rule_decision.reason in HARD_RULE_REASONS:
            # Client intent / safety rules win outright, but note what the classifier said.
            if self.use_learned and self.learned.available:
                proba = self.learned.predict_proba(ctx.features)
                if proba is not None:
                    rule_decision = RouteDecision(
                        logical_model=rule_decision.logical_model,
                        upstream_model=rule_decision.upstream_model,
                        reason=rule_decision.reason,
                        confidence=rule_decision.confidence,
                        detail=(
                            f"{rule_decision.detail} (classifier P={proba:.3f} "
                            f"overridden by hard rule)"
                        ),
                        contributions=rule_decision.contributions,
                        router=self.name,
                    )
            return rule_decision

        if not self.use_learned or not self.learned.available:
            return RouteDecision(
                logical_model=rule_decision.logical_model,
                upstream_model=rule_decision.upstream_model,
                reason=rule_decision.reason,
                confidence=rule_decision.confidence,
                detail=rule_decision.detail,
                contributions=rule_decision.contributions,
                router=self.name,
            )

        try:
            decision = self.learned.route(ctx)
        except Exception as exc:
            # A broken artifact must degrade to rules, never fail the request.
            self._last_error = str(exc)
            log.warning("router_learned_failed", error=str(exc), fallback="rules")
            return RouteDecision(
                logical_model=rule_decision.logical_model,
                upstream_model=rule_decision.upstream_model,
                reason="fallback",
                confidence=rule_decision.confidence,
                detail=f"learned router failed ({exc}); used rules",
                contributions=rule_decision.contributions,
                router=self.name,
            )

        note = f"rules suggested `{rule_decision.logical_model}` via {rule_decision.reason}"
        return RouteDecision(
            logical_model=decision.logical_model,
            upstream_model=decision.upstream_model,
            reason=decision.reason,
            confidence=decision.confidence,
            detail=f"{decision.detail} | {note}",
            contributions=decision.contributions,
            router=self.name,
        )
