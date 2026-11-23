"""Token counting and cost.

The honesty problem, restated
-----------------------------
This deployment is **local** (Ollama on an RTX 4050). Local inference has no marginal
dollar cost, so ``cost_usd`` is legitimately ``0.0`` and must stay that way. Claiming a
percentage cost saving against zero would be meaningless.

What we do instead:

* ``cost_usd``      — real spend. `0.0` here, because it is.
* ``modeled_usd``   — what this *would* have cost at a hosted equivalent's published
  price. Clearly labelled, and never aggregated into a headline without its basis.
* ``basis``         — travels with every number: ``local_inference`` |
  ``modeled_from_hosted_equivalent``.

``PRICE_VERSION`` is stamped on every log row. Provider prices change; without versioning,
recomputing last month's spend would silently produce different numbers.

.. warning::
   The table below records *modeled equivalents*, not live prices. **Verify against the
   providers' current pricing pages before publishing any benchmark chart.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRICE_VERSION = "v1"

#: (input_usd_per_mtok, output_usd_per_mtok) — hosted models we treat as equivalents
#: for the *modelled* figure only.
MODELED_HOSTED_EQUIVALENTS: dict[str, tuple[float, float]] = {
    "llama3.2:1b": (0.15, 0.60),   # ~ gpt-4o-mini
    "llama3.2:3b": (0.15, 0.60),   # ~ gpt-4o-mini
    "qwen2.5-coder:3b": (0.15, 0.60),
    "llama3.1:8b": (0.80, 4.00),   # ~ claude-3-5-haiku
}

DEFAULT_EQUIVALENT = (0.15, 0.60)

#: Below this the gateway treats a cost figure as noise and reports 0.
COST_EPSILON = 1e-9

#: Heuristic used **only** for pre-charging the rate limiter, before the upstream has
#: spoken. Actual accounting always uses the upstream's reported usage.
CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class CostResult:
    cost_usd: float          # real: 0.0 for local inference
    modeled_usd: float       # hosted-equivalent estimate
    basis: str
    price_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cost_usd": round(self.cost_usd, 6),
            "modeled_usd": round(self.modeled_usd, 6),
            "basis": self.basis,
            "price_version": self.price_version,
        }


def estimate_tokens(text: str) -> int:
    """Cheap prompt-token estimate for pre-charging. Never used for accounting."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate across a message list, including per-message framing overhead."""
    if not messages:
        return 0
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        total += estimate_tokens(str(content))
        total += 4  # priming tokens per message, per OpenAI's published accounting
    return total + 3  # reply priming


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostResult:
    """Real cost (0 locally) plus the labelled hosted-equivalent figure."""
    in_price, out_price = MODELED_HOSTED_EQUIVALENTS.get(model, DEFAULT_EQUIVALENT)
    modeled = (
        prompt_tokens / 1_000_000.0 * in_price
        + completion_tokens / 1_000_000.0 * out_price
    )
    if modeled < COST_EPSILON:
        modeled = 0.0
    return CostResult(
        cost_usd=0.0,
        modeled_usd=round(modeled, 6),
        basis="local_inference",
        price_version=PRICE_VERSION,
    )


def precharge_tokens(
    *,
    estimated_prompt_tokens: int,
    max_tokens: int | None,
    default_max: int = 1024,
) -> int:
    """Tokens to reserve from the rate limiter before the upstream call.

    Prompt tokens are estimated; the completion is reserved at ``max_tokens`` (or a
    conservative default when unset). Unused capacity is refunded after the stream ends.
    """
    completion_reserve = max_tokens if max_tokens is not None else default_max
    return max(1, estimated_prompt_tokens + completion_reserve)


def cached_savings(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> CostResult:
    """Cost avoided by serving from cache instead of the upstream."""
    return compute_cost(model, prompt_tokens, completion_tokens)
