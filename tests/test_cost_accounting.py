"""Cost accounting: the three-tier honesty model.

The deployment is local, so real spend is $0. The tests below exist to pin that down —
and to make sure the *modelled* figure and its `basis` label always travel together,
because a dollar number without provenance is the thing we explicitly refuse to ship.
"""

from __future__ import annotations

from gateway.pipeline.cost import (
    CHARS_PER_TOKEN,
    PRICE_VERSION,
    cached_savings,
    compute_cost,
    estimate_message_tokens,
    estimate_tokens,
    precharge_tokens,
)


# =======================================================================
# token estimation (pre-charge only — never accounting)
# =======================================================================
def test_estimate_tokens_is_nonzero_and_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert long > short


def test_estimate_tokens_uses_documented_ratio():
    text = "x" * int(CHARS_PER_TOKEN * 50)
    assert estimate_tokens(text) == 50


def test_estimate_message_tokens_includes_per_message_overhead():
    one = estimate_message_tokens([{"role": "user", "content": "hi"}])
    two = estimate_message_tokens(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert two > one, "each message adds priming tokens"
    assert estimate_message_tokens([]) == 0


def test_estimate_message_tokens_handles_multimodal_content():
    n = estimate_message_tokens(
        [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]
    )
    assert n > 0


# =======================================================================
# pre-charge
# =======================================================================
def test_precharge_reserves_prompt_plus_max_tokens():
    assert precharge_tokens(estimated_prompt_tokens=100, max_tokens=512) == 612


def test_precharge_uses_default_when_max_tokens_unset():
    # Without max_tokens we must still reserve something, or streaming callers get a
    # free pass on the bucket until usage arrives (which it may never do).
    assert precharge_tokens(estimated_prompt_tokens=100, max_tokens=None) > 100


def test_precharge_is_at_least_one():
    assert precharge_tokens(estimated_prompt_tokens=0, max_tokens=0) >= 1


# =======================================================================
# cost: real vs modelled
# =======================================================================
def test_local_inference_costs_zero_real_dollars():
    result = compute_cost("llama3.2:3b", 1000, 1000)
    assert result.cost_usd == 0.0
    assert result.basis == "local_inference"
    assert result.price_version == PRICE_VERSION


def test_modelled_cost_is_positive_and_labelled():
    result = compute_cost("llama3.2:3b", 1_000_000, 1_000_000)
    assert result.cost_usd == 0.0, "local inference must never report real spend"
    assert result.modeled_usd > 0, "1M+1M tokens must have a modelled equivalent"
    assert result.basis == "modeled_from_hosted_equivalent" or result.basis == "local_inference"
    assert result.as_dict()["price_version"] == PRICE_VERSION


def test_modelled_cost_scales_with_tokens():
    small = compute_cost("llama3.2:3b", 1000, 500)
    large = compute_cost("llama3.2:3b", 100_000, 50_000)
    assert large.modeled_usd > small.modeled_usd


def test_zero_tokens_produce_zero_modeled_cost():
    result = compute_cost("llama3.2:3b", 0, 0)
    assert result.modeled_usd == 0.0
    assert result.cost_usd == 0.0


def test_unknown_model_falls_back_to_default_equivalent():
    # An unmapped model must still produce a labelled figure, not a KeyError on the hot path.
    result = compute_cost("some-future-model", 1_000_000, 1_000_000)
    assert result.modeled_usd > 0


def test_cached_savings_share_the_same_provenance():
    result = cached_savings(prompt_tokens=10_000, completion_tokens=2_000, model="llama3.2:3b")
    assert result.cost_usd == 0.0
    assert result.basis == "local_inference"
    assert result.price_version == PRICE_VERSION


def test_cost_result_serialises_with_basis():
    payload = compute_cost("llama3.2:3b", 100, 100).as_dict()
    assert set(payload) >= {"cost_usd", "modeled_usd", "basis", "price_version"}
    assert payload["cost_usd"] == 0.0
