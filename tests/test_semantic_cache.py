"""Semantic cache: similarity threshold, the ambiguity margin, and the system guard.

The margin test is the important one. Without a top1−top2 check, a query sitting between
two near-tied candidates silently gets whichever was inserted first — which is exactly how
a semantic cache earns its reputation for returning confident nonsense.
"""

from __future__ import annotations

from tests.conftest import FakeProvider, cache_header, chat_payload


def test_paraphrase_hits_semantic_cache(client, auth_headers):
    # Identical words, different punctuation → different exact key, identical embedding.
    first = client.post(
        "/v1/chat/completions",
        json=chat_payload("what is a token bucket"),
        headers=auth_headers,
    )
    assert cache_header(first) == "MISS"
    assert len(FakeProvider.calls) == 1

    second = client.post(
        "/v1/chat/completions",
        json=chat_payload("what is a token bucket?"),
        headers=auth_headers,
    )
    assert cache_header(second) == "SEMANTIC_HIT"
    assert second.headers.get("x-gateway-similarity")
    assert len(FakeProvider.calls) == 1, "semantic hit must not reach the upstream"
    assert (
        second.json()["choices"][0]["message"]["content"]
        == first.json()["choices"][0]["message"]["content"]
    )


def test_unrelated_prompt_misses(client, auth_headers):
    client.post(
        "/v1/chat/completions",
        json=chat_payload("what is a token bucket"),
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("explain quantum chromodynamics"),
        headers=auth_headers,
    )
    assert cache_header(resp) == "MISS"
    assert len(FakeProvider.calls) == 2


def test_system_prompt_difference_blocks_semantic_hit(client, app, auth_headers):
    """A differing system prompt is a different request, whatever the user text says."""
    client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "summarize",
            messages=[
                {"role": "system", "content": "Answer as a pirate."},
                {"role": "user", "content": "summarize"},
            ],
        ),
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "summarize",
            messages=[
                {"role": "system", "content": "Answer as an accountant."},
                {"role": "user", "content": "summarize"},
            ],
        ),
        headers=auth_headers,
    )
    assert cache_header(resp) == "MISS"
    assert len(FakeProvider.calls) == 2


def test_margin_guard_rejects_ambiguous_match(client, app, auth_headers):
    """Two near-tied candidates → refuse rather than guess."""
    settings = app.state.settings

    # Phase 1: store two entries that are similar to each other but below threshold.
    settings.semantic_sim_threshold = 0.99
    settings.semantic_margin_guard = 0.05

    a = client.post(
        "/v1/chat/completions",
        json=chat_payload("alpha beta gamma delta"),
        headers=auth_headers,
    )
    b = client.post(
        "/v1/chat/completions",
        json=chat_payload("alpha beta gamma epsilon"),
        headers=auth_headers,
    )
    assert cache_header(a) == "MISS"
    assert cache_header(b) == "MISS", "0.75 similarity must miss at threshold 0.99"
    assert len(FakeProvider.calls) == 2

    # Phase 2: lower the threshold so both become candidates for the same query.
    settings.semantic_sim_threshold = 0.5

    # Query ties between A and B (≈0.75 each) → margin ≈ 0 → rejected.
    ambiguous = client.post(
        "/v1/chat/completions",
        json=chat_payload("alpha beta gamma zeta"),
        headers=auth_headers,
    )
    assert cache_header(ambiguous) == "MISS", "ambiguous match must be rejected"
    assert len(FakeProvider.calls) == 3

    # Query clearly matches A (1.0) and less so B (0.75) → margin 0.25 → accepted.
    clear = client.post(
        "/v1/chat/completions",
        json=chat_payload("alpha beta gamma delta."),
        headers=auth_headers,
    )
    assert cache_header(clear) == "SEMANTIC_HIT"
    assert float(clear.headers["x-gateway-similarity"]) >= 0.99
    assert len(FakeProvider.calls) == 3, "clear match must not reach the upstream"


def test_semantic_threshold_configurable_at_runtime(client, app, auth_headers):
    settings = app.state.settings
    original = settings.semantic_sim_threshold

    resp = client.post(
        "/admin/cache/threshold",
        json={"threshold": 0.75},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 0.75
    assert settings.semantic_sim_threshold == 0.75

    settings.semantic_sim_threshold = original


def test_threshold_endpoint_reports_sweep_artifact(client):
    resp = client.get(
        "/admin/cache/threshold", headers={"Authorization": "Bearer test-admin-token"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < body["threshold"] <= 1
    assert body["margin_guard"] >= 0
    # Day 2 writes this; until then it is legitimately null.
    assert "last_sweep" in body


def test_cache_can_be_cleared(client, auth_headers):
    client.post(
        "/v1/chat/completions", json=chat_payload("clearable"), headers=auth_headers
    )
    resp = client.delete(
        "/admin/cache?scope=all", headers={"Authorization": "Bearer test-admin-token"}
    )
    assert resp.status_code == 200

    again = client.post(
        "/v1/chat/completions", json=chat_payload("clearable"), headers=auth_headers
    )
    assert cache_header(again) == "MISS"
    assert len(FakeProvider.calls) == 2
