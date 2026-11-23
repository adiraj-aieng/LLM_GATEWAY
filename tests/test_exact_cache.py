"""Exact cache: key construction, normalization, and bypass rules.

These are the tests that catch the bugs which make a cache *dangerous* rather than
merely ineffective — specifically, serving one caller's answer to a different request.
"""

from __future__ import annotations

from tests.conftest import FakeProvider, cache_header, chat_payload


def test_second_identical_request_is_served_from_cache(client, auth_headers):
    body = chat_payload("what is a token bucket")

    first = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert first.status_code == 200
    assert cache_header(first) == "MISS"
    assert len(FakeProvider.calls) == 1

    second = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert second.status_code == 200
    assert cache_header(second) == "EXACT_HIT"
    assert len(FakeProvider.calls) == 1, "cache hit must not reach the upstream"

    # A hit returns the same content as the original response.
    assert (
        second.json()["choices"][0]["message"]["content"]
        == first.json()["choices"][0]["message"]["content"]
    )


def test_whitespace_differences_still_hit(client, auth_headers):
    """NFKC + whitespace collapsing: reformatting a prompt should not miss the cache."""
    client.post(
        "/v1/chat/completions",
        json=chat_payload("  explain   the   gateway  "),
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("explain the gateway"),
        headers=auth_headers,
    )
    assert cache_header(resp) == "EXACT_HIT"
    assert len(FakeProvider.calls) == 1


def test_different_system_prompt_is_a_different_request(client, auth_headers):
    """The classic semantic-cache bug: shared cache across apps with different systems."""
    client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "summarize this",
            messages=[
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "summarize this"},
            ],
        ),
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "summarize this",
            messages=[
                {"role": "system", "content": "You are a terse accountant."},
                {"role": "user", "content": "summarize this"},
            ],
        ),
        headers=auth_headers,
    )
    assert cache_header(resp) == "MISS", "differing system prompt must not hit"
    assert len(FakeProvider.calls) == 2


def test_multi_turn_context_changes_the_key(client, auth_headers):
    client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "and then?",
            messages=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "and then?"},
            ],
        ),
        headers=auth_headers,
    )
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "and then?",
            messages=[
                {"role": "user", "content": "a totally different question"},
                {"role": "assistant", "content": "a different answer"},
                {"role": "user", "content": "and then?"},
            ],
        ),
        headers=auth_headers,
    )
    assert cache_header(resp) == "MISS"
    assert len(FakeProvider.calls) == 2


def test_high_temperature_bypasses_cache(client, auth_headers):
    """Sampling means 'correct' isn't unique; returning one stored sample would lie."""
    body = chat_payload("write a poem", temperature=1.0)
    client.post("/v1/chat/completions", json=body, headers=auth_headers)
    resp = client.post("/v1/chat/completions", json=body, headers=auth_headers)

    assert cache_header(resp) == "BYPASS"
    assert len(FakeProvider.calls) == 2, "bypass must never read the cache"


def test_low_temperature_is_cacheable(client, auth_headers):
    body = chat_payload("summarize the log", temperature=0.0)
    client.post("/v1/chat/completions", json=body, headers=auth_headers)
    resp = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert cache_header(resp) == "EXACT_HIT"


def test_client_can_explicitly_bypass(client, auth_headers):
    body = chat_payload("do not cache me", gateway={"no_cache": True})
    client.post("/v1/chat/completions", json=body, headers=auth_headers)
    resp = client.post("/v1/chat/completions", json=body, headers=auth_headers)
    assert cache_header(resp) == "BYPASS"
    assert len(FakeProvider.calls) == 2


def test_cached_response_carries_gateway_metadata(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions", json=chat_payload("metadata please"), headers=auth_headers
    )
    payload = resp.json()
    assert payload["gateway"]["cache"] == "MISS"
    assert payload["gateway"]["cost_basis"] == "local_inference"
    assert payload["usage"]["total_tokens"] > 0


def test_unsupported_parameters_rejected(client, auth_headers):
    """Silent divergence from OpenAI is worse than an explicit 400."""
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hi", n=3),
        headers=auth_headers,
    )
    assert resp.status_code == 400

    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hi", logprobs=True),
        headers=auth_headers,
    )
    assert resp.status_code == 400

    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hi", presence_penalty=0.5),
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_error_envelope_is_openai_shaped(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions", json={"model": "auto"}, headers=auth_headers
    )
    assert resp.status_code == 400
    body = resp.json()
    assert set(body["error"]) >= {"message", "type", "param", "code"}
    assert body["error"]["type"] == "invalid_request_error"
