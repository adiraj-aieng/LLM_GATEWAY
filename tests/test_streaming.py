"""Streaming: SSE framing, usage capture, and cache replay."""

from __future__ import annotations

import json

from tests.conftest import FakeProvider, chat_payload


def _parse_sse(raw: str) -> tuple[list[dict], list[dict]]:
    """Split SSE text into (data events, error events)."""
    events: list[dict] = []
    errors: list[dict] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            errors.append(parsed["error"])
        else:
            events.append(parsed)
    return events, errors


def test_streaming_returns_sse_with_done_sentinel(client, auth_headers):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_payload("stream this please", stream=True),
        headers=auth_headers,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = resp.read().decode("utf-8")

    assert raw.endswith("data: [DONE]\n\n"), "stream must terminate with [DONE]"

    events, errors = _parse_sse(raw)
    assert not errors
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices")
    )
    assert content, "expected content deltas"
    assert "stream this please" in content


def test_stream_finish_reason_is_emitted(client, auth_headers):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_payload("finish me", stream=True),
        headers=auth_headers,
    ) as resp:
        raw = resp.read().decode("utf-8")

    events, _ = _parse_sse(raw)
    reasons = [
        e["choices"][0].get("finish_reason")
        for e in events
        if e.get("choices")
    ]
    assert "stop" in reasons


def test_stream_includes_usage_chunk(client, auth_headers):
    """Without this, every streamed request logs 0 tokens and cost accounting is fiction."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_payload("count my tokens", stream=True),
        headers=auth_headers,
    ) as resp:
        raw = resp.read().decode("utf-8")

    events, _ = _parse_sse(raw)
    usage_chunks = [e for e in events if e.get("usage")]
    assert usage_chunks, "expected a final usage chunk"
    usage = usage_chunks[-1]["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["completion_tokens"] > 0


def test_stream_hit_replays_without_upstream(client, auth_headers):
    body = chat_payload("cache this stream", stream=True)
    with client.stream("POST", "/v1/chat/completions", json=body, headers=auth_headers):
        pass
    assert len(FakeProvider.calls) == 1

    with client.stream("POST", "/v1/chat/completions", json=body, headers=auth_headers) as resp:
        assert resp.headers.get("x-gateway-cache") == "EXACT_HIT"
        raw = resp.read().decode("utf-8")

    assert len(FakeProvider.calls) == 1, "cache hit must not reach the upstream"
    events, _ = _parse_sse(raw)
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices")
    )
    assert content


def test_stream_and_non_stream_share_the_cache(client, auth_headers):
    """A streamed answer must be storable, so the two paths use one cache."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_payload("shared cache", stream=True),
        headers=auth_headers,
    ) as resp:
        assert resp.headers.get("x-gateway-cache") == "MISS"

    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("shared cache"),
        headers=auth_headers,
    )
    assert resp.headers.get("x-gateway-cache") == "EXACT_HIT"
    assert len(FakeProvider.calls) == 1


def test_upstream_failure_becomes_an_sse_error_not_a_broken_stream(client, auth_headers):
    """Raising after bytes reached the client would duplicate content on retry."""
    from gateway.providers.base import ProviderError

    FakeProvider.fail_with = ProviderError(
        "upstream exploded", status_code=503, retryable=True, provider="fake"
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=chat_payload("will fail", stream=True),
        headers=auth_headers,
    ) as resp:
        # The SSE has already begun; the failure is reported in-band.
        raw = resp.read().decode("utf-8")

    _, errors = _parse_sse(raw)
    assert errors, "expected an in-band error event"
    assert raw.rstrip().endswith("data: [DONE]")


def test_upstream_failure_raises_openai_error_when_not_streaming(client, auth_headers):
    from gateway.providers.base import ProviderError

    FakeProvider.fail_with = ProviderError(
        "upstream exploded", status_code=503, retryable=False, provider="fake"
    )
    resp = client.post(
        "/v1/chat/completions", json=chat_payload("will fail"), headers=auth_headers
    )
    assert resp.status_code in (502, 503, 504)
    assert resp.json()["error"]["type"] in ("api_error", "server_error")
