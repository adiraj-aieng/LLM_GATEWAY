"""Rate limiting: Lua semantics, atomicity, and the HTTP contract.

The atomicity test is the important one — it proves a denied request cannot burn another
bucket's tokens, which is what "all-or-nothing across three buckets" has to mean.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from gateway.config import Settings
from gateway.core.auth import Principal
from gateway.core.rate_limit import RateLimiter
from tests.conftest import ADMIN_HEADERS, chat_payload

ADMIN = {"Authorization": "Bearer test-admin-token"}


def _principal(**overrides) -> Principal:
    base = dict(
        user_id=1,
        api_key_id=1,
        label="t",
        plan="free",
        rate_limit_tokens_min=100_000,
        daily_token_budget=None,
        allowed_models=("auto", "fast", "smart", "coder"),
    )
    base.update(overrides)
    return Principal(**base)


async def _limiter() -> RateLimiter:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
    return RateLimiter(redis, Settings(app_env="test"))


# =======================================================================
# Lua script semantics
# =======================================================================
async def test_consume_within_capacity_is_allowed():
    rl = await _limiter()
    keys = ["gw:t:u", "gw:t:k", "gw:t:g"]
    # cost=10, ttl, then (capacity, rate, enabled) x3
    args = [10, 60_000, 1000, 1000, 1, 1000, 1000, 1, 1000, 1000, 1]
    raw = await rl._eval(keys, args)
    assert int(raw[0]) == 1
    assert int(raw[1]) >= 0


async def test_exceeding_capacity_is_denied_with_retry_after():
    rl = await _limiter()
    keys = ["gw:t:u", "gw:t:k", "gw:t:g"]
    args = [50, 60_000, 40, 1, 1, 1000, 1000, 1, 1000, 1000, 1]
    raw = await rl._eval(keys, args)
    assert int(raw[0]) == 0, "bucket smaller than cost must deny"
    assert int(raw[2]) >= 1, "429 must carry a positive retry-after"


async def test_denied_request_does_not_charge_any_bucket():
    """All-or-nothing: a denial in one bucket must leave the others untouched."""
    rl = await _limiter()
    keys = ["gw:t:u", "gw:t:k", "gw:t:g"]
    # bucket 2 has capacity 0 → always denies; buckets 1 and 3 are full.
    args = [5, 60_000, 1000, 1000, 1, 0, 0, 1, 1000, 1000, 1]

    raw = await rl._eval(keys, args)
    assert int(raw[0]) == 0, "expected denial from the zero-capacity bucket"

    tokens_1 = float(await rl._redis.hget(keys[0], "tokens"))
    tokens_3 = float(await rl._redis.hget(keys[2], "tokens"))
    assert tokens_1 == pytest.approx(1000, abs=1), (
        f"bucket 1 was charged despite the denial ({tokens_1})"
    )
    assert tokens_3 == pytest.approx(1000, abs=1), (
        f"bucket 3 was charged despite the denial ({tokens_3})"
    )


async def test_successful_consume_charges_exactly_once():
    rl = await _limiter()
    keys = ["gw:t:u", "gw:t:k", "gw:t:g"]
    args = [100, 60_000, 1000, 1000, 1, 1000, 1000, 1, 1000, 1000, 1]
    await rl._eval(keys, args)
    tokens = float(await rl._redis.hget(keys[0], "tokens"))
    assert tokens == pytest.approx(900, abs=1)


async def test_bucket_refills_over_time():
    rl = await _limiter()
    keys = ["gw:t:u", "gw:t:k", "gw:t:g"]
    args = lambda: [10, 60_000, 10, 200, 1, 10, 200, 1, 10, 200, 1]  # noqa: E731

    # First consume drains the bucket (capacity 10, cost 10).
    assert int((await rl._eval(keys, args()))[0]) == 1

    # Immediately retrying must be denied — no refill time has elapsed.
    assert int((await rl._eval(keys, args()))[0]) == 0, (
        "an empty bucket must deny before any refill time passes"
    )

    # 0.1s at 200 tokens/s = +20, capped at capacity 10 → full again.
    await asyncio.sleep(0.1)
    refilled = await rl._eval(keys, args())
    assert int(refilled[0]) == 1, "bucket must refill after the cooldown"
    assert int(refilled[1]) >= 0


async def test_refund_returns_tokens_without_exceeding_capacity():
    rl = await _limiter()
    p = _principal()
    await rl.check(p, 100)
    before = float(await rl._redis.hget(f"gw:rl:key:{p.api_key_id}", "tokens"))
    await rl.refund(p, 50)
    after = float(await rl._redis.hget(f"gw:rl:key:{p.api_key_id}", "tokens"))
    assert after > before, "refund must give tokens back"

    await rl.refund(p, 10_000_000)
    capped = float(await rl._redis.hget(f"gw:rl:key:{p.api_key_id}", "tokens"))
    assert capped <= p.rate_limit_tokens_min, "refund must not overfill the bucket"


# =======================================================================
# HTTP contract
# =======================================================================
def test_tiny_key_is_rate_limited(client):
    resp = client.post(
        "/admin/api-keys",
        json={
            "label": "tiny",
            "email": "tiny@example.com",
            "rate_limit_tokens_per_min": 10,
        },
        headers=ADMIN_HEADERS,
    )
    key = resp.json()["key"]

    r = client.post(
        "/v1/chat/completions",
        json=chat_payload("hello"),
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 429
    assert r.headers.get("retry-after"), "429 must tell the client when to retry"
    body = r.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["code"] == "rate_limit_exceeded"


def test_generous_key_is_not_rate_limited(client, auth_headers):
    for i in range(5):
        r = client.post(
            "/v1/chat/completions",
            json=chat_payload(f"hello {i}"),
            headers=auth_headers,
        )
        assert r.status_code == 200


def test_rate_limit_headers_present(client, auth_headers):
    r = client.post(
        "/v1/chat/completions", json=chat_payload("hi"), headers=auth_headers
    )
    assert "x-ratelimit-limit" in r.headers
    assert "x-gateway-cache" in r.headers
    assert "x-request-id" in r.headers


def test_admin_exposes_rejection_stats(client, auth_headers):
    client.post(
        "/admin/api-keys",
        json={"label": "tiny", "email": "stats-tiny@example.com",
              "rate_limit_tokens_per_min": 5},
        headers=ADMIN_HEADERS,
    )
    stats = client.get("/admin/stats?window=1h", headers=ADMIN_HEADERS).json()
    assert "rate_limit" in stats
    assert "rejected" in stats["rate_limit"]
