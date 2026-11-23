"""Authentication and authorization."""

from __future__ import annotations

from tests.conftest import ADMIN_HEADERS, chat_payload


def test_missing_key_rejected(client):
    resp = client.post("/v1/chat/completions", json=chat_payload("hello"))
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["code"] == "missing_api_key"
    assert "request_id" in body


def test_malformed_key_rejected(client):
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hello"),
        headers={"Authorization": "Bearer not-a-gateway-key"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_wrong_scheme_rejected(client):
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hello"),
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert resp.status_code == 401


def test_unknown_valid_shaped_key_rejected(client):
    # Right shape, never issued — must not be distinguishable from a revoked key.
    resp = client.post(
        "/v1/chat/completions",
        json=chat_payload("hello"),
        headers={"Authorization": "Bearer sk-gw-" + "a" * 48},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_valid_key_accepted(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions", json=chat_payload("hello"), headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"]


def test_key_is_not_retrievable(client, api_key):
    """The plaintext is shown once; listing keys must expose only the prefix."""
    resp = client.get("/admin/api-keys", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items, "expected at least one key"
    assert api_key not in str(items)
    assert items[0]["prefix"].startswith("sk-gw-")


def test_revoked_key_stops_working(client, api_key, auth_headers):
    listed = client.get("/admin/api-keys", headers=ADMIN_HEADERS).json()["items"]
    key_id = next(k["id"] for k in listed if k["prefix"] == api_key[:9])

    assert (
        client.post(
            "/v1/chat/completions", json=chat_payload("hi"), headers=auth_headers
        ).status_code
        == 200
    )

    assert (
        client.delete(f"/admin/api-keys/{key_id}", headers=ADMIN_HEADERS).status_code
        == 200
    )
    resp = client.post(
        "/v1/chat/completions", json=chat_payload("hi"), headers=auth_headers
    )
    assert resp.status_code == 401


def test_restricted_key_cannot_use_disallowed_model(client):
    resp = client.post(
        "/admin/api-keys",
        json={
            "label": "restricted",
            "email": "restricted@example.com",
            "allowed_models": ["fast"],
        },
        headers=ADMIN_HEADERS,
    )
    key = resp.json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    ok = client.post(
        "/v1/chat/completions", json=chat_payload("hi", model="fast"), headers=headers
    )
    assert ok.status_code == 200

    denied = client.post(
        "/v1/chat/completions", json=chat_payload("hi", model="smart"), headers=headers
    )
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "model_not_allowed"


def test_admin_requires_token(client):
    assert client.get("/admin/stats").status_code == 403
    assert (
        client.get(
            "/admin/stats", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 403
    )
    assert client.get("/admin/stats", headers=ADMIN_HEADERS).status_code == 200


def test_healthz_needs_no_auth(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
