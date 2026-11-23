"""OpenAI-shaped error envelope + FastAPI exception handlers.

Clients branch on ``error.type`` and ``error.code`` — LangChain, the OpenAI SDKs and
OpenRouter all do. Returning a FastAPI default ``{"detail": "..."}`` breaks them silently,
which is why this file exists rather than letting exceptions bubble up.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# --- error types (mirror OpenAI's) ----------------------------------------
INVALID_REQUEST = "invalid_request_error"
AUTHENTICATION = "authentication_error"
PERMISSION = "permission_error"
NOT_FOUND = "not_found_error"
RATE_LIMIT = "rate_limit_error"
SERVER = "server_error"
API = "api_error"


class GatewayError(Exception):
    """Base for every error we intentionally raise."""

    status_code: int = 500
    type: str = SERVER
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, **extra: Any):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.extra = extra

    def to_payload(self, request_id: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.extra.get("param"),
                "code": self.code,
            }
        }
        if request_id:
            body["request_id"] = request_id
        return body


class InvalidRequestError(GatewayError):
    status_code = 400
    type = INVALID_REQUEST
    code = "invalid_body"


class AuthenticationError(GatewayError):
    status_code = 401
    type = AUTHENTICATION
    code = "invalid_api_key"


class PermissionError_(GatewayError):
    status_code = 403
    type = PERMISSION
    code = "model_not_allowed"


class NotFoundError(GatewayError):
    status_code = 404
    type = NOT_FOUND
    code = "not_found"


class RateLimitError(GatewayError):
    status_code = 429
    type = RATE_LIMIT
    code = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after: int, **extra: Any):
        super().__init__(message, retry_after=retry_after, **extra)
        self.retry_after = retry_after


class NoProviderAvailableError(GatewayError):
    status_code = 503
    type = SERVER
    code = "no_provider_available"


class UpstreamTimeoutError(GatewayError):
    status_code = 504
    type = API
    code = "upstream_timeout"


class UpstreamError(GatewayError):
    status_code = 502
    type = API
    code = "upstream_error"


# --- handlers -------------------------------------------------------------
def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _headers_for(exc: GatewayError) -> dict[str, str]:
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(exc.retry_after)
    return headers


async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_payload(_request_id(request)),
        headers=_headers_for(exc),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Translate Starlette's HTTPException into the OpenAI envelope too."""
    code_map = {
        400: (INVALID_REQUEST, "invalid_body"),
        401: (AUTHENTICATION, "invalid_api_key"),
        403: (PERMISSION, "permission_error"),
        404: (NOT_FOUND, "route_not_found"),
        405: (INVALID_REQUEST, "method_not_allowed"),
        413: (INVALID_REQUEST, "prompt_too_long"),
        429: (RATE_LIMIT, "rate_limit_exceeded"),
        500: (SERVER, "internal_error"),
    }
    etype, ecode = code_map.get(exc.status_code, (SERVER, "internal_error"))
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    payload = {
        "error": {
            "message": message,
            "type": etype,
            "param": None,
            "code": ecode,
        },
        "request_id": _request_id(request),
    }
    return JSONResponse(status_code=exc.status_code, content=payload)


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - safety net
    from gateway.logging import get_logger

    get_logger("gateway").exception("unhandled_error", error=str(exc))
    payload = {
        "error": {
            "message": "An internal error occurred.",
            "type": SERVER,
            "param": None,
            "code": "internal_error",
        },
        "request_id": _request_id(request) or str(uuid.uuid4()),
    }
    return JSONResponse(status_code=500, content=payload)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
