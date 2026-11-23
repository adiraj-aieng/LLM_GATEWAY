"""Logging setup: structured JSON via structlog, with `request_id` in every record.

`request_id` is stored in a contextvar by middleware so any log line emitted while
handling a request carries it automatically — no threading it through every call.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

import structlog

#: Set by middleware at the start of each request, cleared at the end.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# Fields attached to every log line.
_SHARED = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def _add_request_id(
    _logger: object, _method: str, event_dict: dict
) -> dict:
    rid = request_id_ctx.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog + stdlib logging. Safe to call more than once."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    processors = list(_SHARED)
    processors.append(_add_request_id)
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)  # type: ignore[union-attr]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(**initial)
