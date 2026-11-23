"""In-process metrics: counters, gauges and latency histograms.

Kept local and dependency-free on purpose. The dashboard reads these through
``/admin/stats``; promoting them to Prometheus later is a change of *sink*, not of call
sites, because every recording goes through :meth:`MetricsRegistry`.

Latency is tracked as a ring buffer of recent samples rather than fixed buckets, so
percentiles are exact for the window instead of an approximation from bucket edges.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from gateway.telemetry.queue import RollingWindow


class MetricsRegistry:
    """Thread-safe (uvicorn may run sync endpoints in a threadpool)."""

    def __init__(self, *, window_capacity: int = 4096):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._latency: dict[str, RollingWindow] = {}
        self._capacity = window_capacity

    # ------------------------------------------------------------------
    def incr(self, name: str, value: int = 1, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value

    def observe(self, name: str, value_ms: float, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            window = self._latency.get(key)
            if window is None:
                window = RollingWindow(self._capacity)
                self._latency[key] = window
            window.add(value_ms)

    # ------------------------------------------------------------------
    def counter(self, name: str, **labels: Any) -> int:
        with self._lock:
            return int(self._counters.get(_key(name, labels), 0))

    def percentiles(self, name: str, **labels: Any) -> dict[str, float]:
        with self._lock:
            window = self._latency.get(_key(name, labels))
        if window is None:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        count = window.count()
        if count == 0:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        return {
            "count": count,
            "p50": round(window.percentile(50), 2),
            "p95": round(window.percentile(95), 2),
            "p99": round(window.percentile(99), 2),
            "mean": round(window.mean(), 2),
        }

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            latency_keys = list(self._latency)
        latency = {}
        for key in latency_keys:
            latency[key] = self.percentiles(_split(key)[0], **_labels_of(key))
        return {"counters": counters, "gauges": gauges, "latency": latency}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latency.clear()


def _key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    parts = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{parts}}}"


def _split(key: str) -> tuple[str, str]:
    if "{" not in key:
        return key, ""
    name, rest = key.split("{", 1)
    return name, rest.rstrip("}")


def _labels_of(key: str) -> dict[str, Any]:
    _, rest = _split(key)
    if not rest:
        return {}
    out: dict[str, Any] = {}
    for pair in rest.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out
