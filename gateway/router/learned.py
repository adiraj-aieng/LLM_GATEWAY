"""Learned router: a small classifier over handcrafted features.

Deliberately **not** a fine-tuned LLM. A gateway must decide in well under a millisecond
on the hot path, so this is a scikit-learn classifier over the fixed-order feature vector
from :mod:`gateway.router.features`.

If no trained artifact exists, the router reports itself unavailable and
:class:`~gateway.router.policy.PolicyRouter` falls back to rules — the gateway must never
depend on a file that might not be there.

Artifact contract (written by ``ml/train_router.py``):

.. code-block:: json

    {
      "model": "<pickled estimator>",
      "feature_order": ["prompt_tokens", "char_len", ...],
      "threshold": 0.42,
      "metrics": {"f1": 0.91, "precision": 0.93, "recall": 0.89},
      "trained_at": "2026-11-13T09:00:00Z"
    }

``feature_order`` is read back rather than assumed — if the training code reorders
features, inference uses the stored order instead of silently scoring garbage.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from gateway.router.base import RouteContext, RouteDecision, Router
from gateway.router.features import FEATURE_ORDER

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "router_model.pkl"


class LearnedRouter(Router):
    name = "learned"

    def __init__(self, settings, *, artifact_path: Path | None = None):
        self._settings = settings
        self._path = Path(artifact_path or DEFAULT_ARTIFACT)
        self._model: Any = None
        self._order: list[str] = FEATURE_ORDER
        self._threshold: float = 0.5
        self._metrics: dict[str, Any] = {}
        self._error: str | None = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load the artifact. Returns False (rather than raising) if it's missing."""
        if not self._path.exists():
            self._error = f"artifact not found: {self._path.name}"
            return False
        try:
            with self._path.open("rb") as fh:
                payload = pickle.load(fh)
            self._model = payload["model"]
            self._order = list(payload.get("feature_order", FEATURE_ORDER))
            self._threshold = float(payload.get("threshold", 0.5))
            self._metrics = dict(payload.get("metrics", {}))
            self._error = None
            return True
        except Exception as exc:  # corrupt / version-mismatched artifact
            self._model = None
            self._error = f"failed to load artifact: {exc}"
            return False

    # ------------------------------------------------------------------
    def predict_proba(self, feats: dict[str, Any]) -> float | None:
        """P(needs_strong), or None when the router is unavailable."""
        if self._model is None and not self.load():
            return None
        try:
            vector = [float(feats.get(name, 0)) for name in self._order]
            proba = self._model.predict_proba([vector])[0]
            # Classes are [0, 1] for a binary "needs strong model" label.
            classes = list(getattr(self._model, "classes_", [0, 1]))
            return float(proba[list(classes).index(1)]) if 1 in classes else float(proba[-1])
        except Exception as exc:
            self._error = f"inference failed: {exc}"
            return None

    # ------------------------------------------------------------------
    def route(self, ctx: RouteContext) -> RouteDecision:
        if self._model is None and not self.load():
            raise RuntimeError(self._error or "router unavailable")

        proba = self.predict_proba(ctx.features)
        if proba is None:
            raise RuntimeError(self._error or "inference failed")

        needs_strong = proba >= self._threshold
        logical = "smart" if needs_strong else "fast"
        detail = (
            f"P(needs_strong)={proba:.3f} at threshold {self._threshold:.3f}"
        )
        return RouteDecision(
            logical_model=logical,
            upstream_model=self._settings.resolve_model(logical),
            reason="classifier",
            confidence=round(proba if needs_strong else 1.0 - proba, 4),
            detail=detail,
            contributions=self._explain(ctx.features),
            router=self.name,
        )

    # ------------------------------------------------------------------
    def _explain(self, feats: dict[str, Any]) -> list[dict[str, Any]]:
        """Top feature contributions, for the dashboard "why" panel.

        Supports linear models exactly (coef × value). Tree ensembles return an empty
        list rather than a fabricated explanation.
        """
        coef = getattr(self._model, "coef_", None)
        if coef is None:
            return []
        try:
            weights = list(coef[0])
            pairs = []
            for name, weight in zip(self._order, weights):
                value = float(feats.get(name, 0))
                pairs.append(
                    {"feature": name, "weight": round(float(weight), 5),
                     "value": value, "contribution": round(float(weight) * value, 5)}
                )
            return sorted(pairs, key=lambda p: abs(p["contribution"]), reverse=True)[:8]
        except Exception:  # pragma: no cover
            return []
