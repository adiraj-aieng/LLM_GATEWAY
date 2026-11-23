"""Provider registry: logical model → (provider, upstream model), plus fallback chains.

Why the indirection? Clients talk to stable *logical* names (``auto``, ``fast``,
``coder``). Those resolve to real upstream models through ``MODEL_MAP``, so swapping
``llama3.2:3b`` for a bigger model is a config change with zero client impact.

``auto`` is special: it is resolved by the router at request time.
"""

from __future__ import annotations

from dataclasses import dataclass

from gateway.config import Settings
from gateway.logging import get_logger
from gateway.providers.base import Provider
from gateway.providers.openai_compat import OpenAICompatProvider
from gateway.providers.resilience import BreakerRegistry

log = get_logger("gateway.registry")


@dataclass(frozen=True)
class Route:
    """A concrete upstream target."""

    provider: Provider
    upstream_model: str
    logical_model: str
    provider_name: str

    @property
    def breaker_key(self) -> str:
        return f"{self.provider_name}:{self.upstream_model}"


class ProviderRegistry:
    def __init__(self, settings: Settings, *, breakers: BreakerRegistry | None = None):
        self.settings = settings
        self.breakers = breakers or BreakerRegistry(settings)
        self._providers: dict[str, Provider] = {}

        # One adapter for now. Adding OpenAI/Groq later is another line here, because
        # they share the same wire format.
        self._providers["ollama"] = OpenAICompatProvider(
            settings, base_url=settings.ollama_base_url, name="ollama"
        )
        self._default = "ollama"

    @property
    def providers(self) -> dict[str, Provider]:
        return self._providers

    def get(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown provider `{name}`") from None

    # ------------------------------------------------------------------
    def resolve(self, logical_model: str) -> Route:
        """Resolve a logical model to a concrete upstream target.

        Raises ``KeyError`` if the logical name is unknown — the API layer turns that
        into a 404 in the OpenAI envelope.
        """
        upstream = self.settings.resolve_model(logical_model)
        if not upstream:
            raise KeyError(logical_model)
        provider = self._providers[self._default]
        return Route(
            provider=provider,
            upstream_model=upstream,
            logical_model=logical_model,
            provider_name=self._default,
        )

    # ------------------------------------------------------------------
    def fallback_chain(self, route: Route, *, exclude: set[str] | None = None) -> list[Route]:
        """Ordered alternatives to try if ``route`` is unavailable.

        With a single configured model this is empty — which is correct. A fake fallback
        that lands on the same model would be worse than none: it hides the outage.
        """
        exclude = set(exclude or ())
        exclude.add(route.upstream_model)
        chain: list[Route] = []
        for logical, upstream in self.settings.model_map.items():
            if upstream in exclude:
                continue
            exclude.add(upstream)
            chain.append(
                Route(
                    provider=self._providers[self._default],
                    upstream_model=upstream,
                    logical_model=logical,
                    provider_name=self._default,
                )
            )
        return chain

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for name, provider in self._providers.items():
            ok = False
            try:
                ok = await provider.health()
            except Exception as exc:  # pragma: no cover
                log.warning("provider_health_error", provider=name, error=str(exc))
            out[name] = {"available": ok, "circuit": self.breakers.get(name).snapshot()}
        return out

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()

    def catalog(self) -> list[dict[str, object]]:
        """Public model catalog — what ``GET /v1/models`` returns."""
        tiers = {"fast": "cheap", "smart": "mid", "coder": "specialist", "auto": "router"}
        seen: set[str] = set()
        out: list[dict[str, object]] = []
        for logical in sorted(self.settings.model_map):
            if logical in seen:
                continue
            seen.add(logical)
            upstream = self.settings.resolve_model(logical)
            out.append(
                {
                    "id": logical,
                    "object": "model",
                    "owned_by": "gateway",
                    "gateway": {
                        "upstream": "router" if logical == "auto" else self._default,
                        "model": "resolved-at-runtime" if logical == "auto" else upstream,
                        "tier": tiers.get(logical, "mid"),
                        "available": True,
                    },
                }
            )
        return out
