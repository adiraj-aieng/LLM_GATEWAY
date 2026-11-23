"""Application settings.

Every tunable lives here so there are no magic constants scattered through the codebase.
Loaded from environment / `.env`, validated at import time, and *fail fast* on bad config —
a gateway that boots with a default admin token is worse than one that refuses to boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_MAP: dict[str, str] = {
    "fast": "llama3.2:3b",
    "smart": "llama3.2:3b",
    "coder": "llama3.2:3b",
    "auto": "llama3.2:3b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app -----------------------------------------------------------------
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    admin_token: str = "change-me-admin-token"
    #: Mixed into the API-key hash. A leaked database alone then can't be brute-forced
    #: into usable keys. High-entropy keys make a slow KDF unnecessary here.
    api_key_pepper: str = "change-me-api-key-pepper"
    public_base_url: str = "http://localhost:8000"

    # --- database ------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./gateway.db"

    # --- redis ---------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- upstream ------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    upstream_timeout_s: float = 60.0
    upstream_idle_timeout_s: float = 30.0
    circuit_failure_threshold: int = 5
    circuit_cooldown_s: float = 30.0
    retry_max_attempts: int = 2
    model_map: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_MAP))

    # --- exact cache ---------------------------------------------------------
    exact_cache_enabled: bool = True
    exact_cache_ttl_s: int = 86400
    exact_cache_max_entries: int = 10000

    # --- semantic cache ------------------------------------------------------
    semantic_cache_enabled: bool = True
    semantic_cache_ttl_s: int = 604800
    semantic_sim_threshold: float = 0.92
    semantic_margin_guard: float = 0.02
    semantic_max_vectors: int = 50000
    embed_provider: Literal["fastembed", "ollama"] = "fastembed"
    cache_max_temp: float = 0.2

    # --- rate limiting -------------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 100000
    rate_limit_refill_per_s: float = 1666.0
    rate_limit_weight_prompt: float = 1.0
    rate_limit_weight_completion: float = 1.0
    concurrency_limit_per_key: int = 8

    # --- telemetry -----------------------------------------------------------
    log_queue_maxsize: int = 10000
    log_flush_interval_s: float = 1.0
    log_flush_batch: int = 500
    prompt_preview_len: int = 200

    # =======================================================================
    # validation
    # =======================================================================
    @model_validator(mode="after")
    def _check_production_safety(self) -> "Settings":
        if self.app_env == "production":
            if self.admin_token == "change-me-admin-token":
                raise ValueError(
                    "ADMIN_TOKEN must be set to a real value when APP_ENV=production"
                )
            if self.api_key_pepper == "change-me-api-key-pepper":
                raise ValueError(
                    "API_KEY_PEPPER must be set to a real value when APP_ENV=production"
                )
            if self.database_url.startswith("sqlite"):
                raise ValueError(
                    "DATABASE_URL must be Postgres when APP_ENV=production "
                    "(SQLite cannot handle concurrent telemetry writes)"
                )
        if not 0.0 < self.semantic_sim_threshold <= 1.0:
            raise ValueError("SEMANTIC_SIM_THRESHOLD must be in (0, 1]")
        if self.semantic_margin_guard < 0.0:
            raise ValueError("SEMANTIC_MARGIN_GUARD must be >= 0")
        if self.rate_limit_capacity <= 0:
            raise ValueError("RATE_LIMIT_CAPACITY must be > 0")
        if self.rate_limit_refill_per_s <= 0:
            raise ValueError("RATE_LIMIT_REFILL_PER_S must be > 0")
        return self

    # =======================================================================
    # derived
    # =======================================================================
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    def resolve_model(self, logical: str) -> str:
        """Map a logical model name (`auto`, `fast`, …) to an upstream model."""
        return self.model_map.get(logical, logical)

    def logical_models(self) -> list[str]:
        return sorted(self.model_map)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. `lru_cache` so `.env` is read once per process."""
    return Settings()


def clear_settings_cache() -> None:
    """For tests that mutate env vars between cases."""
    get_settings.cache_clear()
