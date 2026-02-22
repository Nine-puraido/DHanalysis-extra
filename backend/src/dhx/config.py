"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (DHextra/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Supabase (same project as DHextra, uses dhx schema) ---
    supabase_url: str
    supabase_service_role_key: str

    # --- Storage buckets (dhx- prefix to separate from DHextra) ---
    supabase_bucket_raw_api: str = "dhx-raw-api"
    supabase_bucket_features: str = "dhx-features"
    supabase_bucket_models: str = "dhx-models"
    supabase_bucket_reports: str = "dhx-reports"

    # --- External APIs ---
    sofascore_base_url: str = "https://api.sofascore.com/api/v1"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    default_markets: str = "1x2,ah,totals,btts"

    @property
    def markets(self) -> list[str]:
        return [m.strip() for m in self.default_markets.split(",")]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
