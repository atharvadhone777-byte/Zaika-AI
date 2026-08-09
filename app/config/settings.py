"""
Environment-driven settings. Using pydantic-settings (not raw os.environ
reads scattered through the codebase) means every config value is
declared, typed, and validated in one place, and can be overridden by
environment variables or a .env file without touching code - the standard
12-factor-app approach, and directly relevant to the "environment
variables" deployment requirement.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    environment: str = "development"
    log_level: str = "INFO"

    models_dir: Path = Path(__file__).resolve().parents[2] / "models" / "v1"

    # CORS: explicit allowlist rather than "*" - the frontend origin(s)
    # that are actually allowed to call this API in production. "*" is
    # convenient in a demo but is worth being able to explain why it's
    # NOT what a production config should use (it disables a real browser
    # security boundary for no benefit once the frontend origin is known).
    cors_allowed_origins: list[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
    ]

    default_top_k: int = 5
    max_top_k: int = 20


settings = Settings()
