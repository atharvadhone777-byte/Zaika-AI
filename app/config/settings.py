from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    environment: str = "development"
    log_level: str = "INFO"

    models_dir: Path = Path(__file__).resolve().parents[2] / "models" / "v1"

    cors_allowed_origins: list[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
    ]

    default_top_k: int = 5
    max_top_k: int = 20


settings = Settings()
