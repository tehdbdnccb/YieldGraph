# core/config.py
from __future__ import annotations
import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, Field


class AppConfig(BaseModel):
    """Immutable data structure representing the validated application state."""
    qloo_api_key: SecretStr = Field(..., min_length=10)
    qloo_base_url: str = Field(default="https://hackathon.api.qloo.com")
    groq_api_key: SecretStr = Field(..., min_length=10)
    groq_model: str = Field(default="openai/gpt-oss-120b")
    qloo_cache_mode: str = Field(default="live_cached")


class SettingsLoader:
    """Parses and bounds environment variables into deterministic configurations."""

    def load_env(self) -> AppConfig:
        """Extracts configuration from the environment variables."""
        load_dotenv(override=True)
        raw_url = (
            os.getenv("QLOO_BASE_URL")
            or os.getenv("Qloo Hackathon API Server URL")
            or "https://hackathon.api.qloo.com"
        ).strip()
        return AppConfig(
            qloo_api_key=os.getenv("QLOO_API_KEY", "").strip(),
            qloo_base_url=raw_url.rstrip("/"),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            qloo_cache_mode=os.getenv("QLOO_CACHE_MODE", "live_cached").strip(),
        )

    def validate_keys(self, config: AppConfig) -> AppConfig:
        """Enforces domain constraints on the loaded configuration state."""
        valid_modes = {"live", "live_cached", "replay"}
        if config.qloo_cache_mode not in valid_modes:
            raise ValueError(
                f"Invalid QLOO_CACHE_MODE: '{config.qloo_cache_mode}'. "
                f"Must be one of {valid_modes} to ensure judge demo stability."
            )
        return config
