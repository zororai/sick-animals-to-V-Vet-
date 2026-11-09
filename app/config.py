"""Application configuration loaded from environment variables.

Secrets (API keys) are only ever read from the environment / a local `.env`
file, never hardcoded. The `Settings` object is immutable so configuration
cannot drift at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number") from exc


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


@dataclass(frozen=True)
class Settings:
    # LLM (any OpenAI-compatible chat completions provider)
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    llm_timeout_seconds: float
    llm_max_attempts: int
    # "json_schema" (strict structured output) or "json_object" (JSON mode)
    llm_response_format: str

    # Speech-to-text (any OpenAI-compatible /audio/transcriptions provider)
    transcription_api_key: str | None
    transcription_base_url: str | None
    transcription_model: str
    transcription_timeout_seconds: float

    # Request limits
    max_message_chars: int
    max_audio_bytes: int

    log_level: str


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process from the environment (and `.env`)."""
    load_dotenv()
    llm_api_key = _optional_env("LLM_API_KEY") or _optional_env("OPENAI_API_KEY")
    response_format = os.getenv("LLM_RESPONSE_FORMAT", "json_schema").strip().lower()
    if response_format not in {"json_schema", "json_object"}:
        raise ValueError("LLM_RESPONSE_FORMAT must be 'json_schema' or 'json_object'")

    return Settings(
        llm_api_key=llm_api_key,
        llm_base_url=_optional_env("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 30.0),
        llm_max_attempts=max(1, _int_env("LLM_MAX_ATTEMPTS", 2)),
        llm_response_format=response_format,
        # Transcription falls back to the LLM credentials when not set separately.
        transcription_api_key=_optional_env("TRANSCRIPTION_API_KEY") or llm_api_key,
        transcription_base_url=_optional_env("TRANSCRIPTION_BASE_URL")
        or _optional_env("LLM_BASE_URL"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
        transcription_timeout_seconds=_float_env("TRANSCRIPTION_TIMEOUT_SECONDS", 60.0),
        max_message_chars=_int_env("MAX_MESSAGE_CHARS", 2000),
        max_audio_bytes=_int_env("MAX_AUDIO_BYTES", 10 * 1024 * 1024),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
