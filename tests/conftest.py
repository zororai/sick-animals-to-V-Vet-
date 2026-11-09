"""Shared fixtures. Every test runs offline: the LLM and transcriber are fakes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.config import get_settings as real_get_settings
from app.main import app, get_llm_client, get_transcriber


def make_settings(**overrides: Any) -> Settings:
    base = Settings(
        llm_api_key=None,
        llm_base_url=None,
        llm_model="test-model",
        llm_timeout_seconds=5.0,
        llm_max_attempts=2,
        llm_response_format="json_schema",
        transcription_api_key=None,
        transcription_base_url=None,
        transcription_model="test-whisper",
        transcription_timeout_seconds=5.0,
        max_message_chars=2000,
        max_audio_bytes=1024 * 1024,
        log_level="WARNING",
    )
    return replace(base, **overrides)


class FakeLLM:
    """Returns scripted responses in order (repeating the last one) and records calls.

    A response may be a string (returned as-is), a dict/list (JSON-encoded) or
    an Exception instance (raised).
    """

    def __init__(self, *responses: Any) -> None:
        if not responses:
            raise ValueError("FakeLLM needs at least one response")
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> str:
        self.calls.append(
            {"system": system_prompt, "user": user_prompt, "schema": json_schema}
        )
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return response
        return json.dumps(response)


def findings_json(*findings: tuple[str, str, str]) -> dict[str, Any]:
    return {"findings": [{"sign_id": s, "value": v, "quote": q} for s, v, q in findings]}


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def make_client(settings: Settings) -> Iterator[Callable[..., TestClient]]:
    """Factory: build a TestClient wired to the given fake LLM (and optional overrides)."""

    def factory(
        llm: Any | None = None, transcriber: Any | None = None, **setting_overrides: Any
    ) -> TestClient:
        effective = replace(settings, **setting_overrides)
        app.dependency_overrides[real_get_settings] = lambda: effective
        if llm is not None:
            app.dependency_overrides[get_llm_client] = lambda: llm
        if transcriber is not None:
            app.dependency_overrides[get_transcriber] = lambda: transcriber
        return TestClient(app, raise_server_exceptions=False)

    yield factory
    app.dependency_overrides.clear()
