"""LLM provider abstraction.

The extractor depends on the small `LLMClient` protocol rather than on a
specific SDK. This keeps the provider swappable (OpenAI, Azure, OpenRouter,
a local vLLM/Ollama server, ...) and lets tests inject a fake client so no
network call or API key is ever needed.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """The LLM provider failed or returned no usable content.

    Messages are deliberately generic so that provider details (URLs, keys,
    raw responses) never reach API clients.
    """


class LLMNotConfiguredError(LLMError):
    """No API key is configured for the LLM provider."""


class LLMClient(Protocol):
    def complete_json(
        self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> str:
        """Return the model's raw JSON text for the given prompts."""
        ...


class OpenAICompatibleLLMClient:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMNotConfiguredError("LLM API key is not configured")
        # Imported lazily so the rest of the app (and the tests) never need
        # the SDK to construct a client.
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        self._model = settings.llm_model
        self._response_format = settings.llm_response_format

    def complete_json(
        self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> str:
        from openai import OpenAIError

        if self._response_format == "json_schema":
            # Strict structured output: the provider constrains decoding to the schema.
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "symptom_findings",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        else:
            # JSON mode for providers without schema support; still validated downstream.
            response_format = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            # Log the exception type only: SDK messages can echo request details.
            logger.warning("LLM request failed: %s", type(exc).__name__)
            raise LLMError("LLM request failed") from exc

        if not completion.choices:
            raise LLMError("LLM returned no choices")
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError("LLM refused the request")
        if not message.content:
            raise LLMError("LLM returned empty content")
        return message.content
