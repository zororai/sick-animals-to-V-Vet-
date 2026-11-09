"""HTTP layer. Routes only validate input, call services and shape responses;
all business logic lives in `extractor.py` and `escalation.py`."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.escalation import evaluate_escalation
from app.extractor import ExtractionError, SymptomExtractor
from app.llm import LLMClient, LLMNotConfiguredError, OpenAICompatibleLLMClient
from app.schemas import ErrorResponse, ExtractRequest, ExtractResponse, Finding

logger = logging.getLogger("app")


# ---------------------------------------------------------------------------
# Dependencies (overridden in tests via app.dependency_overrides)
# ---------------------------------------------------------------------------


@lru_cache
def _build_llm_client(settings: Settings) -> LLMClient:
    return OpenAICompatibleLLMClient(settings)


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return _build_llm_client(settings)


def get_extractor(
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> SymptomExtractor:
    return SymptomExtractor(llm, max_attempts=settings.llm_max_attempts)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_response(findings: list[Finding]) -> ExtractResponse:
    escalation = evaluate_escalation(findings)
    return ExtractResponse(
        findings=findings, escalate=escalation.escalate, reason=escalation.reason
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    502: {"model": ErrorResponse, "description": "Upstream AI provider failed or returned invalid output"},
    503: {"model": ErrorResponse, "description": "AI provider is not configured"},
}


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(
        title="V-Vet Symptom Extraction Service",
        version="1.0.0",
        description=(
            "Extracts structured veterinary signs from farmers' messages. "
            "This is an information extraction tool, not a diagnosis system."
        ),
    )

    @app.exception_handler(ExtractionError)
    async def extraction_error_handler(_: Request, __: ExtractionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "Symptom extraction failed. Please try again later."},
        )

    @app.exception_handler(LLMNotConfiguredError)
    async def llm_not_configured_handler(_: Request, __: LLMNotConfiguredError) -> JSONResponse:
        logger.error("LLM provider is not configured (missing API key)")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Symptom extraction service is not configured."},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse, responses=ERROR_RESPONSES)
    def extract(
        request: ExtractRequest,
        extractor: SymptomExtractor = Depends(get_extractor),
        settings: Settings = Depends(get_settings),
    ) -> ExtractResponse:
        if len(request.message) > settings.max_message_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"message must be at most {settings.max_message_chars} characters",
            )
        findings = extractor.extract(request.message)
        return build_response(findings)

    return app


app = create_app()
