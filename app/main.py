"""HTTP layer. Routes only validate input, call services and shape responses;
all business logic lives in `extractor.py` and `escalation.py`."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.escalation import evaluate_escalation
from app.extractor import ExtractionError, SymptomExtractor
from app.llm import LLMClient, LLMNotConfiguredError, OpenAICompatibleLLMClient
from app.schemas import (
    AudioExtractResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    Finding,
)
from app.transcription import (
    AudioValidationError,
    EmptyAudioError,
    EmptyTranscriptionError,
    OpenAICompatibleTranscriber,
    TranscriptionError,
    TranscriptionNotConfiguredError,
    Transcriber,
    validate_audio,
)

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


@lru_cache
def _build_transcriber(settings: Settings) -> Transcriber:
    return OpenAICompatibleTranscriber(settings)


def get_transcriber(settings: Settings = Depends(get_settings)) -> Transcriber:
    try:
        return _build_transcriber(settings)
    except TranscriptionNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio transcription service is not configured.",
        ) from None


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
AUDIO_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    **ERROR_RESPONSES,
    413: {"model": ErrorResponse, "description": "Audio file too large"},
    415: {"model": ErrorResponse, "description": "Unsupported audio format"},
}


def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read at most max_bytes + 1 so oversized files are detected without loading them fully."""
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Audio file must be at most {max_bytes} bytes",
        )
    return data


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

    @app.post(
        "/extract-audio",
        response_model=AudioExtractResponse,
        responses=AUDIO_ERROR_RESPONSES,
    )
    def extract_audio(
        file: UploadFile = File(..., description="WhatsApp voice note or other audio file"),
        transcriber: Transcriber = Depends(get_transcriber),
        extractor: SymptomExtractor = Depends(get_extractor),
        settings: Settings = Depends(get_settings),
    ) -> AudioExtractResponse:
        data = read_upload_limited(file, settings.max_audio_bytes)
        try:
            provider_filename = validate_audio(data, file.filename, file.content_type)
        except EmptyAudioError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
        except AudioValidationError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None

        try:
            transcript = transcriber.transcribe(data, provider_filename).strip()
            if not transcript:
                raise EmptyTranscriptionError("No speech was recognised in the audio")
        except EmptyTranscriptionError:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech was recognised in the audio."
            ) from None
        except TranscriptionError:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "Audio transcription failed. Please try again later."
            ) from None

        if len(transcript) > settings.max_message_chars:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Transcribed message must be at most {settings.max_message_chars} characters",
            )

        logger.info("Transcribed %d bytes of audio into %d chars", len(data), len(transcript))
        # Same extraction + escalation path as /extract.
        result = build_response(extractor.extract(transcript))
        return AudioExtractResponse(transcription=transcript, **result.model_dump())

    return app


app = create_app()
