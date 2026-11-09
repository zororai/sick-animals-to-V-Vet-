"""Speech-to-text for WhatsApp voice notes.

Audio is validated here (declared type, extension and actual file signature)
and transcribed through a `Transcriber` protocol, so the provider can be
swapped or faked in tests. The transcript is then handed to the *same*
`SymptomExtractor` used by /extract; there is no separate audio extraction.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """The uploaded file is not acceptable audio."""


class UnsupportedAudioError(AudioValidationError):
    pass


class EmptyAudioError(AudioValidationError):
    pass


class TranscriptionError(Exception):
    """The transcription provider failed."""


class EmptyTranscriptionError(TranscriptionError):
    """The provider succeeded but no speech was recognised."""


class TranscriptionNotConfiguredError(TranscriptionError):
    pass


# Formats accepted by Whisper-compatible APIs. WhatsApp voice notes are
# Ogg/Opus (".ogg" or ".opus"); forwarded audio is often MP3/M4A.
ALLOWED_EXTENSIONS = frozenset(
    {".ogg", ".oga", ".opus", ".mp3", ".mpeg", ".mpga", ".m4a", ".mp4", ".wav", ".webm", ".flac"}
)
ALLOWED_CONTENT_TYPES = frozenset(
    {
        "audio/ogg", "audio/opus", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a",
        "audio/x-m4a", "audio/aac", "audio/wav", "audio/x-wav", "audio/wave", "audio/webm",
        "audio/flac", "audio/x-flac", "video/mp4", "video/webm",
        # Many clients send a generic type; the file signature check still applies.
        "application/octet-stream",
    }
)

# Vocabulary hint that nudges Whisper towards correct spellings of common terms.
TRANSCRIPTION_PROMPT = (
    "A Zimbabwean farmer describing a sick animal in Shona and English. "
    "Mombe, mbudzi, mbandabvu, haisi kudya, furo, mhino, zvikwekwe, weti."
)


def detect_audio_format(data: bytes) -> str | None:
    """Identify the container from its magic bytes. Returns a file extension or None."""
    if data.startswith(b"OggS"):
        return ".ogg"
    if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return ".mp3"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[4:8] == b"ftyp":
        return ".m4a"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    if data.startswith(b"fLaC"):
        return ".flac"
    return None


def validate_audio(data: bytes, filename: str | None, content_type: str | None) -> str:
    """Validate an upload and return the canonical filename to send to the provider."""
    if not data:
        raise EmptyAudioError("Uploaded audio file is empty")

    if filename and "." in filename:
        extension = "." + filename.rsplit(".", 1)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise UnsupportedAudioError(f"Unsupported audio file extension: {extension}")

    if content_type:
        base_type = content_type.split(";", 1)[0].strip().lower()
        if base_type not in ALLOWED_CONTENT_TYPES:
            raise UnsupportedAudioError(f"Unsupported content type: {base_type}")

    detected = detect_audio_format(data)
    if detected is None:
        raise UnsupportedAudioError("File content is not a recognised audio format")

    # A neutral name with the *detected* extension: avoids forwarding user-supplied
    # filenames and lets providers that reject ".opus" accept WhatsApp voice notes.
    return f"voice_note{detected}"


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, filename: str) -> str:
        ...


class OpenAICompatibleTranscriber:
    """Whisper (or compatible) transcription via the OpenAI SDK."""

    def __init__(self, settings: Settings) -> None:
        if not settings.transcription_api_key:
            raise TranscriptionNotConfiguredError("Transcription API key is not configured")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.transcription_api_key,
            base_url=settings.transcription_base_url,
            timeout=settings.transcription_timeout_seconds,
            max_retries=1,
        )
        self._model = settings.transcription_model

    def transcribe(self, audio: bytes, filename: str) -> str:
        from openai import OpenAIError

        try:
            result = self._client.audio.transcriptions.create(
                model=self._model,
                file=(filename, audio),
                prompt=TRANSCRIPTION_PROMPT,
            )
        except OpenAIError as exc:
            logger.warning("Transcription request failed: %s", type(exc).__name__)
            raise TranscriptionError("Transcription request failed") from exc

        text = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(text, str):
            raise TranscriptionError("Transcription provider returned an unexpected response")
        if not text.strip():
            raise EmptyTranscriptionError("No speech was recognised in the audio")
        return text.strip()
