"""Tests for audio validation, transcription and POST /extract-audio (all offline)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.transcription import (
    EmptyAudioError,
    EmptyTranscriptionError,
    OpenAICompatibleTranscriber,
    TranscriptionError,
    TranscriptionNotConfiguredError,
    UnsupportedAudioError,
    detect_audio_format,
    validate_audio,
)
from tests.conftest import FakeLLM, findings_json, make_settings

ClientFactory = Callable[..., TestClient]

# Minimal byte payloads carrying real container signatures.
OGG = b"OggS" + b"\x00" * 60
MP3 = b"ID3" + b"\x00" * 60
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40
M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 40

SHONA_TRANSCRIPT = "Mombe yangu haisi kudya uye ine furo mumhino."


class FakeTranscriber:
    def __init__(self, result: str | Exception) -> None:
        self._result = result
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def upload(name: str = "voice.ogg", data: bytes = OGG, ctype: str = "audio/ogg") -> dict[str, Any]:
    return {"file": (name, data, ctype)}


# ---------------------------------------------------------------------------
# Endpoint: success
# ---------------------------------------------------------------------------


def test_audio_transcription_success_uses_shared_extractor(make_client: ClientFactory) -> None:
    transcriber = FakeTranscriber(SHONA_TRANSCRIPT)
    llm = FakeLLM(
        findings_json(
            ("reduced_appetite", "yes", "haisi kudya"),
            ("froth_nose_mouth", "yes", "ine furo mumhino"),
        )
    )
    response = make_client(llm, transcriber).post("/extract-audio", files=upload())

    assert response.status_code == 200
    assert response.json() == {
        "transcription": SHONA_TRANSCRIPT,
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "haisi kudya"},
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "ine furo mumhino"},
        ],
        "escalate": True,
        "reason": "Escalation triggered by froth_nose_mouth (yes).",
    }
    # The transcript, not the audio, went to the same LLM extraction prompt.
    assert SHONA_TRANSCRIPT in llm.calls[0]["user"]
    assert transcriber.calls == [(OGG, "voice_note.ogg")]


def test_audio_quotes_are_validated_against_transcript(make_client: ClientFactory) -> None:
    transcriber = FakeTranscriber("My cow is fine.")
    llm = FakeLLM(findings_json(("ticks", "yes", "lots of ticks")))  # not in transcript
    response = make_client(llm, transcriber).post("/extract-audio", files=upload())
    assert response.status_code == 502


@pytest.mark.parametrize(
    ("name", "data", "ctype", "sent_as"),
    [
        ("PTT-20260101-WA0001.opus", OGG, "audio/ogg; codecs=opus", "voice_note.ogg"),
        ("note.mp3", MP3, "audio/mpeg", "voice_note.mp3"),
        ("note.wav", WAV, "audio/wav", "voice_note.wav"),
        ("note.m4a", M4A, "audio/mp4", "voice_note.m4a"),
        ("blob", OGG, "application/octet-stream", "voice_note.ogg"),
    ],
)
def test_common_whatsapp_formats_are_accepted(
    make_client: ClientFactory, name: str, data: bytes, ctype: str, sent_as: str
) -> None:
    transcriber = FakeTranscriber("Hello")
    response = make_client(FakeLLM({"findings": []}), transcriber).post(
        "/extract-audio", files=upload(name, data, ctype)
    )
    assert response.status_code == 200
    assert response.json()["findings"] == []
    assert response.json()["escalate"] is False
    assert transcriber.calls[0][1] == sent_as


# ---------------------------------------------------------------------------
# Endpoint: failures
# ---------------------------------------------------------------------------


def test_audio_transcription_failure_returns_502(make_client: ClientFactory) -> None:
    llm = FakeLLM({"findings": []})
    transcriber = FakeTranscriber(TranscriptionError("provider down, key sk-secret"))
    response = make_client(llm, transcriber).post("/extract-audio", files=upload())

    assert response.status_code == 502
    assert response.json() == {"detail": "Audio transcription failed. Please try again later."}
    assert "sk-secret" not in response.text
    assert llm.calls == []  # never extract from a failed transcription


@pytest.mark.parametrize("result", ["", "   ", EmptyTranscriptionError("silence")])
def test_empty_transcription_returns_422(make_client: ClientFactory, result: Any) -> None:
    llm = FakeLLM({"findings": []})
    response = make_client(llm, FakeTranscriber(result)).post("/extract-audio", files=upload())
    assert response.status_code == 422
    assert llm.calls == []


def test_oversized_audio_returns_413(make_client: ClientFactory) -> None:
    transcriber = FakeTranscriber("x")
    client = make_client(FakeLLM({"findings": []}), transcriber, max_audio_bytes=100)
    response = client.post("/extract-audio", files=upload(data=b"OggS" + b"\x00" * 200))
    assert response.status_code == 413
    assert transcriber.calls == []


def test_empty_audio_returns_422(make_client: ClientFactory) -> None:
    transcriber = FakeTranscriber("x")
    response = make_client(FakeLLM({"findings": []}), transcriber).post(
        "/extract-audio", files=upload(data=b"")
    )
    assert response.status_code == 422
    assert transcriber.calls == []


@pytest.mark.parametrize(
    ("name", "data", "ctype"),
    [
        ("notes.txt", OGG, "audio/ogg"),  # bad extension
        ("voice.ogg", OGG, "text/plain"),  # bad content type
        ("voice.ogg", b"<html>not audio</html>", "audio/ogg"),  # bad signature
        ("payload.exe", b"MZ\x90\x00", "application/octet-stream"),
    ],
    ids=["extension", "content_type", "signature", "executable"],
)
def test_unsupported_audio_returns_415(
    make_client: ClientFactory, name: str, data: bytes, ctype: str
) -> None:
    transcriber = FakeTranscriber("x")
    response = make_client(FakeLLM({"findings": []}), transcriber).post(
        "/extract-audio", files=upload(name, data, ctype)
    )
    assert response.status_code == 415
    assert transcriber.calls == []


def test_missing_file_returns_422(make_client: ClientFactory) -> None:
    response = make_client(FakeLLM({"findings": []}), FakeTranscriber("x")).post("/extract-audio")
    assert response.status_code == 422


def test_overlong_transcript_returns_422(make_client: ClientFactory) -> None:
    llm = FakeLLM({"findings": []})
    client = make_client(llm, FakeTranscriber("a" * 51), max_message_chars=50)
    assert client.post("/extract-audio", files=upload()).status_code == 422
    assert llm.calls == []


def test_unconfigured_transcription_returns_503(make_client: ClientFactory) -> None:
    response = make_client(FakeLLM({"findings": []}), transcription_api_key=None).post(
        "/extract-audio", files=upload()
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Audio transcription service is not configured."}


def test_text_endpoint_works_without_transcription_configured(make_client: ClientFactory) -> None:
    """Audio is optional: /extract must not depend on transcription settings."""
    client = make_client(FakeLLM({"findings": []}), transcription_api_key=None)
    assert client.post("/extract", json={"message": "Hello"}).status_code == 200


# ---------------------------------------------------------------------------
# Unit: validation helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (OGG, ".ogg"),
        (MP3, ".mp3"),
        (b"\xff\xfb\x90\x00", ".mp3"),
        (WAV, ".wav"),
        (M4A, ".m4a"),
        (b"\x1a\x45\xdf\xa3\x00", ".webm"),
        (b"fLaC\x00", ".flac"),
        (b"hello", None),
        (b"", None),
    ],
)
def test_detect_audio_format(data: bytes, expected: str | None) -> None:
    assert detect_audio_format(data) == expected


def test_validate_audio_errors() -> None:
    with pytest.raises(EmptyAudioError):
        validate_audio(b"", "a.ogg", "audio/ogg")
    with pytest.raises(UnsupportedAudioError):
        validate_audio(OGG, "a.amr", "audio/ogg")
    assert validate_audio(OGG, None, None) == "voice_note.ogg"


# ---------------------------------------------------------------------------
# Unit: OpenAI-compatible transcriber with a stubbed SDK (no network)
# ---------------------------------------------------------------------------


def _transcriber_with_stub(create: Any) -> OpenAICompatibleTranscriber:
    t = OpenAICompatibleTranscriber(make_settings(transcription_api_key="sk-test"))
    t._client = SimpleNamespace(  # type: ignore[assignment]
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    return t


def test_transcriber_requires_api_key() -> None:
    with pytest.raises(TranscriptionNotConfiguredError):
        OpenAICompatibleTranscriber(make_settings(transcription_api_key=None))


def test_transcriber_success() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(text="  Mombe yangu haisi kudya.  ")

    assert _transcriber_with_stub(create).transcribe(OGG, "voice_note.ogg") == (
        "Mombe yangu haisi kudya."
    )
    assert captured["file"] == ("voice_note.ogg", OGG)
    assert captured["model"] == "test-whisper"


def test_transcriber_maps_sdk_errors() -> None:
    from openai import APIConnectionError

    def create(**_: Any) -> Any:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    with pytest.raises(TranscriptionError, match="Transcription request failed"):
        _transcriber_with_stub(create).transcribe(OGG, "voice_note.ogg")


def test_transcriber_empty_text_raises() -> None:
    with pytest.raises(EmptyTranscriptionError):
        _transcriber_with_stub(lambda **_: SimpleNamespace(text=" ")).transcribe(OGG, "x.ogg")
