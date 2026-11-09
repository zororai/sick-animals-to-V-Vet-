"""Pydantic models for API requests/responses and for validating LLM output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignId(str, Enum):
    """The only nine veterinary signs this service is allowed to report."""

    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


class FindingValue(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"


class Finding(BaseModel):
    """A single sign reported by the farmer, backed by a verbatim quote."""

    # extra="forbid": unexpected keys from the LLM are treated as invalid output.
    model_config = ConfigDict(extra="forbid")

    sign_id: SignId
    value: FindingValue
    quote: str = Field(min_length=1)

    @field_validator("quote")
    @classmethod
    def quote_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("quote must not be blank")
        return v


class LLMExtractionOutput(BaseModel):
    """The exact JSON shape the LLM is required to return."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]

    @field_validator("findings")
    @classmethod
    def no_duplicate_signs(cls, findings: list[Finding]) -> list[Finding]:
        seen: set[SignId] = set()
        for finding in findings:
            if finding.sign_id in seen:
                raise ValueError(f"duplicate sign_id: {finding.sign_id.value}")
            seen.add(finding.sign_id)
        return findings


class ExtractRequest(BaseModel):
    # Length limit is enforced in the route using configurable settings; this
    # hard ceiling is a defensive backstop.
    message: str = Field(min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str


class AudioExtractResponse(ExtractResponse):
    transcription: str


class ErrorResponse(BaseModel):
    detail: str
