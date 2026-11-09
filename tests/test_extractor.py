"""Extraction pipeline tests.

The LLM is faked, so these tests verify everything *around* the model:
prompting, parsing, schema validation, quote verification and retries.
Model accuracy itself is measured separately (see README "Accuracy evaluation").
"""

from types import SimpleNamespace
from typing import Any

import pytest

from app.extractor import (
    FINDINGS_JSON_SCHEMA,
    SYSTEM_PROMPT,
    ExtractionError,
    InvalidLLMOutputError,
    SymptomExtractor,
    locate_quote,
    parse_llm_output,
)
from app.llm import LLMError, LLMNotConfiguredError, OpenAICompatibleLLMClient
from app.schemas import FindingValue, SignId
from tests.conftest import FakeLLM, findings_json, make_settings


def extract(message: str, *responses: Any) -> list[tuple[str, str, str]]:
    extractor = SymptomExtractor(FakeLLM(*responses))
    return [(f.sign_id.value, f.value.value, f.quote) for f in extractor.extract(message)]


# ---------------------------------------------------------------------------
# One scenario per sign (1-9)
# ---------------------------------------------------------------------------

SIGN_CASES = [
    ("reduced_appetite", "My cow has stopped eating.", "has stopped eating"),
    ("weakness", "The goat is very weak and lying down all day.", "very weak and lying down all day"),
    ("swelling", "She has mbandabvu under the jaw.", "mbandabvu under the jaw"),
    ("rapid_breathing", "He is breathing very fast even when resting.", "breathing very fast even when resting"),
    ("cloudy_eye", "Her left eye has gone milky white.", "eye has gone milky white"),
    ("froth_nose_mouth", "There's foam coming from the nose.", "foam coming from the nose"),
    ("red_urine", "The urine is the colour of strong tea.", "urine is the colour of strong tea"),
    ("unable_to_stand", "The bull cannot get up since this morning.", "cannot get up"),
    ("ticks", "I can see ticks all over her belly.", "ticks all over her belly"),
]


@pytest.mark.parametrize(("sign", "message", "quote"), SIGN_CASES)
def test_each_sign_is_extracted(sign: str, message: str, quote: str) -> None:
    assert extract(message, findings_json((sign, "yes", quote))) == [(sign, "yes", quote)]


def test_every_sign_has_a_case() -> None:
    assert {case[0] for case in SIGN_CASES} == {s.value for s in SignId}


# ---------------------------------------------------------------------------
# Negation, uncertainty, language, multiplicity
# ---------------------------------------------------------------------------


def test_explicitly_absent_symptom_is_no_and_quote_keeps_negation() -> None:
    message = "My cow is eating normally and there are no ticks."
    result = extract(
        message,
        findings_json(
            ("reduced_appetite", "no", "eating normally"),
            ("ticks", "no", "no ticks"),
        ),
    )
    assert result == [
        ("reduced_appetite", "no", "eating normally"),
        ("ticks", "no", "no ticks"),
    ]


def test_uncertain_symptom_is_not_sure_and_quote_keeps_uncertainty() -> None:
    message = "I think the eye looks cloudy, but I'm not sure."
    quote = "I think the eye looks cloudy, but I'm not sure"
    assert extract(message, findings_json(("cloudy_eye", "not_sure", quote))) == [
        ("cloudy_eye", "not_sure", quote)
    ]


def test_pure_shona_message() -> None:
    message = "Mombe yangu haisi kudya."
    assert extract(message, findings_json(("reduced_appetite", "yes", "haisi kudya"))) == [
        ("reduced_appetite", "yes", "haisi kudya")
    ]


def test_mixed_shona_english_message() -> None:
    message = "Mombe yangu haisi kudya, and her urine is dark like tea."
    result = extract(
        message,
        findings_json(
            ("reduced_appetite", "yes", "Mombe yangu haisi kudya"),
            ("red_urine", "yes", "her urine is dark like tea"),
        ),
    )
    assert [r[0] for r in result] == ["reduced_appetite", "red_urine"]


def test_multiple_symptoms() -> None:
    message = "There's foam coming from her nose and she can't get up. Lots of ticks too."
    result = extract(
        message,
        findings_json(
            ("froth_nose_mouth", "yes", "foam coming from her nose"),
            ("unable_to_stand", "yes", "she can't get up"),
            ("ticks", "yes", "Lots of ticks"),
        ),
    )
    assert len(result) == 3


def test_message_with_no_symptoms_returns_empty_list() -> None:
    assert extract("Good morning, when is the vet visiting?", {"findings": []}) == []


# ---------------------------------------------------------------------------
# Untrusted output handling (each must be rejected, not silently accepted)
# ---------------------------------------------------------------------------

MESSAGE = "My cow has stopped eating."

INVALID_OUTPUTS = {
    "unknown_sign_id": findings_json(("fever", "yes", "has stopped eating")),
    "invalid_value": findings_json(("reduced_appetite", "maybe", "has stopped eating")),
    "boolean_value": {"findings": [{"sign_id": "reduced_appetite", "value": True, "quote": "x"}]},
    "missing_quote": {"findings": [{"sign_id": "reduced_appetite", "value": "yes"}]},
    "empty_quote": findings_json(("reduced_appetite", "yes", "")),
    "blank_quote": findings_json(("reduced_appetite", "yes", "   ")),
    "fabricated_quote": findings_json(("reduced_appetite", "yes", "refuses all food")),
    "translated_quote": findings_json(("reduced_appetite", "yes", "haisi kudya")),
    "duplicate_sign": findings_json(
        ("reduced_appetite", "yes", "has stopped eating"),
        ("reduced_appetite", "yes", "stopped eating"),
    ),
    "malformed_json": '{"findings": [{"sign_id": "reduced_appetite",',
    "not_json_at_all": "The cow has reduced appetite.",
    "top_level_list": [{"sign_id": "reduced_appetite", "value": "yes", "quote": "stopped eating"}],
    "missing_findings_key": {"results": []},
    "findings_not_a_list": {"findings": "reduced_appetite"},
    "extra_top_level_key": {"findings": [], "escalate": True},
    "extra_finding_key": {
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "stopped eating", "severity": "high"}
        ]
    },
    "null": "null",
}


@pytest.mark.parametrize("raw", INVALID_OUTPUTS.values(), ids=INVALID_OUTPUTS.keys())
def test_invalid_llm_output_is_rejected(raw: Any) -> None:
    import json

    text = raw if isinstance(raw, str) else json.dumps(raw)
    with pytest.raises(InvalidLLMOutputError):
        parse_llm_output(text, MESSAGE)


@pytest.mark.parametrize("raw", INVALID_OUTPUTS.values(), ids=INVALID_OUTPUTS.keys())
def test_extractor_raises_after_retries_on_persistently_invalid_output(raw: Any) -> None:
    llm = FakeLLM(raw)
    with pytest.raises(ExtractionError):
        SymptomExtractor(llm, max_attempts=2).extract(MESSAGE)
    assert len(llm.calls) == 2


def test_one_bad_finding_rejects_the_whole_response() -> None:
    """Partially valid output must not be silently trimmed."""
    raw = findings_json(
        ("reduced_appetite", "yes", "has stopped eating"),
        ("ticks", "yes", "ticks everywhere"),  # fabricated
    )
    with pytest.raises(ExtractionError):
        SymptomExtractor(FakeLLM(raw), max_attempts=1).extract(MESSAGE)


def test_retry_recovers_from_a_single_malformed_response() -> None:
    llm = FakeLLM("not json", findings_json(("reduced_appetite", "yes", "has stopped eating")))
    findings = SymptomExtractor(llm, max_attempts=2).extract(MESSAGE)
    assert findings[0].sign_id is SignId.REDUCED_APPETITE
    assert len(llm.calls) == 2


def test_provider_error_is_retried_then_raised() -> None:
    llm = FakeLLM(LLMError("timeout"))
    with pytest.raises(ExtractionError):
        SymptomExtractor(llm, max_attempts=3).extract(MESSAGE)
    assert len(llm.calls) == 3


def test_not_configured_error_is_not_retried() -> None:
    llm = FakeLLM(LLMNotConfiguredError("no key"))
    with pytest.raises(LLMNotConfiguredError):
        SymptomExtractor(llm, max_attempts=3).extract(MESSAGE)
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("has stopped eating", "has stopped eating"),
        ("HAS STOPPED EATING", "has stopped eating"),  # case-insensitive, original casing returned
        ("has   stopped\neating", "has stopped eating"),  # whitespace-insensitive
        ('"has stopped eating"', "has stopped eating"),  # wrapping quotation marks
        ("refuses food", None),
        ("", None),
    ],
)
def test_locate_quote(quote: str, expected: str | None) -> None:
    assert locate_quote(quote, "My cow has stopped eating.") == expected


def test_locate_quote_handles_curly_apostrophes() -> None:
    message = "There’s foam and she can’t get up"
    assert locate_quote("she can't get up", message) == "she can’t get up"


def test_returned_quote_is_sliced_from_original_message() -> None:
    message = "Mombe Yangu HAISI kudya"
    findings = parse_llm_output(
        '{"findings":[{"sign_id":"reduced_appetite","value":"yes","quote":"haisi kudya"}]}',
        message,
    )
    assert findings[0].quote == "HAISI kudya"
    assert findings[0].quote in message


def test_quote_regex_special_characters_are_escaped() -> None:
    message = "Is it (maybe) cloudy? Not sure."
    assert locate_quote("(maybe) cloudy? Not sure", message) == "(maybe) cloudy? Not sure"
    assert locate_quote("(maybe.*sure", message) is None


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------


def test_prompt_contains_message_signs_and_schema() -> None:
    llm = FakeLLM({"findings": []})
    SymptomExtractor(llm).extract("Ine mbandabvu")
    call = llm.calls[0]

    assert "Ine mbandabvu" in call["user"]
    for sign in SignId:
        assert sign.value in call["system"]
    assert "mbandabvu" in call["system"]
    assert call["schema"] is FINDINGS_JSON_SCHEMA
    item = FINDINGS_JSON_SCHEMA["properties"]["findings"]["items"]
    assert item["properties"]["sign_id"]["enum"] == [s.value for s in SignId]
    assert item["properties"]["value"]["enum"] == [v.value for v in FindingValue]


def test_system_prompt_forbids_escalation_and_translation() -> None:
    assert "escalat" not in SYSTEM_PROMPT.lower()  # LLM is never asked about escalation
    assert "Do NOT translate" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# OpenAI-compatible client (SDK replaced with a stub; no network)
# ---------------------------------------------------------------------------


def _client_with_stub(response_format: str, create: Any) -> OpenAICompatibleLLMClient:
    client = OpenAICompatibleLLMClient(
        make_settings(llm_api_key="sk-test", llm_response_format=response_format)
    )
    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


def _completion(content: str | None, refusal: str | None = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal))]
    )


def test_openai_client_requires_api_key() -> None:
    with pytest.raises(LLMNotConfiguredError):
        OpenAICompatibleLLMClient(make_settings(llm_api_key=None))


@pytest.mark.parametrize("fmt", ["json_schema", "json_object"])
def test_openai_client_sends_structured_output_request(fmt: str) -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion('{"findings": []}')

    client = _client_with_stub(fmt, create)
    assert client.complete_json("sys", "user", FINDINGS_JSON_SCHEMA) == '{"findings": []}'
    assert captured["temperature"] == 0
    assert captured["response_format"]["type"] == fmt
    if fmt == "json_schema":
        assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_client_maps_sdk_errors_to_llm_error() -> None:
    from openai import APIConnectionError

    def create(**_: Any) -> Any:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    with pytest.raises(LLMError) as exc_info:
        _client_with_stub("json_schema", create).complete_json("s", "u", {})
    assert str(exc_info.value) == "LLM request failed"


@pytest.mark.parametrize(
    "completion",
    [_completion(None), _completion("x", refusal="no"), SimpleNamespace(choices=[])],
    ids=["empty", "refusal", "no_choices"],
)
def test_openai_client_rejects_unusable_completions(completion: Any) -> None:
    with pytest.raises(LLMError):
        _client_with_stub("json_schema", lambda **_: completion).complete_json("s", "u", {})
