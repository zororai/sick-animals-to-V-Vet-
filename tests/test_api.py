"""End-to-end HTTP tests for POST /extract with a fake LLM."""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeLLM, findings_json

ClientFactory = Callable[..., TestClient]

# The five sample messages from the brief, with the output a correct LLM
# should produce and the response the API must return.
SAMPLES: list[dict[str, Any]] = [
    {
        "message": "My cow has not been eating since yesterday and she is breathing very fast.",
        "llm": findings_json(
            ("reduced_appetite", "yes", "has not been eating since yesterday"),
            ("rapid_breathing", "yes", "she is breathing very fast"),
        ),
        "escalate": False,
        "reason": "No specified escalation criteria were identified.",
    },
    {
        "message": "There's foam coming from her nose and she can't get up.",
        "llm": findings_json(
            ("froth_nose_mouth", "yes", "foam coming from her nose"),
            ("unable_to_stand", "yes", "she can't get up"),
        ),
        "escalate": True,
        "reason": "Escalation triggered by froth_nose_mouth (yes) and unable_to_stand (yes).",
    },
    {
        "message": (
            "I think the eyes look a bit cloudy but I'm not sure. "
            "She has mbandabvu under the ear."
        ),
        "llm": findings_json(
            ("cloudy_eye", "not_sure", "I think the eyes look a bit cloudy but I'm not sure"),
            ("swelling", "yes", "She has mbandabvu under the ear"),
        ),
        "escalate": True,
        "reason": "Escalation triggered by cloudy_eye (not_sure).",
    },
    {
        "message": "The calf is fine, just lots of ticks on the ears.",
        "llm": findings_json(("ticks", "yes", "lots of ticks on the ears")),
        "escalate": False,
        "reason": "No specified escalation criteria were identified.",
    },
    {
        "message": "Mombe yangu haisi kudya, and her urine is dark like tea.",
        "llm": findings_json(
            ("reduced_appetite", "yes", "Mombe yangu haisi kudya"),
            ("red_urine", "yes", "her urine is dark like tea"),
        ),
        "escalate": False,
        "reason": "No specified escalation criteria were identified.",
    },
]


@pytest.mark.parametrize("sample", SAMPLES, ids=[f"sample_{i + 1}" for i in range(len(SAMPLES))])
def test_sample_messages(make_client: ClientFactory, sample: dict[str, Any]) -> None:
    llm = FakeLLM(sample["llm"])
    response = make_client(llm).post("/extract", json={"message": sample["message"]})

    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == sample["llm"]["findings"]
    assert body["escalate"] is sample["escalate"]
    assert body["reason"] == sample["reason"]
    for finding in body["findings"]:
        assert finding["quote"] in sample["message"]
    assert sample["message"] in llm.calls[0]["user"]


def test_spec_example_request(make_client: ClientFactory) -> None:
    message = "My cow has not been eating and has foam coming from her nose."
    llm = FakeLLM(
        findings_json(
            ("reduced_appetite", "yes", "has not been eating"),
            ("froth_nose_mouth", "yes", "foam coming from her nose"),
        )
    )
    body = make_client(llm).post("/extract", json={"message": message}).json()

    assert body == {
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "has not been eating"},
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
        ],
        "escalate": True,
        "reason": "Escalation triggered by froth_nose_mouth (yes).",
    }


def test_no_symptoms(make_client: ClientFactory) -> None:
    body = (
        make_client(FakeLLM({"findings": []}))
        .post("/extract", json={"message": "Hello, how much is the dip?"})
        .json()
    )
    assert body == {
        "findings": [],
        "escalate": False,
        "reason": "No specified escalation criteria were identified.",
    }


def test_negated_escalation_sign_does_not_escalate(make_client: ClientFactory) -> None:
    message = "No foam from the mouth and she is standing fine."
    llm = FakeLLM(
        findings_json(
            ("froth_nose_mouth", "no", "No foam from the mouth"),
            ("unable_to_stand", "no", "she is standing fine"),
        )
    )
    body = make_client(llm).post("/extract", json={"message": message}).json()
    assert body["escalate"] is False
    assert [f["value"] for f in body["findings"]] == ["no", "no"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [{"message": ""}, {"message": "   \n\t"}, {}, {"message": None}, {"message": 123}],
    ids=["empty", "whitespace", "missing", "null", "not_string"],
)
def test_invalid_request_is_rejected(make_client: ClientFactory, payload: dict[str, Any]) -> None:
    llm = FakeLLM({"findings": []})
    response = make_client(llm).post("/extract", json=payload)
    assert response.status_code == 422
    assert llm.calls == []  # never reaches the LLM


def test_oversized_message_is_rejected(make_client: ClientFactory) -> None:
    llm = FakeLLM({"findings": []})
    client = make_client(llm, max_message_chars=100)

    assert client.post("/extract", json={"message": "a" * 100}).status_code == 200
    response = client.post("/extract", json={"message": "a" * 101})
    assert response.status_code == 422
    assert "at most 100 characters" in response.json()["detail"]
    assert len(llm.calls) == 1


def test_message_above_hard_ceiling_is_rejected(make_client: ClientFactory) -> None:
    llm = FakeLLM({"findings": []})
    response = make_client(llm, max_message_chars=50_000).post(
        "/extract", json={"message": "a" * 10_001}
    )
    assert response.status_code == 422
    assert llm.calls == []


def test_malformed_json_body_is_rejected(make_client: ClientFactory) -> None:
    response = make_client(FakeLLM({"findings": []})).post(
        "/extract", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "definitely not json",
        findings_json(("fever", "yes", "stopped eating")),
        findings_json(("reduced_appetite", "yes", "invented quote")),
    ],
    ids=["malformed", "unknown_sign", "fabricated_quote"],
)
def test_invalid_llm_output_returns_502_without_fabricating(
    make_client: ClientFactory, raw: Any
) -> None:
    response = make_client(FakeLLM(raw)).post(
        "/extract", json={"message": "My cow has stopped eating."}
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "Symptom extraction failed. Please try again later."}


def test_missing_api_key_returns_503(make_client: ClientFactory) -> None:
    # No fake LLM override: the real client is built and finds no API key.
    response = make_client(llm=None, llm_api_key=None).post(
        "/extract", json={"message": "My cow has stopped eating."}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Symptom extraction service is not configured."}


def test_unexpected_error_returns_generic_500(make_client: ClientFactory) -> None:
    llm = FakeLLM(RuntimeError("secret internal detail sk-12345"))
    response = make_client(llm).post("/extract", json={"message": "My cow has stopped eating."})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "sk-12345" not in response.text


def test_health(make_client: ClientFactory) -> None:
    assert make_client().get("/health").json() == {"status": "ok"}
