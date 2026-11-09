import pytest

from app.escalation import ESCALATION_SIGNS, NO_ESCALATION_REASON, evaluate_escalation
from app.schemas import Finding, FindingValue, SignId


def finding(sign: SignId, value: FindingValue, quote: str = "some quote") -> Finding:
    return Finding(sign_id=sign, value=value, quote=quote)


def test_escalation_signs_are_exactly_the_specified_three() -> None:
    assert ESCALATION_SIGNS == {
        SignId.FROTH_NOSE_MOUTH,
        SignId.UNABLE_TO_STAND,
        SignId.CLOUDY_EYE,
    }


@pytest.mark.parametrize(
    "sign", [SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE]
)
@pytest.mark.parametrize("value", [FindingValue.YES, FindingValue.NOT_SURE])
def test_escalation_sign_with_yes_or_not_sure_escalates(
    sign: SignId, value: FindingValue
) -> None:
    result = evaluate_escalation([finding(sign, value)])

    assert result.escalate is True
    assert result.reason == f"Escalation triggered by {sign.value} ({value.value})."
    assert [f.sign_id for f in result.triggered_by] == [sign]


@pytest.mark.parametrize(
    "sign", [SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE]
)
def test_escalation_sign_explicitly_absent_does_not_escalate(sign: SignId) -> None:
    result = evaluate_escalation([finding(sign, FindingValue.NO)])

    assert result.escalate is False
    assert result.reason == NO_ESCALATION_REASON


@pytest.mark.parametrize("sign", [s for s in SignId if s not in ESCALATION_SIGNS])
@pytest.mark.parametrize("value", list(FindingValue))
def test_non_escalation_signs_never_escalate(sign: SignId, value: FindingValue) -> None:
    result = evaluate_escalation([finding(sign, value)])

    assert result.escalate is False
    assert result.reason == NO_ESCALATION_REASON


def test_no_findings_does_not_escalate() -> None:
    result = evaluate_escalation([])

    assert result.escalate is False
    assert result.reason == "No specified escalation criteria were identified."


def test_spec_example_cloudy_eye_not_sure_escalates() -> None:
    result = evaluate_escalation(
        [finding(SignId.CLOUDY_EYE, FindingValue.NOT_SURE, "I'm not sure about the cloudy eye")]
    )
    assert result.escalate is True
    assert "cloudy_eye" in result.reason


def test_multiple_triggers_are_all_named_and_non_triggers_are_not() -> None:
    result = evaluate_escalation(
        [
            finding(SignId.REDUCED_APPETITE, FindingValue.YES),
            finding(SignId.FROTH_NOSE_MOUTH, FindingValue.YES),
            finding(SignId.CLOUDY_EYE, FindingValue.NO),
            finding(SignId.UNABLE_TO_STAND, FindingValue.NOT_SURE),
        ]
    )

    assert result.escalate is True
    assert result.reason == (
        "Escalation triggered by froth_nose_mouth (yes) and unable_to_stand (not_sure)."
    )
    assert "cloudy_eye" not in result.reason
    assert "reduced_appetite" not in result.reason


def test_three_triggers_reason_lists_all() -> None:
    result = evaluate_escalation(
        [
            finding(SignId.CLOUDY_EYE, FindingValue.NOT_SURE),
            finding(SignId.FROTH_NOSE_MOUTH, FindingValue.YES),
            finding(SignId.UNABLE_TO_STAND, FindingValue.YES),
        ]
    )
    assert result.reason == (
        "Escalation triggered by cloudy_eye (not_sure), froth_nose_mouth (yes) "
        "and unable_to_stand (yes)."
    )
