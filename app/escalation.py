"""Deterministic escalation rules.

Escalation is decided by plain Python, never by the LLM. The LLM only
extracts what the farmer said; this module decides what is urgent.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import Finding, FindingValue, SignId

# Signs that require a vet's attention whenever they are present OR possible.
ESCALATION_SIGNS: frozenset[SignId] = frozenset(
    {SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE}
)

# "not_sure" escalates too: for urgent signs, a possible positive must not be missed.
ESCALATION_VALUES: frozenset[FindingValue] = frozenset(
    {FindingValue.YES, FindingValue.NOT_SURE}
)

NO_ESCALATION_REASON = "No specified escalation criteria were identified."


@dataclass(frozen=True)
class EscalationResult:
    escalate: bool
    reason: str
    triggered_by: tuple[Finding, ...] = ()


def evaluate_escalation(findings: list[Finding]) -> EscalationResult:
    """Escalate if any escalation sign is reported as `yes` or `not_sure`.

    A sign the farmer did not mention is simply absent from `findings`, and a
    sign reported as `no` is an explicit negative; neither triggers escalation.
    """
    triggered = tuple(
        f
        for f in findings
        if f.sign_id in ESCALATION_SIGNS and f.value in ESCALATION_VALUES
    )
    if not triggered:
        return EscalationResult(escalate=False, reason=NO_ESCALATION_REASON)

    parts = [f"{f.sign_id.value} ({f.value.value})" for f in triggered]
    return EscalationResult(
        escalate=True,
        reason=f"Escalation triggered by {_join(parts)}.",
        triggered_by=triggered,
    )


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]
