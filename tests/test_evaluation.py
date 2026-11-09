"""Offline tests for the evaluation metrics (no LLM involved)."""

import json
from pathlib import Path

import pytest

from evaluation.evaluate import evaluate, format_report, quote_matches
from app.schemas import Finding

DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.jsonl"


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


def test_dataset_is_valid_and_quotes_are_verbatim() -> None:
    for example in load_dataset():
        for f in example["findings"]:
            Finding.model_validate(f)
            assert f["quote"] in example["message"]


def test_perfect_extractor_scores_one() -> None:
    examples = load_dataset()
    gold = {e["message"]: [Finding.model_validate(f) for f in e["findings"]] for e in examples}
    report = evaluate(examples, lambda m: gold[m])

    assert report.overall.f1 == 1.0
    assert report.value_accuracy == 1.0
    assert report.quote_accuracy == 1.0
    assert report.escalation.recall == 1.0
    assert report.escalation.fn == 0
    assert "Per sign" in format_report(report)


def test_empty_extractor_misses_everything_including_urgent_cases() -> None:
    examples = load_dataset()
    report = evaluate(examples, lambda m: [])

    assert report.overall.tp == 0
    assert report.overall.recall == 0.0
    assert report.escalation.fn > 0  # missed urgent cases are counted


def test_failed_extraction_counts_as_misses() -> None:
    def boom(_: str) -> list[Finding]:
        raise RuntimeError("provider down")

    report = evaluate(load_dataset()[:1], boom)
    assert report.failures == 1
    assert report.overall.fn == 2


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("has not been eating", "has not been eating since yesterday", True),
        ("She has mbandabvu under the ear", "mbandabvu under the ear", True),
        ("breathing", "foam", False),
    ],
)
def test_quote_matches(pred: str, gold: str, ok: bool) -> None:
    assert quote_matches(pred, gold) is ok
