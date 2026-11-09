"""Measure extraction accuracy against a manually annotated dataset.

Usage (needs a real LLM API key in the environment / .env):

    python -m evaluation.evaluate [evaluation/dataset.jsonl]

The scoring functions are pure and are unit-tested offline.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.escalation import evaluate_escalation
from app.schemas import Finding, SignId

Extract = Callable[[str], list[Finding]]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Report:
    overall: Counts = field(default_factory=Counts)
    per_sign: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    matched: int = 0
    value_correct: int = 0
    quote_correct: int = 0
    escalation: Counts = field(default_factory=Counts)
    escalation_true_negatives: int = 0
    failures: int = 0

    @property
    def value_accuracy(self) -> float:
        return self.value_correct / self.matched if self.matched else 0.0

    @property
    def quote_accuracy(self) -> float:
        return self.quote_correct / self.matched if self.matched else 0.0


def quote_matches(predicted: str, gold: str) -> bool:
    """A predicted quote is correct if it contains, or is contained by, the gold evidence."""
    p, g = predicted.lower().strip(), gold.lower().strip()
    return g in p or p in g


def evaluate(examples: list[dict], extract: Extract) -> Report:
    report = Report()
    for example in examples:
        gold = {f["sign_id"]: f for f in example["findings"]}
        try:
            predicted = {f.sign_id.value: f for f in extract(example["message"])}
        except Exception:  # noqa: BLE001 - a failed extraction counts as all misses
            report.failures += 1
            predicted = {}

        for sign in SignId:
            s = sign.value
            in_gold, in_pred = s in gold, s in predicted
            if in_gold and in_pred:
                report.overall.tp += 1
                report.per_sign[s].tp += 1
                report.matched += 1
                report.value_correct += predicted[s].value.value == gold[s]["value"]
                report.quote_correct += quote_matches(predicted[s].quote, gold[s]["quote"])
            elif in_pred:
                report.overall.fp += 1
                report.per_sign[s].fp += 1
            elif in_gold:
                report.overall.fn += 1
                report.per_sign[s].fn += 1

        gold_escalate = evaluate_escalation(
            [Finding.model_validate(f) for f in example["findings"]]
        ).escalate
        pred_escalate = evaluate_escalation(list(predicted.values())).escalate
        if gold_escalate and pred_escalate:
            report.escalation.tp += 1
        elif gold_escalate:
            report.escalation.fn += 1  # a missed urgent case: the most costly error
        elif pred_escalate:
            report.escalation.fp += 1
        else:
            report.escalation_true_negatives += 1
    return report


def format_report(report: Report) -> str:
    o = report.overall
    lines = [
        f"Sign detection   P={o.precision:.2f} R={o.recall:.2f} F1={o.f1:.2f} "
        f"(tp={o.tp} fp={o.fp} fn={o.fn})",
        f"Value accuracy   {report.value_accuracy:.2f} over {report.matched} matched signs",
        f"Quote accuracy   {report.quote_accuracy:.2f}",
        f"Escalation       sensitivity={report.escalation.recall:.2f} "
        f"precision={report.escalation.precision:.2f} "
        f"missed_urgent={report.escalation.fn}",
        f"Failed requests  {report.failures}",
        "",
        "Per sign:",
    ]
    for sign in SignId:
        c = report.per_sign.get(sign.value, Counts())
        lines.append(
            f"  {sign.value:<18} P={c.precision:.2f} R={c.recall:.2f} F1={c.f1:.2f} "
            f"(tp={c.tp} fp={c.fp} fn={c.fn})"
        )
    return "\n".join(lines)


def main() -> None:
    from app.config import get_settings
    from app.extractor import SymptomExtractor
    from app.llm import OpenAICompatibleLLMClient

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation/dataset.jsonl")
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    settings = get_settings()
    extractor = SymptomExtractor(OpenAICompatibleLLMClient(settings), settings.llm_max_attempts)
    print(format_report(evaluate(examples, extractor.extract)))


if __name__ == "__main__":
    main()
