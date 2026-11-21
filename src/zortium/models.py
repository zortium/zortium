from __future__ import annotations

from dataclasses import dataclass, field

from zortium.constants import SuitePriority, SuiteSeverity


@dataclass
class TestCase:
    id: str
    prompt: str
    # Image is optional so that text-channel LLM attacks (encoding tricks, GCG
    # suffixes, refusal-suppression, etc.) can run through the same runner
    # without needing a placeholder pixel. Pure-VLM suites still set bytes.
    image: bytes | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    test_case_id: str
    passed: bool  # True = model resisted attack, False = attack succeeded
    model_response: str
    metadata: dict = field(default_factory=dict)  # optional structured info for reporting UIs


@dataclass
class SuiteResult:
    suite_id: str
    suite_name: str
    priority: SuitePriority = SuitePriority.MEDIUM
    severity: SuiteSeverity = SuiteSeverity.MID
    results: list[EvaluationResult] = field(default_factory=list)

    def attack_success_rate(self) -> float | None:
        if self.priority == SuitePriority.DIAGNOSTIC:
            return None
        evaluable = [r for r in self.results if r.metadata.get("role") != "skipped"]
        if not evaluable:
            return None
        return sum(1 for r in evaluable if not r.passed) / len(evaluable)
