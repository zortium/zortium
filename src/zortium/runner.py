from __future__ import annotations

from enum import Enum
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from zortium.logging import get_logger
from zortium.attacks.base import AttackSuite
from zortium.evaluators.judge import LLMJudge
from zortium.constants import EvaluatorType, SuitePriority
from zortium.models import EvaluationResult, SuiteResult, TestCase
from zortium.providers.base import ProviderFatalError, ProviderSkippedError, VLMProvider

logger = get_logger("runner")

DEFAULT_STREAM_WORKERS = 4  # suite-level parallelism for the web background thread


class JudgeSkipStatus(str, Enum):
    NO_JUDGE_CONFIGURED = "skipped_no_judge_configured"
    NO_HARM_REQUEST_IN_METADATA = "skipped_no_harm_request_in_metadata"


class TestRunner:
    """
    Orchestrates running one or more attack suites against a target VLM/LLM.

    Critical suites: LLM judge is mandatory. If no judge_provider is supplied,
    all test cases in critical suites are skipped (not keyword-evaluated).

    Non-critical suites: LLM judge used when evaluator_type == "llm_judge" in
    metadata AND a judge_provider is available. Falls back to suite's keyword
    evaluation otherwise.

    Diagnostic suites: run normally but attack_success_rate() returns None —
    excluded from headline ASR.
    """

    def __init__(
        self,
        provider: VLMProvider,
        *,
        judge_provider: VLMProvider | None = None,
        judge_is_self: bool = False,
        attacker_provider: VLMProvider | None = None,
    ) -> None:
        self._provider = provider
        self._judge = LLMJudge(judge_provider) if judge_provider else None
        self._judge_is_self = judge_is_self
        self._attacker_provider = attacker_provider

    @staticmethod
    def _skipped_result(test_case: TestCase, reason: str) -> EvaluationResult:
        return EvaluationResult(
            test_case_id=test_case.id,
            passed=True,
            model_response="",
            metadata={**test_case.metadata, "role": "skipped", "skip_reason": reason},
        )

    @staticmethod
    def _skip_reason_from_error(exc: ProviderSkippedError, schema: dict | None) -> str:
        raw = str(exc).strip().lower()
        if "rate" in raw:
            return "rate_limited"
        if "connection" in raw:
            return "connection_error"
        if schema is not None:
            return "structured_output_unsupported"
        return raw.replace(" ", "_") or "provider_skipped"

    def _maybe_judge(
        self,
        test_case: TestCase,
        base: EvaluationResult,
        *,
        force_judge: bool = False,
    ) -> EvaluationResult:
        if test_case.metadata.get("control_case"):
            return base

        # Cases scored by an exact-match evaluator (Image Hijacks' forced URL and
        # EmailAPI exfil) carry no harm_request_text because they come from a
        # behaviour spec, not a harm category. A judge adds nothing over an exact
        # match, and without this they hit the critical-suite harm_request guard
        # below and get dropped as unavailable.
        if test_case.metadata.get("deterministic_eval"):
            return base

        evaluator_type = test_case.metadata.get("evaluator_type", EvaluatorType.KEYWORD)
        if evaluator_type != EvaluatorType.LLM_JUDGE and not force_judge:
            return base
        if self._judge is None:
            base.metadata["judge_status"] = JudgeSkipStatus.NO_JUDGE_CONFIGURED
            return base

        harm_request = test_case.metadata.get("harm_request_text", "")
        if not harm_request:
            if force_judge:
                return self._skipped_result(test_case, "no_harm_request_for_critical_suite")
            base.metadata["judge_status"] = JudgeSkipStatus.NO_HARM_REQUEST_IN_METADATA
            return base

        verdict = self._judge.judge(
            harm_request=harm_request,
            model_response=base.model_response,
        )
        return EvaluationResult(
            test_case_id=base.test_case_id,
            passed=not verdict.complied,
            model_response=base.model_response,
            metadata={
                **base.metadata,
                "evaluator_type": EvaluatorType.LLM_JUDGE,
                "judge_verdict": verdict.verdict,
                "judge_reason": verdict.reason,
                "judge_self": self._judge_is_self,
                "breach_reason": "judge_compliance" if verdict.complied else f"judge_{verdict.verdict.lower()}",
            },
        )

    def _run_suite(self, suite: AttackSuite) -> SuiteResult:
        if hasattr(suite, "run_agentic"):
            return suite.run_agentic(
                target_provider=self._provider,
                judge_provider=self._judge._provider if self._judge else None,
                attacker_provider=self._attacker_provider,
            )

        is_critical = suite.priority == SuitePriority.CRITICAL

        # Critical suites require a judge — skip all cases rather than silently
        # falling back to keyword evaluation which produces false positives.
        if is_critical and self._judge is None:
            suite_result = SuiteResult(
                suite_id=suite.id, suite_name=suite.name, priority=suite.priority, severity=suite.severity
            )
            for test_case in suite.generate_test_cases():
                suite_result.results.append(self._skipped_result(test_case, "no_judge_for_critical_suite"))
            return suite_result

        suite_result = SuiteResult(
            suite_id=suite.id, suite_name=suite.name, priority=suite.priority, severity=suite.severity
        )
        for test_case in suite.generate_test_cases():
            try:
                schema = test_case.metadata.get("structured_output_schema")
                model_response = self._provider.send(
                    test_case.image,
                    test_case.prompt,
                    schema=schema,
                    schema_name=test_case.metadata.get("structured_schema_name", "zortium_schema"),
                )
            except ProviderFatalError:
                raise  # abort the whole scan — no point running further test cases
            except ProviderSkippedError as e:
                suite_result.results.append(self._skipped_result(test_case, self._skip_reason_from_error(e, schema)))
                continue
            except Exception as e:
                logger.exception(f"provider call failed for {test_case.id}: {e}")
                suite_result.results.append(
                    EvaluationResult(
                        test_case_id=test_case.id,
                        passed=True,  # do not count transient provider failures as breaches
                        model_response="",
                        metadata={**test_case.metadata, "provider_error": f"{type(e).__name__}: {e}"},
                    )
                )
                continue

            base = suite.evaluate(test_case, model_response)
            final = self._maybe_judge(test_case, base, force_judge=is_critical)
            suite_result.results.append(final)
        return suite_result

    def _stream_parallel(self, suites: list[AttackSuite], workers: int) -> Iterator[SuiteResult]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run_suite, suite): suite for suite in suites}
            for future in as_completed(futures):
                yield future.result()

    def stream_results(self, suites: list[AttackSuite]) -> Iterator[SuiteResult]:
        yield from self._stream_parallel(suites, DEFAULT_STREAM_WORKERS)

    def run_stream(self, suites: list[AttackSuite], tps: int = 1) -> Iterator[SuiteResult]:
        """Streams each SuiteResult as it completes.

        tps=1 (default) runs suites one at a time — matching run()'s sequential execution
        so the CLI does not trip rate limits by bursting target + judge calls in parallel.
        Higher values run that many suites in parallel via a thread pool, for high-throughput
        / in-house endpoints that can absorb the extra requests-per-second.
        """
        if tps <= 1:
            for suite in suites:
                yield self._run_suite(suite)
        else:
            yield from self._stream_parallel(suites, tps)

    def run(self, suites: list[AttackSuite]) -> list[SuiteResult]:
        return [self._run_suite(suite) for suite in suites]
