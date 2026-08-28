from __future__ import annotations

import json
from abc import ABC
from typing import Any
from pathlib import Path

from zortium.utils.image import ImageUtils
from zortium.utils.payload_pack import load as load_pack
from zortium.evaluators.refusal_aware import RefusalAware
from zortium.models import EvaluationResult, SuiteResult, TestCase
from zortium.constants import ScanMode, SuitePriority, SuiteSeverity

PRIORITY_TO_SEVERITY = {
    SuitePriority.CRITICAL: SuiteSeverity.HIGH,
    SuitePriority.MEDIUM: SuiteSeverity.MID,
    SuitePriority.DIAGNOSTIC: SuiteSeverity.DIAGNOSTIC,
}

# Version the ATLAS tags are pinned to — technique IDs are renumbered across
# ATLAS releases, so this travels with the report to keep the mapping legible.
ATLAS_VERSION = "v5.1.0"


class AttackConfig:
    _CONFIG_PATH = Path(__file__).parent.parent / "config" / "attacks.json"
    _cache: dict | None = None

    @classmethod
    def load(cls) -> dict | None:
        if cls._cache is None:
            with open(cls._CONFIG_PATH, encoding="utf-8") as f:
                cls._cache = json.load(f)
        return cls._cache

    @classmethod
    def severity_by_id(cls) -> dict[str, SuiteSeverity]:
        """Map every suite id → its display severity, for resolving saved scans."""
        out: dict[str, SuiteSeverity] = {}
        for entry in (cls.load() or {}).values():
            if not (isinstance(entry, dict) and "id" in entry):
                continue
            raw = entry.get("severity")
            if raw is not None:
                out[entry["id"]] = SuiteSeverity(raw)
            else:
                priority = SuitePriority(entry.get("priority", SuitePriority.MEDIUM))
                out[entry["id"]] = PRIORITY_TO_SEVERITY[priority]
        return out

    @classmethod
    def tags_by_id(cls) -> dict[str, dict]:
        """Map every suite id → its {atlas, owasp} tags, for resolving saved scans."""
        out: dict[str, dict] = {}
        for entry in (cls.load() or {}).values():
            if not (isinstance(entry, dict) and "id" in entry):
                continue
            out[entry["id"]] = {"atlas": entry.get("atlas"), "owasp": entry.get("owasp")}
        return out


class AttackSuite(ABC):
    """
    Base class for all attack suites.
    Suite metadata and test cases are driven by config/attacks.json.
    Subclasses implement generate_test_cases() and optionally override
    evaluate() for suites that need custom decision logic.
    """

    def __init__(self) -> None:
        self._config = AttackConfig.load()[self.__class__.__name__]

    @property
    def id(self) -> str:
        return self._config["id"]

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def paper_ref(self) -> str:
        return self._config["paper_ref"]

    @property
    def priority(self) -> SuitePriority:
        return SuitePriority(self._config.get("priority", SuitePriority.MEDIUM))

    @property
    def severity(self) -> SuiteSeverity:
        raw = self._config.get("severity")
        return SuiteSeverity(raw) if raw is not None else PRIORITY_TO_SEVERITY[self.priority]

    def configure_for_mode(self, mode: ScanMode) -> bool:
        """Configure this suite for the given scan mode. Return False to exclude it."""
        return True

    def preflight(self) -> str | None:
        """Return a message explaining why this suite cannot run yet (and how to fix
        it), or None when it is ready. Suites that depend on downloaded assets override
        this so a runner can warn and skip rather than aborting the whole scan."""
        return None

    def _resolve_payload_pairs(self) -> list[tuple[Any, Any]]:
        """Returns (HarmCategory, HarmPrompt) pairs from config harm_categories."""
        pack = load_pack()
        category_ids = self._config.get("harm_categories", ["system_prompt_extraction"])
        prompts_per_category = int(self._config.get("prompts_per_category", 1))
        return pack.resolve(category_ids, prompts_per_category)

    def _base_test_metadata(self, v_id: str, cat, prompt, **extra) -> dict:
        """Standard metadata dict for a payload-pack test case. Pass suite-specific fields as kwargs."""
        return {
            "variant": v_id,
            "harm_category": cat.id,
            "harm_category_label": cat.label,
            "harm_prompt_id": prompt.id,
            "harm_request_text": prompt.text,
            "evaluator_type": cat.evaluator,
            "compliance_keywords": list(cat.compliance_keywords),
            **extra,
        }

    def _generate_image(self, test_id: str) -> bytes:
        """
        Override in subclasses where image is null in the config and must be
        generated programmatically. Raises by default.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}: no image source for test '{test_id}' "
            f"and _generate_image() is not implemented."
        )

    def generate_test_cases(self) -> list[TestCase]:
        cases = []
        for test in self._config["tests"]:
            image_source = test.get("image")
            if image_source is not None:
                image = ImageUtils.load(image_source)
            else:
                image = self._generate_image(test["id"])
            cases.append(TestCase(id=test["id"], image=image, prompt=test["prompt"]))
        return cases

    def evaluate(self, test_case: TestCase, model_response: str) -> EvaluationResult:
        """
        Default evaluation using refusal-aware decision. All test_case metadata
        is passed through to the result (minus compliance_keywords), with
        decision fields appended. Override for suites with custom logic.
        """
        decide_kwargs: dict = {
            "compliance_keywords": test_case.metadata.get("compliance_keywords", []),
            "flag_soft_breach": bool(self._config.get("flag_soft_breach", False)),
        }

        if self._config.get("refusal_phrases"):
            decide_kwargs["refusal_phrases"] = self._config["refusal_phrases"]
        decision = RefusalAware.decide(model_response, **decide_kwargs)

        passthrough = {k: v for k, v in test_case.metadata.items() if k != "compliance_keywords"}

        return EvaluationResult(
            test_case_id=test_case.id,
            passed=decision.passed,
            model_response=model_response,
            metadata={
                **passthrough,
                "breach_reason": decision.reason,
                "refusal_phrase_hit": decision.refusal_phrase_hit,
                "compliance_hits": decision.compliance_hits,
            },
        )
