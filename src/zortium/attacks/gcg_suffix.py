"""
Precomputed transfer-suffix jailbreak suite.

Based on: Zou et al. (2023). "Universal and Transferable Adversarial Attacks
on Aligned Language Models" (arXiv:2307.15043).

Mechanism: append published-style transferable suffix strings to the base
request. This is NOT Greedy Coordinate Gradient optimization. GCG requires
white-box access to gradients / logits, which OpenAI-compatible customer
endpoints do not expose. Zortium therefore treats this as a transfer-suffix
probe: useful to test, but not the original optimization attack.

Architecture (post harm-pack refactor):
  variant (mechanism)  ×  harm_category (payload pack)  →  one test case

Each variant carries the suffix string. Each harm_category carries the
payload(s) drawn from `config/harm_payloads.json` (system-prompt extraction,
malware, weapons, deception, illegal activity, harassment, PII).

Evaluation: base verdict from utils.refusal_aware.decide(); runner upgrades
to llm_judge when the category requests it (config/harm_payloads.json,
`evaluator: "llm_judge"`).
"""

from __future__ import annotations

from zortium.attacks.base import AttackSuite
from zortium.models import TestCase


class GCGSuffixSuite(AttackSuite):

    def generate_test_cases(self) -> list[TestCase]:
        variants = self._config.get("variants", [])
        category_prompts = self._resolve_payload_pairs()
        cases: list[TestCase] = []

        for variant in variants:
            v_id = variant["id"]
            suffix = variant["suffix"]
            for cat, prompt in category_prompts:
                full_prompt = f"{prompt.text} {suffix}".strip()
                cases.append(
                    TestCase(
                        id=f"{self.id}-{v_id}-{cat.id}-{prompt.id}",
                        prompt=full_prompt,
                        image=None,
                        metadata=self._base_test_metadata(
                            v_id,
                            cat,
                            prompt,
                            suffix_preview=suffix[:60] + ("…" if len(suffix) > 60 else ""),
                        ),
                    )
                )

        return cases
