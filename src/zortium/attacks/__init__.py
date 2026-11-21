from __future__ import annotations

from zortium.constants import ScanMode
from zortium.attacks.base import AttackSuite
from zortium.attacks.stroop import StroopSuite
from zortium.attacks.figstep import FigStepSuite
from zortium.attacks.gcg_suffix import GCGSuffixSuite
from zortium.attacks.ui_spoofing import UISpoofingSuite
from zortium.attacks.image_hijacks import ImageHijacksSuite
from zortium.attacks.many_shot_text import ManyShotTextSuite
from zortium.attacks.goal_hijacking import GoalHijackingSuite
from zortium.attacks.jailbreakv_28k import JailBreakV28KSuite
from zortium.attacks.mm_safetybench import MMSafetyBenchSuite
from zortium.attacks.encoding_attack import EncodingAttackSuite
from zortium.attacks.jailbreak_visual import JailBreakVisualSuite
from zortium.attacks.multimodal_linkage import MultiModalLinkageSuite
from zortium.attacks.refusal_suppression import RefusalSuppressionSuite
from zortium.attacks.steganographic_text import SteganographicTextSuite
from zortium.attacks.typographic_injection import TypographicInjectionSuite
from zortium.attacks.split_modality_injection import SplitModalityInjectionSuite
from zortium.attacks.structured_output_injection import StructuredOutputInjectionSuite
from zortium.attacks.cross_modal_semantic_injection import CrossModalSemanticInjectionSuite

ACTIVE_SUITE_CLASSES: list[type[AttackSuite]] = [
    # ── VLM (vision-channel) suites ──
    TypographicInjectionSuite,
    SplitModalityInjectionSuite,
    CrossModalSemanticInjectionSuite,
    JailBreakVisualSuite,
    SteganographicTextSuite,
    UISpoofingSuite,
    StroopSuite,
    FigStepSuite,
    MultiModalLinkageSuite,
    MMSafetyBenchSuite,
    # PairAgentSuite — temporarily deactivated (attacker+judge loop is too costly
    # to run per scan). The suite, its config entry and its tests all remain; add
    # it back here to re-enable. Surfaced as "Coming soon" in the UI.
    StructuredOutputInjectionSuite,
    ImageHijacksSuite,
    JailBreakV28KSuite,
    # ── LLM (text-channel) suites ──
    GCGSuffixSuite,
    EncodingAttackSuite,
    RefusalSuppressionSuite,
    ManyShotTextSuite,
    GoalHijackingSuite,
]


class SuiteRegistry:

    @staticmethod
    def build_active_suites(mode: ScanMode = ScanMode.FAST) -> list[AttackSuite]:
        """
        Build fresh instances of every active suite. Fresh instances matter
        because some suites (e.g. CrossModalSemanticInjectionSuite) cache
        per-scan state such as baseline responses. Reusing instances across
        scans would leak that state.

        Each suite class declares its own mode behaviour via configure_for_mode()
        — SuiteRegistry has no suite-specific knowledge.
        """
        suites: list[AttackSuite] = []
        for cls in ACTIVE_SUITE_CLASSES:
            suite = cls()
            if suite.configure_for_mode(mode):
                suites.append(suite)
        return suites
