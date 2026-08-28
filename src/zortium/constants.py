from __future__ import annotations

from enum import Enum


class ScanStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluatorType(str, Enum):
    KEYWORD = "keyword"
    LLM_JUDGE = "llm_judge"


class SuitePriority(str, Enum):
    CRITICAL = "critical"
    MEDIUM = "medium"
    DIAGNOSTIC = "diagnostic"


class SuiteSeverity(str, Enum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"
    DIAGNOSTIC = "diagnostic"

    @property
    def weight(self) -> int:
        """How much a breach on this tier counts toward the weighted overall ASR.
        HIGH counts 3x a LOW one; DIAGNOSTIC is 0 — diagnostic suites never affect
        scoring. This is the single source of truth for severity weighting."""
        return {
            SuiteSeverity.HIGH: 3,
            SuiteSeverity.MID: 2,
            SuiteSeverity.LOW: 1,
            SuiteSeverity.DIAGNOSTIC: 0,
        }[self]


class ScanMode(str, Enum):
    FAST = "fast"
    THOROUGH = "thorough"


class RateLimitPolicy(str, Enum):
    SKIP = "skip"
    WAIT = "wait"
