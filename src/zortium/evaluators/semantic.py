"""
Lightweight semantic-drift scoring for comparing two VLM descriptions of the
same underlying scene. Intentionally dependency-free: tokenises on alphabetic
runs, strips stopwords and short tokens, and computes cosine similarity on
term-frequency vectors.

For Zortium' current stage this is sufficient to flag gross semantic drift
(e.g. "cat on a bed" -> "geometric coloured pattern"). It can be swapped for
an embedding-based judge later without changing call sites.
"""

from __future__ import annotations

import re
import math
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z]+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "for",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "into",
        "onto",
        "over",
        "under",
        "there",
        "here",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "also",
        "very",
        "some",
        "any",
        "all",
        "each",
        "both",
        "few",
        "more",
        "most",
        "other",
        "such",
        "than",
        "too",
        "so",
        "just",
        "only",
        "not",
        "image",
        "picture",
        "photo",
        "photograph",
        "shows",
        "show",
        "showing",
        "see",
        "seen",
        "appears",
        "appear",
        "depicts",
        "depict",
        "looks",
        "look",
        "seems",
        "seem",
        "contains",
        "contain",
        "describes",
        "describe",
        "overall",
        "main",
        "scene",
        "setting",
        "detail",
        "details",
    }
)

MIN_TOKEN_LEN = 3


class SemanticSimilarity:

    @staticmethod
    def __tokens(text: str) -> list[str]:
        if not text:
            return []
        return [t.lower() for t in TOKEN_RE.findall(text) if len(t) >= MIN_TOKEN_LEN and t.lower() not in STOPWORDS]

    def cosine_similarity(self, a: str, b: str) -> float:
        """
        Cosine similarity of term-frequency vectors over content tokens.
        Returns a value in [0, 1]. Returns 0.0 if either side has no content tokens.
        """
        ta, tb = Counter(self.__tokens(a)), Counter(self.__tokens(b))
        if not ta or not tb:
            return 0.0

        shared = set(ta) & set(tb)
        dot = sum(ta[k] * tb[k] for k in shared)

        if dot == 0:
            return 0.0

        na = math.sqrt(sum(v * v for v in ta.values()))
        nb = math.sqrt(sum(v * v for v in tb.values()))
        return dot / (na * nb)

    def semantic_drift(self, baseline: str, candidate: str) -> float:
        """
        Drift score in [0, 1]: 0.0 = descriptions identical in content vocabulary,
        1.0 = completely disjoint.
        """
        return 1.0 - self.cosine_similarity(baseline, candidate)
