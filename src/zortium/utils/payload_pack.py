"""
Harm-payload pack loader.

Loads `config/harm_payloads.json` and exposes a typed view of its categories
and prompts. Suites read from this loader so that the *mechanism* (FigStep,
GCG, encoding, etc.) is decoupled from the *payload* (system-prompt
extraction, malware, phishing, etc.).

The pack is treated as a singleton: it's loaded once and cached. Buyers who
want to override the pack can either edit `config/harm_payloads.json`
directly or set `ZORTIUM_PAYLOAD_PACK_PATH` to point at their own JSON file.

Schema (v1):
{
  "schema_version": 1,
  "license": "...",
  "attribution": [...],
  "categories": {
    "<category_id>": {
      "label": "...",
      "description": "...",
      "evaluator": "keyword" | "llm_judge",
      "compliance_keywords": ["..."],     # only for evaluator=="keyword"
      "prompts": [
        {"id": "...", "text": "the actual harmful request"}
      ]
    }
  }
}
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "config" / "harm_payloads.json"


@dataclass(frozen=True)
class HarmPrompt:
    id: str
    text: str


@dataclass(frozen=True)
class HarmCategory:
    id: str
    label: str
    description: str
    evaluator: str  # "keyword" | "llm_judge"
    compliance_keywords: tuple[str, ...]
    prompts: tuple[HarmPrompt, ...]


@dataclass(frozen=True)
class PayloadPack:
    schema_version: int
    license: str
    attribution: tuple[str, ...]
    categories: dict[str, HarmCategory]

    def category(self, category_id: str) -> HarmCategory:
        if category_id not in self.categories:
            raise KeyError(f"Unknown harm category {category_id!r}. " f"Known: {sorted(self.categories.keys())}")
        return self.categories[category_id]

    def resolve(
        self, category_ids: list[str] | None, prompts_per_category: int | None = None
    ) -> list[tuple[HarmCategory, HarmPrompt]]:
        """
        Resolve a list of (category_id) into the actual (category, prompt) pairs
        a suite should iterate over. If `category_ids` is None or empty, returns
        an empty list (suite should fall back to its own default behavior).
        If `prompts_per_category` is set, takes only the first N prompts from
        each category.
        """
        if not category_ids:
            return []
        out: list[tuple[HarmCategory, HarmPrompt]] = []
        for cid in category_ids:
            cat = self.category(cid)
            prompts = cat.prompts
            if prompts_per_category is not None:
                prompts = prompts[:prompts_per_category]
            for p in prompts:
                out.append((cat, p))
        return out


def _path() -> Path:
    override = os.environ.get("ZORTIUM_PAYLOAD_PACK_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_PATH


_cache: PayloadPack | None = None


def load() -> PayloadPack:
    global _cache
    if _cache is not None:
        return _cache
    raw = json.loads(_path().read_text(encoding="utf-8"))
    cats: dict[str, HarmCategory] = {}
    for cid, c in raw.get("categories", {}).items():
        prompts = tuple(HarmPrompt(id=p["id"], text=p["text"]) for p in c.get("prompts", []))
        cats[cid] = HarmCategory(
            id=cid,
            label=c.get("label", cid),
            description=c.get("description", ""),
            evaluator=c.get("evaluator", "keyword"),
            compliance_keywords=tuple(k.lower() for k in c.get("compliance_keywords", [])),
            prompts=prompts,
        )
    _cache = PayloadPack(
        schema_version=int(raw.get("schema_version", 1)),
        license=str(raw.get("license", "")),
        attribution=tuple(raw.get("attribution", [])),
        categories=cats,
    )
    return _cache


def reset_cache() -> None:
    """Test-only hook: forget the cached pack so it's reloaded on next access."""
    global _cache
    _cache = None
