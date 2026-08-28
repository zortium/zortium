from __future__ import annotations

from pathlib import Path

import yaml

# Recognised keys in a config file, mapped to the scan command's parameter names.
# Anything else in the file is reported so typos surface instead of silently no-op'ing.
CONFIG_KEYS = frozenset(
    {
        "base_url",
        "model",
        "api_key",
        "judge_model",
        "judge_base_url",
        "judge_api_key",
        "max_asr",
        "fast",
        "wait",
        "tps",
    }
)


class ConfigLoader:
    """Loads a YAML or JSON settings file into a dict of scan options.

    yaml.safe_load parses JSON too, so a single loader covers both formats.
    Values only supply defaults — a flag or env var for the same option wins
    (precedence is resolved by the caller via the click parameter source).
    """

    @staticmethod
    def load(path: Path) -> dict:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("config file must contain a mapping of option: value")

        unknown = set(data) - CONFIG_KEYS
        if unknown:
            known = ", ".join(sorted(CONFIG_KEYS))
            raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}. Valid keys: {known}")

        return data
