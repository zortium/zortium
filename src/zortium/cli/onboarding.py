"""
First-run setup for the CLI: an arrow-key menu for the target, the judge, their
keys and the run options.

Opened by `zortium --init` (or a bare `zortium --config`), and automatically the
first time someone runs `zortium` in a terminal with nothing configured. It first
asks where the answers go: saved as your defaults (`<config dir>/config.yaml`, so a
plain `zortium` uses them), written to a YAML/JSON file for `-c FILE`, or nothing
saved and the equivalent command printed. Files holding keys are owner-only.
Flags, env vars and `-c FILE` still override saved defaults.
"""

from __future__ import annotations

import os
import json
import shlex
from enum import Enum
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import yaml
import questionary
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table
from questionary import Choice
from rich.console import Console

from zortium.settings import settings
from zortium.cli.config import ConfigLoader
from zortium.constants import ProviderKind

CONFIG_FILENAME = "config.yaml"
FILE_MODE = 0o600
DIR_MODE = 0o700

FILE_HEADER = (
    "# Zortium CLI settings — written by `zortium --init`.\n"
    "# Holds API keys: keep it private. Flags, ZORTIUM_* env vars and `-c FILE` override these values.\n"
)

# ── Question text ───────────────────────────────────────────────────────────

Q_DESTINATION = "How do you want to use these settings?"
Q_FORMAT = "File format:"
Q_FILE_PATH = "Write it to:"
Q_MAX_ASR = "Fail the scan when more than how many attacks succeed?"
Q_MAX_ASR_CUSTOM = "Fail when more than this percent of attacks succeed (0-100):"
Q_TARGET_PROVIDER = "Which provider hosts the model you want to test?"
Q_BASE_URL = "Endpoint base URL (OpenAI chat-completions compatible):"
Q_MODEL = "Which model?"
Q_MODEL_TYPED = "Model name:"
Q_API_KEY = "API key:"
Q_JUDGE_MODE = "Who grades the responses?"
Q_JUDGE_PROVIDER = "Which provider hosts the judge model?"
Q_JUDGE_MODEL = "Which judge model?"
Q_MODE = "Scan depth?"
Q_RATE_LIMIT = "When the endpoint is rate-limited?"
Q_TPS = "Suites in parallel?"
Q_SAVE = "Save these settings?"
Q_LOOKS_RIGHT = "Looks right?"

TYPE_MODEL = "Type a model name…"

# ── Menu text ───────────────────────────────────────────────────────────────

ACCENT = "#f59e0b"  # Zortium amber, same as the scan output

STYLE = questionary.Style(
    [
        ("qmark", f"fg:{ACCENT} bold"),
        ("question", "bold"),
        ("pointer", f"fg:{ACCENT} bold"),
        ("highlighted", f"fg:{ACCENT} bold"),
        ("selected", f"fg:{ACCENT}"),
        ("answer", f"fg:{ACCENT} bold"),
        ("instruction", "fg:#8a8a8a italic"),
    ]
)

SELECT_HINT = "(↑/↓ to move, Enter to pick)"

INTRO_TITLE = f"[bold {ACCENT}]Zortium setup[/bold {ACCENT}]"
INTRO_BODY = (
    "Four short steps: [bold]where settings go[/bold], the [bold]target[/bold] model to attack, "
    "the [bold]judge[/bold] that grades its replies, and [bold]run options[/bold].\n"
    "[dim]Every step has a sensible default, so Enter is usually enough. Ctrl-C quits and saves nothing.[/dim]"
)

# Each step's header: (title, one-line explanation).
STEPS = (
    ("Where your settings go", "Choose how you'll run scans. Run zortium --init again any time to change this."),
    (
        "The model to test",
        "The target is the vision model Zortium attacks. Its key is only ever sent to its own endpoint.",
    ),
    (
        "Who grades the replies",
        "A judge model reads each reply and decides whether the target complied. "
        "A separate model grades more strictly than the target grading itself.",
    ),
    ("Run options", "How a scan runs. The highlighted choice is the recommended one: Enter keeps it."),
)

# One line printed above each run-options question, saying what the option is (the choices carry the reasons).
HELP_MODE = "How many cases each attack suite runs.  [--fast / --no-fast]"
HELP_RATE_LIMIT = "What happens when the endpoint says it's busy (HTTP 429).  [--wait / --no-wait]"
HELP_TPS = "How many attack suites run at the same time.  [--tps N]"
HELP_MAX_ASR = (
    "For CI pipelines: make the zortium command fail (exit code 1) when too many attacks get through, "
    "so the build stops.  [--max-asr PCT]"
)

NEXT_STEPS_TITLE = "[bold]Next[/bold]"
NEXT_STEPS = (
    ("{run}", "start a scan"),
    ("{run} --output report.json", "also save every reply and verdict"),
    ("zortium --init", "change these settings"),
    ("zortium --help", "every flag"),
)
SAVED_NOTE = "[green]Saved to {path}[/green] [dim](readable only by you)[/dim]"
URL_INVALID = "Start with http:// or https://"
PERCENT_INVALID = "Enter a number from 0 to 100"

NEVER_FAIL = 100.0
GATE_CUSTOM = "custom"
NEVER_FAIL_LABEL = "Never fail, just report the results (recommended: see your model's baseline before setting a bar)"
CUSTOM_GATE_LABEL = "A different percent…"

# (percent, label) presets for the CI gate.
GATE_PRESETS = (
    (10.0, "More than 10%: strict, for models that should resist almost everything"),
    (20.0, "More than 20%: balanced, a common starting bar"),
    (30.0, "More than 30%: lenient, while you're still hardening the model"),
)
FILE_EXISTS_NOTE = "[yellow]{path} exists — it will be overwritten.[/yellow]"
WROTE_FILE_NOTE = "[green]Wrote {path}[/green]"
RUN_COMMAND_NOTE = "[dim]Nothing saved. Run this to scan:[/dim]"
KEY_ENV_NOTE = "[dim]Set your keys first (they stay out of the command):[/dim]"

DEFAULTS_LABEL = "Save as my defaults: a plain `zortium` uses them (recommended: set up once, nothing to retype)"
FILE_LABEL = "Write a config file (YAML or JSON) for a project or CI: run with `zortium -c FILE`"
COMMAND_LABEL = "Save nothing: just show me the full command to run"
MODEL_REQUIRED = "A model name is required"

JUDGE_SEPARATE_LABEL = "A separate judge model (recommended: a model grading itself under-reports breaches)"
JUDGE_SELF_LABEL = "Let the target grade itself (lenient: breaches may be under-reported)"

FAST_LABEL = "Fast: representative cases from every suite (recommended: minutes, right for routine checks and CI)"
THOROUGH_LABEL = "Thorough: every case in every suite (slower; use it before a release)"

SKIP_LABEL = "Skip: mark it unavailable, leave it out of the score (recommended: a scan never stalls)"
WAIT_LABEL = "Wait: retry until it clears so every case gets a verdict (can take hours; not for CI)"

TPS_OPTIONS = (1, 2, 4, 8)
TPS_CUSTOM = "custom"
CUSTOM_TPS_LABEL = "A different number…"
Q_TPS_CUSTOM = "How many suites at once? (a whole number, 1 or more)"
TPS_INVALID = "Enter a whole number, 1 or more"
TPS_SAFE_NOTE = " suite at a time (recommended: safest for shared or rate-limited keys)"
TPS_FAST_NOTE = " suites at once (faster on endpoints with high rate limits)"
KEEP_KEY_HINT = "Enter keeps the saved key"
REUSE_KEY_HINT = "Enter reuses the target key"
NO_KEY_HINT = "leave empty if the endpoint needs no key"
ENV_KEY_HINT = "leave empty to read {env} instead, safer if the file is committed"

TARGET_KEY_ENV = "ZORTIUM_API_KEY"
JUDGE_KEY_ENV = "ZORTIUM_JUDGE_API_KEY"

# Run options the menu offers, with the values used when the user doesn't change them.
RUN_DEFAULTS = {"fast": True, "wait": False, "tps": 1, "max_asr": 100.0}


class Destination(str, Enum):
    DEFAULTS = "defaults"
    FILE = "file"
    COMMAND = "command"


class FileFormat(str, Enum):
    YAML = "yaml"
    JSON = "json"


class JudgeMode(str, Enum):
    SEPARATE = "separate"
    SELF = "self"


class SetupCancelled(Exception):
    """Raised when the user backs out of the menu (Ctrl-C / Esc); nothing is saved."""


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    kind: ProviderKind
    base_url: str
    models: tuple[str, ...]
    needs_key: bool = True


# Same endpoints as the web scanner's presets (server/config/providers.json), which the pip package doesn't ship.
PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        "openrouter",
        "OpenRouter — hundreds of open models, one key",
        ProviderKind.OPENAI,
        "https://openrouter.ai/api/v1",
        ("qwen/qwen3-vl-32b-instruct", "google/gemma-3-27b-it", "meta-llama/llama-4-scout"),
    ),
    ProviderPreset(
        "openai",
        "OpenAI",
        ProviderKind.OPENAI,
        "https://api.openai.com/v1",
        ("gpt-4o-mini", "gpt-4o"),
    ),
    ProviderPreset(
        "anthropic",
        "Anthropic (Claude)",
        ProviderKind.ANTHROPIC,
        "https://api.anthropic.com",
        ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    ),
    ProviderPreset(
        "gemini",
        "Google Gemini",
        ProviderKind.OPENAI,
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("gemini-2.5-flash", "gemini-2.5-flash-lite"),
    ),
    ProviderPreset(
        "together",
        "Together AI",
        ProviderKind.OPENAI,
        "https://api.together.xyz/v1",
        (),
    ),
    ProviderPreset(
        "groq",
        "Groq (text models only)",
        ProviderKind.OPENAI,
        "https://api.groq.com/openai/v1",
        ("llama-3.3-70b-versatile", "openai/gpt-oss-120b"),
    ),
    ProviderPreset(
        "ollama",
        "Ollama — local, no key",
        ProviderKind.OPENAI,
        "http://localhost:11434/v1",
        (),
        needs_key=False,
    ),
    ProviderPreset(
        "custom",
        "Custom OpenAI-compatible endpoint (vLLM, TGI, your gateway)",
        ProviderKind.OPENAI,
        "",
        (),
    ),
)

CUSTOM = PRESETS[-1]


class UserConfig:
    """Reads and writes the saved CLI settings file, and the config files setup can export."""

    @staticmethod
    def path() -> Path:
        return settings.config_dir / CONFIG_FILENAME

    @staticmethod
    def load() -> dict:
        """The saved settings, or {} when none exist yet. Raises ValueError on a malformed file."""
        path = UserConfig.path()
        if not path.is_file():
            return {}
        return ConfigLoader.load(path)

    @staticmethod
    def write(path: Path, values: dict, fmt: FileFormat = FileFormat.YAML) -> Path:
        """Writes values as YAML (with a header comment) or JSON, owner read/write only: it may hold keys."""
        path.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)

        if fmt == FileFormat.JSON:
            body = json.dumps(values, indent=2) + "\n"
        else:
            body = FILE_HEADER + yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

        # Create owner-only from the first byte (no window where the keys are world-readable),
        # then re-apply in case the file already existed with looser permissions.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.chmod(path, FILE_MODE)

        return path

    @staticmethod
    def save(values: dict) -> Path:
        return UserConfig.write(UserConfig.path(), values)


class SetupWizard:
    """The interactive menu. `run()` returns the chosen settings and where they went, or raises SetupCancelled."""

    def __init__(self, console: Console) -> None:
        self._console = console

    @staticmethod
    def __ask(question: questionary.Question) -> Any:
        answer = question.ask()
        if answer is None:
            raise SetupCancelled()
        return answer

    @staticmethod
    def __mask(key: str) -> str:
        if not key:
            return "not set"
        if len(key) <= 10:
            return "•" * len(key)
        return f"{key[:4]}…{key[-4:]}"

    @staticmethod
    def __preset_for(base_url: str, kind: str) -> ProviderPreset:
        for preset in PRESETS:
            if preset.base_url and preset.base_url.rstrip("/") == (base_url or "").rstrip("/"):
                return preset

        if kind == ProviderKind.ANTHROPIC:
            return next(p for p in PRESETS if p.kind == ProviderKind.ANTHROPIC)

        return CUSTOM

    @staticmethod
    def _choice_for(choices: list[Choice], value: object) -> Choice | None:
        return next((choice for choice in choices if choice.value == value), None)

    @staticmethod
    def command_for(values: dict) -> str:
        """The `zortium …` command equivalent to values. Keys are left out: they belong in env vars."""
        parts = ["zortium"]

        if values["provider"] != ProviderKind.OPENAI:
            parts += ["--provider", values["provider"]]
        if values["base_url"]:
            parts += ["--base-url", values["base_url"]]
        parts += ["--model", values["model"]]

        if values["judge_model"]:
            if values["judge_provider"] != values["provider"]:
                parts += ["--judge-provider", values["judge_provider"]]
            if values["judge_base_url"] != values["base_url"]:
                parts += ["--judge-base-url", values["judge_base_url"]]
            parts += ["--judge-model", values["judge_model"]]

        if not values["fast"]:
            parts.append("--no-fast")
        if values["wait"]:
            parts.append("--wait")
        if values["tps"] != RUN_DEFAULTS["tps"]:
            parts += ["--tps", str(values["tps"])]
        if values["max_asr"] != RUN_DEFAULTS["max_asr"]:
            parts += ["--max-asr", f"{values['max_asr']:g}"]

        return " ".join(shlex.quote(part) for part in parts)

    def _pick_destination(self) -> tuple[Destination, Path | None, FileFormat]:
        choices = [
            Choice(DEFAULTS_LABEL, value=Destination.DEFAULTS),
            Choice(FILE_LABEL, value=Destination.FILE),
            Choice(COMMAND_LABEL, value=Destination.COMMAND),
        ]
        destination = self.__ask(
            questionary.select(Q_DESTINATION, choices=choices, style=STYLE, instruction=SELECT_HINT)
        )
        if destination != Destination.FILE:
            return destination, None, FileFormat.YAML

        fmt = self.__ask(
            questionary.select(
                Q_FORMAT,
                choices=[Choice("YAML", value=FileFormat.YAML), Choice("JSON", value=FileFormat.JSON)],
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )
        name = self.__ask(questionary.text(Q_FILE_PATH, default=f"zortium.{fmt.value}", style=STYLE))
        path = Path(name.strip()).expanduser()
        if path.exists():
            self._console.print(FILE_EXISTS_NOTE.format(path=path))

        return destination, path, fmt

    def _pick_provider(self, message: str, saved_url: str, saved_kind: str) -> ProviderPreset:
        default = self.__preset_for(saved_url, saved_kind) if saved_url else PRESETS[0]
        choices = [Choice(preset.label, value=preset) for preset in PRESETS]

        return self.__ask(
            questionary.select(
                message,
                choices=choices,
                default=self._choice_for(choices, default),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

    def _pick_base_url(self, preset: ProviderPreset, saved_url: str) -> str:
        if preset is not CUSTOM:
            return preset.base_url

        url = self.__ask(
            questionary.text(
                Q_BASE_URL,
                default=saved_url or "",
                validate=lambda v: v.strip().startswith(("http://", "https://")) or URL_INVALID,
                style=STYLE,
            )
        )
        return url.strip().rstrip("/")

    def _pick_model(self, message: str, preset: ProviderPreset, saved_model: str) -> str:
        if preset.models:
            choices = [Choice(model, value=model) for model in preset.models]
            choices.append(Choice(TYPE_MODEL, value=TYPE_MODEL))

            default = saved_model if saved_model in preset.models else (TYPE_MODEL if saved_model else None)
            picked = self.__ask(
                questionary.select(
                    message,
                    choices=choices,
                    default=self._choice_for(choices, default),
                    style=STYLE,
                    instruction=SELECT_HINT,
                )
            )
            if picked != TYPE_MODEL:
                return picked

        model = self.__ask(
            questionary.text(
                Q_MODEL_TYPED,
                default=saved_model or "",
                validate=lambda v: bool(v.strip()) or MODEL_REQUIRED,
                style=STYLE,
            )
        )
        return model.strip()

    def _pick_key(
        self,
        label: str,
        preset: ProviderPreset,
        destination: Destination,
        *,
        env: str,
        fallback: str,
        fallback_hint: str = KEEP_KEY_HINT,
    ) -> str:
        """Asks for one endpoint's key; empty input returns fallback (the saved key, or the target's)."""
        if not preset.needs_key or destination == Destination.COMMAND:
            return ""

        if destination == Destination.FILE:
            hint = ENV_KEY_HINT.format(env=env)
        else:
            hint = fallback_hint if fallback else NO_KEY_HINT

        entered = self.__ask(questionary.password(f"{label} {Q_API_KEY} ({hint})", style=STYLE))
        return entered.strip() or fallback

    def _pick_run_options(self, saved: dict) -> dict:
        current = {key: saved.get(key, default) for key, default in RUN_DEFAULTS.items()}

        self._hint(HELP_MODE)
        mode_choices = [Choice(FAST_LABEL, value=True), Choice(THOROUGH_LABEL, value=False)]
        fast = self.__ask(
            questionary.select(
                Q_MODE,
                choices=mode_choices,
                default=self._choice_for(mode_choices, current["fast"]),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

        self._hint(HELP_RATE_LIMIT)
        wait_choices = [Choice(SKIP_LABEL, value=False), Choice(WAIT_LABEL, value=True)]
        wait = self.__ask(
            questionary.select(
                Q_RATE_LIMIT,
                choices=wait_choices,
                default=self._choice_for(wait_choices, current["wait"]),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

        self._hint(HELP_TPS)
        tps_choices = [Choice(f"{n}" + (TPS_SAFE_NOTE if n == 1 else TPS_FAST_NOTE), value=n) for n in TPS_OPTIONS]
        tps_choices.append(Choice(CUSTOM_TPS_LABEL, value=TPS_CUSTOM))

        saved_tps = int(current["tps"])
        tps = self.__ask(
            questionary.select(
                Q_TPS,
                choices=tps_choices,
                default=self._choice_for(tps_choices, saved_tps if saved_tps in TPS_OPTIONS else TPS_CUSTOM),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

        if tps == TPS_CUSTOM:
            entered = self.__ask(
                questionary.text(
                    Q_TPS_CUSTOM,
                    default=str(saved_tps),
                    validate=lambda v: (v.strip().isdigit() and int(v) >= 1) or TPS_INVALID,
                    style=STYLE,
                )
            )
            tps = int(entered)

        self._hint(HELP_MAX_ASR)
        gate_choices = [Choice(NEVER_FAIL_LABEL, value=NEVER_FAIL)]
        gate_choices += [Choice(label, value=pct) for pct, label in GATE_PRESETS]
        gate_choices.append(Choice(CUSTOM_GATE_LABEL, value=GATE_CUSTOM))

        saved_gate = float(current["max_asr"])
        gate_default = saved_gate if any(c.value == saved_gate for c in gate_choices) else GATE_CUSTOM
        max_asr = self.__ask(
            questionary.select(
                Q_MAX_ASR,
                choices=gate_choices,
                default=self._choice_for(gate_choices, gate_default),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

        if max_asr == GATE_CUSTOM:
            max_asr = self.__ask(
                questionary.text(
                    Q_MAX_ASR_CUSTOM,
                    default=f"{saved_gate:g}" if saved_gate < NEVER_FAIL else "",
                    validate=lambda v: v.strip().replace(".", "", 1).isdigit() and float(v) <= 100 or PERCENT_INVALID,
                    style=STYLE,
                )
            )

        return {"fast": fast, "wait": wait, "tps": tps, "max_asr": float(max_asr)}

    def _hint(self, text: str) -> None:
        self._console.print(f"[dim]{text}[/dim]", highlight=False, markup=True)

    def _section(self, step: int) -> None:
        title, blurb = STEPS[step - 1]
        self._console.print()
        self._console.print(
            Rule(f"[bold {ACCENT}]{step}/{len(STEPS)}[/bold {ACCENT}]  [bold]{title}[/bold]", align="left", style="dim")
        )
        self._console.print(f"[dim]{blurb}[/dim]")

    def _next_steps(self, run: str) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", no_wrap=True)
        table.add_column(style="dim")
        for command, meaning in NEXT_STEPS:
            table.add_row(command.format(run=run), meaning)
        return Panel(table, title=NEXT_STEPS_TITLE, title_align="left", border_style="dim", expand=False)

    def _summary(self, values: dict, destination: Destination) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()

        table.add_row("target", f"{values['model']}  [dim]{values['base_url']}[/dim]")
        if destination != Destination.COMMAND:
            table.add_row(
                "target key", self.__mask(values["api_key"]) if values["api_key"] else f"from {TARGET_KEY_ENV}"
            )

        if values["judge_model"]:
            table.add_row("judge", f"{values['judge_model']}  [dim]{values['judge_base_url']}[/dim]")
            if destination != Destination.COMMAND:
                table.add_row(
                    "judge key",
                    self.__mask(values["judge_api_key"]) if values["judge_api_key"] else f"from {JUDGE_KEY_ENV}",
                )
        else:
            table.add_row("judge", "[yellow]target grades itself (lenient)[/yellow]")

        table.add_row("scan", "fast" if values["fast"] else "thorough")
        table.add_row("on 429", "wait and retry" if values["wait"] else "skip the case")
        table.add_row("parallel", f"{values['tps']} suite(s)")
        table.add_row(
            "CI gate",
            (
                "never fail, just report"
                if values["max_asr"] >= NEVER_FAIL
                else f"fail when more than {values['max_asr']:g}% of attacks succeed"
            ),
        )

        return table

    def _finish(
        self, values: dict, destination: Destination, path: Path | None, fmt: FileFormat, key_envs: list[str]
    ) -> None:
        if destination == Destination.DEFAULTS:
            self._console.print(SAVED_NOTE.format(path=UserConfig.save(values)))
            self._console.print(self._next_steps("zortium"))
            return

        if destination == Destination.FILE:
            # Blank keys are left out so the env vars fill them at run time.
            written = UserConfig.write(path, {k: v for k, v in values.items() if v != ""}, fmt)
            self._console.print(WROTE_FILE_NOTE.format(path=written))
            self._console.print(self._next_steps(f"zortium -c {shlex.quote(str(path))}"))
            return

        if key_envs:
            self._console.print(KEY_ENV_NOTE)
            for env in key_envs:
                self._console.print(f"  export {env}=…", markup=False, highlight=False)
        self._console.print(RUN_COMMAND_NOTE)
        self._console.print(f"  {self.command_for(values)}", markup=False, highlight=False)

    def run(self, saved: dict) -> tuple[dict, Destination]:
        self._console.print()
        self._console.print(Panel(INTRO_BODY, title=INTRO_TITLE, title_align="left", border_style=ACCENT))

        self._section(1)
        destination, path, fmt = self._pick_destination()

        self._section(2)
        target = self._pick_provider(Q_TARGET_PROVIDER, saved.get("base_url", ""), saved.get("provider", ""))
        base_url = self._pick_base_url(target, saved.get("base_url", ""))
        # Saved model and key only carry over while the endpoint is unchanged.
        same_target = base_url == (saved.get("base_url") or "").rstrip("/")
        model = self._pick_model(Q_MODEL, target, saved.get("model", "") if same_target else "")
        api_key = self._pick_key(
            "Target", target, destination, env=TARGET_KEY_ENV, fallback=saved.get("api_key", "") if same_target else ""
        )

        self._section(3)
        judge_choices = [
            Choice(JUDGE_SEPARATE_LABEL, value=JudgeMode.SEPARATE),
            Choice(JUDGE_SELF_LABEL, value=JudgeMode.SELF),
        ]
        judge_default = JudgeMode.SEPARATE if saved.get("judge_model") or not saved else JudgeMode.SELF
        judge_mode = self.__ask(
            questionary.select(
                Q_JUDGE_MODE,
                choices=judge_choices,
                default=self._choice_for(judge_choices, judge_default),
                style=STYLE,
                instruction=SELECT_HINT,
            )
        )

        key_envs = [TARGET_KEY_ENV] if target.needs_key else []
        judge = {"judge_provider": "", "judge_base_url": "", "judge_model": "", "judge_api_key": ""}
        if judge_mode == JudgeMode.SEPARATE:
            saved_judge_url = saved.get("judge_base_url") or saved.get("base_url", "")
            judge_preset = self._pick_provider(Q_JUDGE_PROVIDER, saved_judge_url, saved.get("judge_provider", ""))
            judge_url = self._pick_base_url(judge_preset, saved.get("judge_base_url", ""))
            same_judge = judge_url == (saved.get("judge_base_url") or "").rstrip("/")
            judge_model = self._pick_model(
                Q_JUDGE_MODEL, judge_preset, saved.get("judge_model", "") if same_judge else ""
            )

            # The judge's key is asked on its own; on the target's endpoint, Enter reuses the target key.
            if judge_url == base_url and api_key:
                fallback, fallback_hint = api_key, REUSE_KEY_HINT
            else:
                fallback, fallback_hint = (saved.get("judge_api_key", "") if same_judge else ""), KEEP_KEY_HINT
            judge_key = self._pick_key(
                "Judge", judge_preset, destination, env=JUDGE_KEY_ENV, fallback=fallback, fallback_hint=fallback_hint
            )

            # A judge on its own endpoint needs its own key; on the target's, the target key is inherited.
            if judge_preset.needs_key and judge_url != base_url:
                key_envs.append(JUDGE_KEY_ENV)
            judge = {
                "judge_provider": judge_preset.kind.value,
                "judge_base_url": judge_url,
                "judge_model": judge_model,
                "judge_api_key": judge_key,
            }

        self._section(4)
        values = {
            "provider": target.kind.value,
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            **judge,
            **self._pick_run_options(saved),
        }

        self._console.print()
        self._console.print(self._summary(values, destination))
        self._console.print()

        question = Q_LOOKS_RIGHT if destination == Destination.COMMAND else Q_SAVE
        if not self.__ask(questionary.confirm(question, default=True, style=STYLE)):
            raise SetupCancelled()

        self._finish(values, destination, path, fmt, key_envs)

        return values, destination
