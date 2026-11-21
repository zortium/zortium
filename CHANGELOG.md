# Changelog

Notable changes to Zortium. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: [SemVer](https://semver.org/spec/v2.0.0.html); dates are UTC.

## [Unreleased]

## [0.2.3] - 2026-11-21

### Fixed

- **CLI:** 0.2.2 crashed on startup on a fresh install (newer typer no longer installs click).
- **CLI:** key-less endpoints such as Ollama work again with newer OpenAI SDKs; the OpenAI and Anthropic SDKs are capped at their tested major versions.

## [0.2.2] - 2026-11-21

### Added

- **CLI:** `zortium --init` walks you through the target, the judge, their keys and run options, recommending a default for each and saying why, then saves them as your defaults, writes a YAML/JSON file, or prints the command.
- **CLI:** native Anthropic support with `--provider anthropic`.
- **Cloud:** a redesigned scan page with live validation and your recent scans.

### Fixed

- Every scan crashed in the Visual Perturbation Drift suite (introduced in 0.2.1).
- Runaway model replies could stall a scan; replies are now capped at 4,096 tokens.
- Provider and judge errors are reported as unavailable instead of counting as resisted.
- The `--max-asr` gate no longer passes a scan that evaluated nothing.
- **CLI:** a judge on a different endpoint no longer gets the target's API key.

## [0.2.1] - 2026-08-28

### Fixed

- Installs on machines without a system Cairo library.

## [0.2.0] - 2026-08-18

### Added

- Bring-your-own LLM judge (`--judge-model`); without one, the target grades itself with a warning.
- `--fast` / `--no-fast`, `--wait` for rate limits, and `--tps` for parallel suites.
- `--config`, `--output` JSON reports and the `--max-asr` CI gate.
- New suites (Image Hijacks, JailBreakV-28K, Query-Relevant Typographic, Structured Output Injection) and severity tiers.
- Apache-2.0 license.

### Removed

- Marker-string (`PWNED`) scoring, which inflated results. The PAIR suite is switched off.

## [0.1.0] - 2026-05-03

### Added

- First release: scan any OpenAI-compatible vision model with visual and text-channel attack suites.

[Unreleased]: https://github.com/zortium/zortium
[0.2.3]: https://pypi.org/project/zortium/0.2.3/
[0.2.2]: https://pypi.org/project/zortium/0.2.2/
[0.2.1]: https://pypi.org/project/zortium/0.2.1/
[0.2.0]: https://pypi.org/project/zortium/0.2.0/
[0.1.0]: https://pypi.org/project/zortium/0.1.0/
