# Zortium

**Adversarial attack test suite for vision-language models.**

[![PyPI](https://img.shields.io/pypi/v/zortium.svg)](https://pypi.org/project/zortium/)
[![Python](https://img.shields.io/pypi/pyversions/zortium.svg)](https://pypi.org/project/zortium/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Zortium runs a structured battery of adversarial attacks against any OpenAI-compatible VLM endpoint and reports a per-suite Attack Success Rate (ASR). Think [promptfoo](https://github.com/promptfoo/promptfoo), for vision models — systematic, repeatable, CI-ready.

## Install

```bash
pip install zortium
```

Python 3.10+. No system dependencies.

## Quick start

```bash
zortium \
  --base-url https://openrouter.ai/api/v1 --model google/gemma-3-27b-it --api-key $OPENROUTER_API_KEY \
  --judge-model gpt-4o-mini --judge-base-url https://api.openai.com/v1 --judge-api-key $OPENAI_API_KEY
```

The command has two halves — the **target** you're testing (`--base-url` · `--model` · `--api-key`) and the **judge** that grades whether the target complied (`--judge-*`). Every flag has a `ZORTIUM_*` environment variable, so keys never touch your shell history.

> No judge configured? Zortium falls back to the **target grading itself** with a warning — fine for a first look, but self-grading is lenient, so treat those numbers as a floor. Pass `--judge-model` for results you can act on.

You get a per-suite breach-rate table and a headline ASR:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Attack Suite                         ┃ Severity ┃ Breach Rate ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ Typographic Visual Prompt Injection  │   HIGH   │          0% │  PASS  │
│ FigStep Typographic Jailbreak        │   HIGH   │         25% │  WARN  │
│ Goal Hijacking                       │   MID    │         50% │  FAIL  │
│ …                                    │    …     │           … │   …    │
├──────────────────────────────────────┼──────────┼─────────────┼────────┤
│ Overall                              │          │         12% │  WARN  │
└──────────────────────────────────────┴──────────┴─────────────┴────────┘
```

## CI gate

```bash
zortium --max-asr 20   # exits 1 if overall ASR exceeds 20%
```

Every flag has a `ZORTIUM_*` env var, so secrets go in CI variables — see the [docs](https://zortium.dev/docs#ci) for a copy-paste GitHub Actions job.

## Documentation

Full documentation — every flag, all 18 attack suites, scan modes, the LLM judge, config files, CI recipes, supported providers, and architecture — lives at **[zortium.dev/docs](https://zortium.dev/docs)**.

## Safety

Zortium is a **defensive** tool. Run it only against models you own or are authorized to test — by design it submits real jailbreak prompts, harmful requests, and adversarial images to measure refusal behaviour.

## License

Apache-2.0 — see [LICENSE](LICENSE). Research attribution in [CITATIONS.md](CITATIONS.md).
