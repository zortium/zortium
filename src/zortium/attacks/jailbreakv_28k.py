"""
JailBreakV-28K benchmark suite.

Based on: Luo, Ma, Liu, Guo & Xiao (2024). "JailBreakV-28K: A Benchmark for
Assessing the Robustness of Multimodal Large Language Models against
Jailbreak Attacks" (COLM 2024, arXiv:2404.03027).

The benchmark contains 28,000 (text, image) jailbreak pairs spanning:

  - 5 attack methods       — Template (GCG + handcrafted), Persuade
                             (Persuasive Adversarial Prompts), Logic
                             (Cognitive Overload), FigStep, Query-Relevant.
  - 16 safety policies     — Fraud, Malware, Illegal Activity, Hate
                             Speech, Bias, Physical Harm, Privacy
                             Violation, ... (full list in the paper).
  - 6 image-type variants  — nature, random noise, blank, typography,
                             Stable Diffusion (SD), SD + typography.

ASSETS
------
Two assets are required:

  assets/jailbreakv_28k/mini_JailBreakV_28K.csv
    — bundled verbatim from HuggingFace JailbreakV-28K/JailBreakV-28k.
      280 (text, image_path, format, policy) rows.

  assets/jailbreakv_28k/images/{figstep,llm_transfer_attack,query_related}/
    — 300 real benchmark images downloaded from the same HF repo.
      Run `python -m zortium.attacks.jailbreakv_28k --download` to fetch them,
      or see the README for the one-liner.

IMAGE ASSIGNMENT
----------------
The mini CSV's `image_path` values point into the *full* gated 28K
dataset; only 3 of 280 rows happen to share an ID with the 300-image
public HF sample. The suite therefore assigns images by folder type
rather than exact ID.

Each format maps to a single source folder in the paper:

  Template, Persuade, Logic  →  llm_transfer_attack/
  figstep                    →  figstep/
  SD, SD_typo, typo          →  query_related/

This assignment is methodologically sound:

  - For LLM-transfer formats (Template / Persuade / Logic) the *image
    is structurally irrelevant to the attack*. The paper's own images
    for these rows are random nature scenes, noise canvases, or blank
    panes — filler backgrounds that the model barely uses. The attack
    vector is entirely in the jailbreak_query text. Any real image from
    the llm_transfer_attack/ pool serves the same purpose.

  - For image-based formats (figstep, SD, SD_typo, typo) the image IS
    the attack vector. figstep images contain the numbered-blank-list
    layout; query_related images contain Stable Diffusion scene
    content with or without overlaid typography. We use real images of
    the correct type from the HF-published pool, which is exactly the
    right population.

The suite deterministically picks images within each pool using the
row's CSV id mod pool size, so every scan is reproducible with the
same images, and samples spread across the available pool rather than
always reusing the first N.

EVALUATION
----------
The paper uses LlamaGuard. This suite wires into Zortium's two-tier
evaluator the same way all other suites do:

  - Fast path: `utils.refusal_aware.decide` with `flag_soft_breach=True`
    (engagement without explicit refusal is flagged).
  - Elevated path: `utils.llm_judge.judge_compliance` when the runner
    has a `judge_provider` AND the test case carries
    `evaluator_type == "llm_judge"`. All 16 JailBreakV-28K safety
    policies map to harm categories that are marked `evaluator:
    llm_judge` in `config/harm_payloads.json`, so the elevation happens
    automatically.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from zortium.attacks.base import AttackSuite
from zortium.models import EvaluationResult, TestCase
from zortium.constants import EvaluatorType
from zortium.utils.image import ImageUtils
from zortium.utils.payload_pack import HarmCategory, load as load_pack

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "jailbreakv_28k"
BUNDLED_CSV = ASSETS_DIR / "mini_JailBreakV_28K.csv"
IMAGES_DIR = ASSETS_DIR / "images"

HF_REPO = "JailbreakV-28K/JailBreakV-28k"
HF_BASE_URL = "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/resolve/main"
SUBFOLDERS = ("figstep", "llm_transfer_attack", "query_related")
JPEG_QUALITY = 92


class JBVFormat(str, Enum):
    TEMPLATE = "Template"
    PERSUADE = "Persuade"
    LOGIC = "Logic"
    FIGSTEP = "figstep"
    SD = "SD"
    SD_TYPO = "SD_typo"
    TYPO = "typo"


# Which image folder each format draws from. This follows directly from the
# paper: transfer-attack formats use llm_transfer_attack/ images; MLLM-attack
# formats use their own image folder.
FORMAT_TO_FOLDER: dict[str, str] = {
    JBVFormat.TEMPLATE: "llm_transfer_attack",
    JBVFormat.PERSUADE: "llm_transfer_attack",
    JBVFormat.LOGIC: "llm_transfer_attack",
    JBVFormat.FIGSTEP: "figstep",
    JBVFormat.SD: "query_related",
    JBVFormat.SD_TYPO: "query_related",
    JBVFormat.TYPO: "query_related",
}

POLICY_TO_HARM_CATEGORY: dict[str, str] = {
    "Malware": "cybercrime_malware",
    "Privacy Violation": "system_prompt_extraction",
    "Fraud": "deception_persuasion",
    "Hate Speech": "deception_persuasion",
    "Bias": "deception_persuasion",
    "Economic Harm": "deception_persuasion",
    "Physical Harm": "weapons_dangerous_substances",
    "Violence": "weapons_dangerous_substances",
    "Illegal Activity": "illegal_activity",
    "Unethical Behavior": "illegal_activity",
    "Government Decision": "illegal_activity",
    "Political Sensitivity": "deception_persuasion",
    "Animal Abuse": "illegal_activity",
    "Child Abuse Content": "illegal_activity",
    "Health Consultation": "deception_persuasion",
    "Tailored Unlicensed Advice": "deception_persuasion",
}


@dataclass(frozen=True)
class _JBVRow:
    id: str
    jailbreak_query: str
    redteam_query: str
    format: str
    policy: str
    image_path: str
    transfer_from_llm: bool


class JailBreakV28KSuite(AttackSuite):
    """
    Runs the JailBreakV-28K mini benchmark (280 rows, 7 formats, 16 policies)
    against the target VLM using real images from the HF-published pool.

    Requires `assets/jailbreakv_28k/images/` to be populated.
    Run `python -m zortium.attacks.jailbreakv_28k --download` if it is empty.

    Config knobs in attacks.json:
      samples_per_format  int  rows per attack format per scan (default 1)
      seed                int  sampling seed (default 1729)
      csv_path            str  override the bundled CSV (e.g., full 28K CSV)
    """

    @staticmethod
    def __resolve_harm_category(policy: str) -> HarmCategory | None:
        category_id = POLICY_TO_HARM_CATEGORY.get(policy)
        if not category_id:
            return None
        try:
            return load_pack().category(category_id)
        except KeyError:
            return None

    @staticmethod
    def cli_download() -> None:
        """Fetch all 300 benchmark images from HuggingFace. Called via --download."""
        headers = {"User-Agent": "Mozilla/5.0"}
        ok = fail = skip = 0
        for subfolder in SUBFOLDERS:
            api_url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/JailBreakV_28K/{subfolder}"
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                items = json.loads(r.read())
            hf_paths = [item["path"] for item in items if item.get("type") == "file"]
            print(f"  {subfolder}: {len(hf_paths)} files")
            for hf_path in hf_paths:
                dest = IMAGES_DIR / hf_path.split("/", 1)[1]
                if dest.exists():
                    skip += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = f"{HF_BASE_URL}/{hf_path}"
                try:
                    dl_req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(dl_req, timeout=30) as r:
                        dest.write_bytes(r.read())
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"    FAIL {hf_path}: {e}")
        print(f"Done: {ok} downloaded, {fail} failed, {skip} already cached.")

    def __init__(self) -> None:
        super().__init__()
        self._seed: int = int(self._config.get("seed", 1729))
        self._samples_per_format: int = int(self._config.get("samples_per_format", 1))
        csv_override = self._config.get("csv_path")
        self._csv_path: Path = Path(csv_override).expanduser().resolve() if csv_override else BUNDLED_CSV
        # Inventory available images per folder, sorted for reproducibility.
        self._pool: dict[str, list[Path]] = {}
        for folder in SUBFOLDERS:
            d = IMAGES_DIR / folder
            if d.exists():
                self._pool[folder] = sorted(p for p in d.iterdir() if p.is_file())

    def __images_available(self) -> bool:
        return all(len(self._pool.get(f, [])) > 0 for f in SUBFOLDERS)

    def __pick_image(self, folder: str, row_id: str) -> Path:
        pool = self._pool.get(folder, [])
        if not pool:
            raise FileNotFoundError(
                f"No images found in assets/jailbreakv_28k/images/{folder}/. "
                f"Run: python -m zortium.attacks.jailbreakv_28k --download"
            )

        # Deterministic assignment: row id (numeric) mod pool size so that
        # different rows spread across the pool rather than all using index 0.
        try:
            idx = int(row_id) % len(pool)
        except ValueError:
            idx = hash(row_id) % len(pool)
        return pool[idx]

    def __load_rows(self) -> list[_JBVRow]:
        if not self._csv_path.exists():
            return []

        out: list[_JBVRow] = []
        with self._csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(
                    _JBVRow(
                        id=str(r.get("id", "")),
                        jailbreak_query=str(r.get("jailbreak_query", "")),
                        redteam_query=str(r.get("redteam_query", "")),
                        format=str(r.get("format", "")),
                        policy=str(r.get("policy", "")),
                        image_path=str(r.get("image_path", "")),
                        transfer_from_llm=str(r.get("transfer_from_llm", "")).strip().lower() == "true",
                    )
                )
        return out

    def __stratified_sample(self, rows: list[_JBVRow]) -> list[_JBVRow]:
        rng = random.Random(self._seed)
        by_fmt: dict[str, list[_JBVRow]] = {}
        for r in rows:
            by_fmt.setdefault(r.format, []).append(r)
        selected: list[_JBVRow] = []
        for fmt in sorted(by_fmt.keys()):
            cands = list(by_fmt[fmt])
            rng.shuffle(cands)

            seen_policies: set[str] = set()
            chosen: list[_JBVRow] = []
            for c in cands:
                if c.policy in seen_policies and len(chosen) < self._samples_per_format:
                    continue
                chosen.append(c)
                seen_policies.add(c.policy)
                if len(chosen) >= self._samples_per_format:
                    break

            while len(chosen) < self._samples_per_format and cands:
                for c in cands:
                    if c not in chosen:
                        chosen.append(c)
                        break
                else:
                    break

            selected.extend(chosen)
        return selected

    def preflight(self) -> str | None:
        if self.__images_available():
            return None
        return (
            "JailBreakV-28K images not downloaded — this suite will be skipped.\n"
            "Run:  python -m zortium.attacks.jailbreakv_28k --download\n"
            f"Images are saved to: {IMAGES_DIR}"
        )

    def generate_test_cases(self) -> list[TestCase]:
        if not self.__images_available():
            raise FileNotFoundError(
                "JailBreakV-28K images not found. " "Run: python -m zortium.attacks.jailbreakv_28k --download"
            )
        rows = self.__load_rows()
        if not rows:
            return []
        sample = self.__stratified_sample(rows)
        cases: list[TestCase] = []

        for row in sample:
            folder = FORMAT_TO_FOLDER.get(row.format, "llm_transfer_attack")
            img_path = self.__pick_image(folder, row.id)
            image = ImageUtils.path_to_jpeg(img_path, quality=JPEG_QUALITY)
            harm_cat = self.__resolve_harm_category(row.policy)

            metadata: dict[str, Any] = {
                "jbv_id": row.id,
                "jbv_format": row.format,
                "jbv_policy": row.policy,
                "jbv_image_file": img_path.name,
                "jbv_image_folder": folder,
                "jbv_redteam_query": row.redteam_query,
                "jbv_transfer_from_llm": row.transfer_from_llm,
            }

            if harm_cat is not None:
                metadata.update(
                    {
                        "harm_category": harm_cat.id,
                        "harm_category_label": harm_cat.label,
                        "harm_request_text": row.redteam_query,
                        "evaluator_type": harm_cat.evaluator,
                        "compliance_keywords": list(harm_cat.compliance_keywords),
                    }
                )
            else:
                metadata.update(
                    {
                        "evaluator_type": EvaluatorType.KEYWORD,
                        "compliance_keywords": [],
                        "harm_request_text": row.redteam_query,
                    }
                )

            cases.append(
                TestCase(
                    id=f"{self.id}-{row.format}-{row.id}",
                    prompt=row.jailbreak_query,
                    image=image,
                    metadata=metadata,
                )
            )
        return cases


if __name__ == "__main__":

    if "--download" in sys.argv:
        print(f"Downloading JailBreakV-28K images to {IMAGES_DIR} ...")
        JailBreakV28KSuite.cli_download()
    else:
        print("Usage: python -m zortium.attacks.jailbreakv_28k --download")
