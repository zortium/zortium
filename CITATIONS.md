# Citations and Attributions

Zortium builds on published research in adversarial machine learning and VLM safety evaluation.
Each attack suite in this repository adapts or is directly inspired by the work listed below.

---

## Benchmark Datasets

### MM-SafetyBench
> Liu, X., Zhu, F., Gu, J., Yu, B., Ding, Y., Liu, Y., & Nie, L. (2024).
> *MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models.*
> ECCV 2024. arXiv:2311.17600.

Inspiration for: `MMSafetyBenchSuite` (user-facing: **Query-Relevant Typographic Jailbreak**)
License: CC BY NC 4.0
Source: https://github.com/isXinLiu/MM-SafetyBench

> Zortium's Query-Relevant Typographic Jailbreak suite adapts the *methodology* of
> MM-SafetyBench — moving a harmful instruction out of the text prompt and into a
> rendered-text image. It does **not** ship or derive from the MM-SafetyBench corpus or
> images; every key phrase and prompt is Zortium-original content. No CC BY-NC material is
> bundled or distributed.

---

### JailBreakV-28K
> Luo, D., Ma, D., Liu, Z., Guo, L., & Xiao, C. (2024).
> *JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal LLMs against Jailbreak Attacks.*
> COLM 2024. arXiv:2404.03027.

Used by: `JailBreakV28KSuite`
License: MIT
Source: https://huggingface.co/datasets/JailBreakV-28K/JailBreakV-28K

---

## Attack Techniques

### FigStep — Typographic Visual Prompts
> Gong, Y., et al. (2023).
> *FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts.*
> arXiv:2311.05608.

Used by: `FigStepSuite`
License: MIT
Source: https://github.com/ThuCCSLab/FigStep

---

### GCG — Greedy Coordinate Gradient Adversarial Suffixes
> Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023).
> *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
> arXiv:2307.15043.

Used by: `GCGSuffixSuite` (transfer variant — precomputed suffixes only, no gradient access)
License: MIT
Source: https://github.com/llm-attacks/llm-attacks

---

### PAIR — Prompt Automatic Iterative Refinement
> Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023).
> *Jailbreaking Black Box Large Language Models in Twenty Queries.*
> arXiv:2310.08419.

Used by: `PairAgentSuite`
License: MIT
Source: https://github.com/patrickrchao/JailbreakingLLMs

---

### Image Hijacks — Behaviour-Matching Visual Attacks
> Bailey, L., Ong, E., Russell, S., & Emmons, S. (2023).
> *Image Hijacks: Adversarial Images can Control Generative Models at Runtime.*
> arXiv:2309.00236.

Used by: `ImageHijacksSuite` (black-box transfer adaptation — no gradient access to target VLM)

---

### Many-Shot Jailbreaking
> Anil, C., et al. (2024).
> *Many-Shot Jailbreaking.*
> Anthropic Technical Report.

Used by: `ManyShotTextSuite`

---

### Jailbroken — Competing Objectives / Refusal Suppression
> Wei, A., Haghtalab, N., & Steinhardt, J. (2023).
> *Jailbroken: How Does LLM Safety Training Fail?*
> NeurIPS 2023. arXiv:2307.02483.

Used by: `RefusalSuppressionSuite`, `JailBreakVisualSuite`, `EncodingAttackSuite` (Base64 / ROT13 / hex payload obfuscation)

---

### Typographic Attacks on CLIP
> Goh, G., Cammarata, N., Voss, C., Carter, S., Petrov, M., Schubert, L., Radford, A., & Olah, C. (2021).
> *Multimodal Neurons in Artificial Neural Networks.*
> Distill, 6(3). https://distill.pub/2021/multimodal-neurons/

Used by: `TypographicInjectionSuite` (generalises CLIP-era typographic attacks to instruction injection)

---

### Stroop Effect — Visual–Linguistic Conflict
> Stroop, J. R. (1935).
> *Studies of Interference in Serial Verbal Reactions.*
> Journal of Experimental Psychology, 18(6), 643–662.

Used by: `StroopSuite` (cross-modal adaptation — diagnostic probe of text/image conflict handling)

---

### Ignore Previous Prompt — Goal Hijacking / Prompt Injection
> Perez, F., & Ribeiro, I. (2022).
> *Ignore Previous Prompt: Attack Techniques for Language Models.*
> arXiv:2211.09527.

Used by: `GoalHijackingSuite`

---

### Multi-Modal Linkage
> Wang, Y., et al. (2024).
> *Jailbreak Large Vision-Language Models Through Multi-Modal Linkage.*
> arXiv:2412.00473.

Used by: `MultiModalLinkageSuite`

---

### HADES — Steganographic / Hidden Text Attacks
> Li, Y., et al. (2024).
> *Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models.*
> arXiv:2403.09792.

Used by: `SteganographicTextSuite` (simplified black-box variant)

---

## Original Techniques

The following suites implement attack patterns original to Zortium and are not direct adaptations
of a specific published paper:

- `SplitModalityInjectionSuite` — cross-modal instruction splitting to bypass per-modality filters
- `UISpoofingSuite` — authoritative UI surface spoofing (terminal, IDE, browser chrome)
- `CrossModalSemanticInjectionSuite` — pixel perturbation robustness probe (diagnostic, not jailbreak)
- `StructuredOutputInjectionSuite` — JSON schema / constrained-decoding control-plane injection
