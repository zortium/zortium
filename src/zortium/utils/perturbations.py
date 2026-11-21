"""
Adversarial perturbation strategies for attack suites.

All strategies take a float32 image array in [0, 1] with shape (H, W, 3) and
return a perturbed array clipped to [0, 1]. They are pure-numpy and deterministic
when given a seeded `np.random.Generator`.

IMPORTANT — every strategy here is a BLACK-BOX variant. Gradient-based attacks
(FGSM/PGD/Sin.Attack/MixAttack in their original papers) require white-box
access to the target model to compute the perturbation direction that maximises
loss. Zortium runs against arbitrary OpenAI-compatible API endpoints, where no
such access exists. We therefore implement the *pattern family* honestly: random
sign noise within an L_inf budget, sinusoidal patterns at a chosen frequency
with random orientation, etc. This tests the encoder's robustness to the
attack's perturbation distribution; it does not claim the per-target success
rates reported in the original papers (those require gradient guidance).

References
----------
- FGSM:        Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
- PGD:         Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
- Patch:       Brown et al., "Adversarial Patch" (2018)
- Sin/Mix:     Tu et al., "How Many Unicorns Are in This Image?" — UCSC vllm-safety-benchmark (ECCV 2024)
- Doubly-UAP:  "Doubly-UAP: Universal Adversarial Perturbation Against VLMs" (2024).
               Same mechanistic insight as Sin/Mix above — disruption of CLIP-style
               encoder mid-to-late-layer value vectors via high-spatial-frequency
               perturbation. Their contribution is gradient-trained per-encoder
               specificity, which requires white-box access to the target VLM and
               is therefore out of scope for Zortium's black-box runtime. The
               *pattern family* is covered by sin_attack and mix_attack below.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image


class PerturbationStrategy(str, Enum):
    NONE = "none"
    FGSM = "fgsm"
    PGD = "pgd"
    UNIVERSAL_PATCH = "universal_patch"
    CHECKERBOARD = "checkerboard"
    LOW_FREQ = "low_freq"
    CHANNEL_SHIFT = "channel_shift"
    SIN_ATTACK = "sin_attack"
    MIX_ATTACK = "mix_attack"


class Perturbations:

    @staticmethod
    def fgsm(image: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=image.shape)
        return np.clip(image + epsilon * sign, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def pgd(
        image: np.ndarray,
        epsilon: float,
        alpha: float,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Iterative L_inf projected random attack. Each step takes a random sign
        step of size `alpha` and projects back into the `epsilon`-ball around
        the original image.
        """
        adv = image + rng.uniform(-epsilon, epsilon, size=image.shape).astype(np.float32)
        adv = np.clip(adv, 0.0, 1.0)
        low = np.maximum(image - epsilon, 0.0)
        high = np.minimum(image + epsilon, 1.0)
        for _ in range(steps):
            step = alpha * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=image.shape)
            adv = np.clip(adv + step, low, high)
        return adv.astype(np.float32)

    @staticmethod
    def universal_patch(
        image: np.ndarray,
        epsilon: float,
        patch_fraction: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Concentrates the perturbation budget into a square region in the centre
        of the image. Probes whether the vision encoder anchors to a dominant
        spatial region (cf. Brown et al., "Adversarial Patch").
        """
        H, W, C = image.shape
        ph = max(1, int(round(H * patch_fraction)))
        pw = max(1, int(round(W * patch_fraction)))
        y0 = (H - ph) // 2
        x0 = (W - pw) // 2
        patch_noise = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(ph, pw, C))
        adv = image.copy()
        adv[y0 : y0 + ph, x0 : x0 + pw] = np.clip(image[y0 : y0 + ph, x0 : x0 + pw] + epsilon * patch_noise, 0.0, 1.0)
        return adv.astype(np.float32)

    @staticmethod
    def checkerboard(image: np.ndarray, epsilon: float) -> np.ndarray:
        """
        High-frequency structured noise. Adjacent pixels alternate sign, and
        R and B channels are given opposing signs vs G to bias the encoder's
        colour-channel response.
        """
        H, W, _ = image.shape
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]
        base = ((ys + xs) % 2.0) * 2.0 - 1.0
        signs = np.stack([base, -base, base], axis=-1)
        return np.clip(image + epsilon * signs, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def low_freq_noise(
        image: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
        lowres: int = 8,
    ) -> np.ndarray:
        """
        Smooth global noise: sample a small lowres x lowres noise field in [-1, 1]
        per channel, bilinearly upsample to the full image, then add within the
        epsilon budget. Tests robustness to smooth colour-field shifts.
        """
        H, W, C = image.shape
        small = rng.uniform(-1.0, 1.0, size=(lowres, lowres, C)).astype(np.float32)
        upsampled = np.empty((H, W, C), dtype=np.float32)
        for c in range(C):
            as_u8 = ((small[..., c] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            up = np.array(
                Image.fromarray(as_u8, mode="L").resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            )
            upsampled[..., c] = up / 127.5 - 1.0
        return np.clip(image + epsilon * upsampled, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def channel_shift(image: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Uniform RGB channel bias: R shifted up by +epsilon, B shifted down by
        -epsilon (G left alone). Probes how much the vision encoder relies on
        absolute colour calibration vs. relative scene content.
        """
        shift = np.zeros_like(image, dtype=np.float32)
        shift[..., 0] = +epsilon
        shift[..., 2] = -epsilon
        return np.clip(image + shift, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _sinusoidal_field(
        H: int,
        W: int,
        C: int,
        frequency: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Build a sinusoidal interference pattern in [-1, 1] with shape (H, W, C).
        Random orientation, per-channel phase offset. Frequency is in cycles
        across the longer image dimension.
        """
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(H, dtype=np.float32)[:, None]
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_per_channel = rng.uniform(0.0, 2.0 * np.pi, size=C).astype(np.float32)
        longest = float(max(H, W))
        coord = (np.cos(theta) * xs + np.sin(theta) * ys) / longest
        return np.cos(2.0 * np.pi * frequency * coord[..., None] + phase_per_channel[None, None, :]).astype(np.float32)

    @staticmethod
    def sin_attack(
        image: np.ndarray,
        epsilon: float,
        frequency: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sinusoidal high-frequency overlay (black-box variant of Sin.Attack from
        UCSC vllm-safety-benchmark). `frequency` is in cycles across the longer
        image dimension; values 8–32 produce visibly periodic but readable images.
        """
        pattern = Perturbations._sinusoidal_field(image.shape[0], image.shape[1], image.shape[2], frequency, rng)
        return np.clip(image + epsilon * pattern, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def mix_attack(
        image: np.ndarray,
        epsilon: float,
        frequency: float,
        lowres: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Combined low-frequency smooth noise + sinusoidal high-frequency overlay
        (black-box variant of MixAttack from UCSC vllm-safety-benchmark). Budget
        split evenly across the two modes, then projected back into the L_inf
        epsilon-ball.
        """
        H, W, C = image.shape
        small = rng.uniform(-1.0, 1.0, size=(lowres, lowres, C)).astype(np.float32)
        upsampled = np.empty((H, W, C), dtype=np.float32)
        for c in range(C):
            as_u8 = ((small[..., c] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            up = np.array(
                Image.fromarray(as_u8, mode="L").resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            )
            upsampled[..., c] = up / 127.5 - 1.0
        sin_pattern = Perturbations._sinusoidal_field(H, W, C, frequency, rng)
        combined = 0.5 * upsampled + 0.5 * sin_pattern
        delta = np.clip(epsilon * combined, -epsilon, epsilon)
        return np.clip(image + delta, 0.0, 1.0).astype(np.float32)
