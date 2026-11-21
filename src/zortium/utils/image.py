from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ImageUtils:
    """
    Global utility for loading images from any source and normalizing to JPEG bytes.
    Handles remote URLs (including SVG) and local file paths. Relative local
    paths are resolved against the project root so config entries work
    regardless of the process's current working directory.
    """

    # Some hosts (notably Wikimedia) return 403 to requests with Python's default
    # User-Agent. Identify ourselves properly per Wikimedia's UA policy.
    _USER_AGENT = "Zortium/0.2 (+https://github.com/zortium/zortium; VLM adversarial test suite)"
    _FETCH_TIMEOUT_SEC = 20

    @staticmethod
    def _fetch_url(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": ImageUtils._USER_AGENT})
        with urllib.request.urlopen(req, timeout=ImageUtils._FETCH_TIMEOUT_SEC) as response:
            return response.read()

    @staticmethod
    def _read_file(path: str) -> bytes:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        with open(p, "rb") as f:
            return f.read()

    @staticmethod
    def _to_jpeg(raw: bytes) -> bytes:
        if b"<svg" in raw[:256]:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            drawing = svg2rlg(io.BytesIO(raw))  # type: ignore[arg-type]
            if drawing is None:
                raise ValueError("Failed to parse SVG")
            raw = bytes(renderPM.drawToString(drawing, fmt="PNG"))  # type: ignore[arg-type]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    @staticmethod
    def bytes_to_array(image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0

    @staticmethod
    def array_to_jpeg(arr: np.ndarray, quality: int = 95) -> bytes:
        arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(arr_u8, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def path_to_jpeg(path: Path, quality: int = 92) -> bytes:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def load(source: str) -> bytes:
        """
        Load an image from a URL or local file path and return JPEG bytes.
        SVG sources are rasterized to PNG first, then converted to JPEG.
        """
        if source.startswith(("http://", "https://")):
            raw = ImageUtils._fetch_url(source)
        else:
            raw = ImageUtils._read_file(source)
        return ImageUtils._to_jpeg(raw)
