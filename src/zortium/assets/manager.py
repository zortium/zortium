import io
import urllib.request
from pathlib import Path

import cairosvg
from PIL import Image


class AssetManager:
    """
    Fetches images from URLs and stores them as JPEG files under assets/<suite_id>/.
    SVGs are converted to JPEG automatically. Assets are fetched once and reused.
    """

    _ASSETS_ROOT = Path(__file__).parent

    def __init__(self, suite_id: str) -> None:
        self._suite_dir = self._ASSETS_ROOT / suite_id
        self._suite_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _download(url: str) -> bytes:
        with urllib.request.urlopen(url) as response:
            return response.read()

    @staticmethod
    def _to_jpeg(url: str, raw: bytes) -> bytes:
        if url.endswith(".svg") or raw.lstrip()[:4] == b"<svg" or b"<svg" in raw[:256]:
            png_bytes = cairosvg.svg2png(bytestring=raw)
            raw = png_bytes
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def fetch(self, url: str, filename: str) -> bytes:
        """
        Return JPEG bytes for the given URL, using the cached file if it exists.
        filename should be the desired name without extension e.g. 'injection-pwned'.
        """
        dest = self._suite_dir / f"{filename}.jpg"
        if not dest.exists():
            raw = self._download(url)
            jpeg_bytes = self._to_jpeg(url, raw)
            dest.write_bytes(jpeg_bytes)
        return dest.read_bytes()
