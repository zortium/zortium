"""
Shared image-rendering primitives for VLM attack suites.
"""

from __future__ import annotations

import io

from PIL import Image, ImageFont

_BODY_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_BOLD_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)

_MONO_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Courier New.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
)


class Render:

    @staticmethod
    def load_font(size: int, *, kind: str = "body") -> ImageFont.ImageFont:
        """Try a list of platform fonts; fall back to Pillow's bundled default."""
        candidates = {
            "body": _BODY_FONT_CANDIDATES,
            "bold": _BOLD_FONT_CANDIDATES,
            "mono": _MONO_FONT_CANDIDATES,
        }.get(kind, _BODY_FONT_CANDIDATES)
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    @staticmethod
    def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        """Greedy word-wrap by pixel width using font.getbbox()."""
        out: list[str] = []
        for paragraph in text.splitlines():
            if not paragraph.strip():
                out.append("")
                continue
            current = ""
            for word in paragraph.split():
                candidate = (current + " " + word).strip()
                bbox = font.getbbox(candidate)
                width = bbox[2] - bbox[0]
                if width <= max_width or not current:
                    current = candidate
                else:
                    out.append(current)
                    current = word
            if current:
                out.append(current)
        return out

    @staticmethod
    def to_jpeg(img: Image.Image, *, quality: int = 92) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def to_png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
