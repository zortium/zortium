from __future__ import annotations

import base64
from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam


class PromptBuilder(ABC):

    @abstractmethod
    def build(self, prompt: str, image: bytes | None = None) -> list[ChatCompletionMessageParam]:
        """
        Convert a text prompt (and optionally raw image bytes) into the
        provider-specific messages payload expected by the underlying SDK.
        When `image` is None, the message contains only text — used by
        text-channel LLM attacks.
        """
        ...


class OpenAIPromptBuilder(PromptBuilder):
    """
    Builds messages for any OpenAI-compatible API (OpenAI, Groq, etc.).
    Detects image format from bytes and encodes as a base64 data URI.
    Falls back to a text-only message when no image is supplied.
    """

    _IMAGE_DETAIL = "auto"

    def build(self, prompt: str, image: bytes | None = None) -> list[ChatCompletionMessageParam]:
        text_part = ChatCompletionContentPartTextParam(type="text", text=prompt)
        if image is None:
            message = ChatCompletionUserMessageParam(role="user", content=[text_part])
            return [message]
        mime = self._detect_mime(image)
        encoded = base64.b64encode(image).decode("utf-8")
        image_part = ChatCompletionContentPartImageParam(
            type="image_url",
            image_url={"url": f"data:{mime};base64,{encoded}", "detail": self._IMAGE_DETAIL},
        )
        message = ChatCompletionUserMessageParam(role="user", content=[image_part, text_part])
        return [message]

    @staticmethod
    def _detect_mime(image: bytes) -> str:
        if image[:2] == b"\xff\xd8":
            return "image/jpeg"
        if image[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"
