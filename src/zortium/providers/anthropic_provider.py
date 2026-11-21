"""
Anthropic (Claude) VLM provider — native Messages API via the official SDK.

Same shape as OpenAICompatibleProvider: raw (base_url, api_key, model), the
shared HostedProvider retry/error/token plumbing, structured output through
`output_config.format`. `base_url` may be None for the public Anthropic API or
point at an Anthropic-compatible gateway.

Deliberately NOT enabled: server-side refusal `fallbacks`. Zortium measures the
target model; letting a refusal silently re-run on a different model would
attribute that model's answer to the target.
"""

from __future__ import annotations

from enum import Enum

from anthropic import Anthropic, Timeout
from anthropic.types import Message, TextBlock, OutputConfigParam, JSONOutputFormatParam

from zortium.constants import RateLimitPolicy
from zortium.providers.base import ProviderSkippedError
from zortium.providers.prompt_builder import AnthropicPromptBuilder
from zortium.providers.hosted import BAD_REQUEST, MAX_OUTPUT_TOKENS, STRUCTURED_OUTPUT_UNSUPPORTED, HostedProvider

DEFAULT_BASE_URL = "https://api.anthropic.com"

# A safety-classifier decline can come back with no text at all. Surface it as an
# explicit refusal so the judge / keyword evaluator scores it instead of seeing "".
CLASSIFIER_REFUSAL_TEXT = "[Refused by the model's safety classifier: {category}]"


class AnthropicStopReason(str, Enum):
    REFUSAL = "refusal"


class AnthropicProvider(HostedProvider[Message]):

    def __init__(
        self,
        base_url: str | None,
        api_key: str,
        model: str,
        *,
        rate_limit_policy: RateLimitPolicy = RateLimitPolicy.SKIP,
    ) -> None:
        super().__init__(base_url or DEFAULT_BASE_URL, model, rate_limit_policy=rate_limit_policy)
        self._client = Anthropic(
            api_key=api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        )
        self._prompt_builder = AnthropicPromptBuilder()

    def _read_response(self, response: Message) -> tuple[str, int, int]:
        text = "".join(block.text for block in response.content if isinstance(block, TextBlock))
        if not text and response.stop_reason == AnthropicStopReason.REFUSAL:
            category = response.stop_details.category if response.stop_details else None
            text = CLASSIFIER_REFUSAL_TEXT.format(category=category or "unspecified")
        return text, response.usage.input_tokens, response.usage.output_tokens

    def send(
        self,
        image: bytes | None,
        prompt: str,
        schema: dict | None = None,
        *,
        schema_name: str = "zortium_schema",
    ) -> str:
        messages = self._prompt_builder.build(prompt, image=image)
        if schema is not None:
            # Anthropic's json_schema format has no name field, so schema_name is unused here.
            output_config = OutputConfigParam(format=JSONOutputFormatParam(type="json_schema", schema=schema))
            return self._complete(
                lambda: self._client.messages.create(
                    model=self._model,
                    messages=messages,
                    output_config=output_config,
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                bad_request_error=ProviderSkippedError(STRUCTURED_OUTPUT_UNSUPPORTED),
            )
        # Same 400 policy as OpenAICompatibleProvider: a rejected body skips this case only.
        return self._complete(
            lambda: self._client.messages.create(model=self._model, messages=messages, max_tokens=MAX_OUTPUT_TOKENS),
            bad_request_error=ProviderSkippedError(BAD_REQUEST),
        )
