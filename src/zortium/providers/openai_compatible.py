"""
Generic OpenAI-compatible VLM provider.

Takes raw (base_url, api_key, model) — no env-variable indirection. Designed
for the HTTP server path where credentials arrive per-request from the
frontend form. Works for any endpoint that speaks the OpenAI chat-completions
protocol: OpenAI itself, Groq, Together, Mistral, Ollama (via openai shim),
vLLM, TGI with OpenAI adapter, and most in-house VLM gateways.

Retry, error mapping and token accounting live in HostedProvider.
"""

from __future__ import annotations

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from zortium.constants import RateLimitPolicy
from zortium.providers.base import ProviderSkippedError
from zortium.providers.prompt_builder import OpenAIPromptBuilder
from zortium.providers.hosted import BAD_REQUEST, MAX_OUTPUT_TOKENS, STRUCTURED_OUTPUT_UNSUPPORTED, HostedProvider

# Sent as the bearer token when the endpoint needs no key; such endpoints ignore it.
NO_API_KEY = "no-key"


class OpenAICompatibleProvider(HostedProvider[ChatCompletion]):

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        rate_limit_policy: RateLimitPolicy = RateLimitPolicy.SKIP,
    ) -> None:
        super().__init__(base_url, model, rate_limit_policy=rate_limit_policy)
        self._client = OpenAI(
            # Key-less endpoints (Ollama, local vLLM) still need a non-empty value for the SDK.
            api_key=api_key or NO_API_KEY,
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        )
        self._prompt_builder = OpenAIPromptBuilder()

    def _read_response(self, response: ChatCompletion) -> tuple[str, int, int]:
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is None:
            return text, 0, 0
        return text, usage.prompt_tokens or 0, usage.completion_tokens or 0

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
            response_format = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(name=schema_name, strict=True, schema=schema),
            )
            return self._complete(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                bad_request_error=ProviderSkippedError(STRUCTURED_OUTPUT_UNSUPPORTED),
            )
        # A 400 here is per-request, not per-scan: genuine misconfiguration surfaces
        # as 401/403 (auth) or 404 (not-found), which stay fatal above. A 400 means
        # the endpoint accepted the URL+model but rejected THIS body — an input
        # guardrail, a content filter, or an adversarial payload (e.g. a GCG suffix
        # a provider's tokenizer/moderation rejects). Skip the case and keep going;
        # aborting the whole scan over one bad input is wrong.
        return self._complete(
            lambda: self._client.chat.completions.create(
                model=self._model, messages=messages, max_tokens=MAX_OUTPUT_TOKENS
            ),
            bad_request_error=ProviderSkippedError(BAD_REQUEST),
        )
