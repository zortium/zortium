"""
Generic OpenAI-compatible VLM provider.

Takes raw (base_url, api_key, model) — no env-variable indirection. Designed
for the HTTP server path where credentials arrive per-request from the
frontend form. Works for any endpoint that speaks the OpenAI chat-completions
protocol: OpenAI itself, Groq, Together, Mistral, Ollama (via openai shim),
vLLM, TGI with OpenAI adapter, and most in-house VLM gateways.

For env-driven CLI usage, keep using GroqProvider.
"""

from __future__ import annotations

import httpx
import threading
from collections.abc import Callable

from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from zortium.constants import RateLimitPolicy
from zortium.providers.ratelimit import RateLimitResolver
from zortium.providers.prompt_builder import OpenAIPromptBuilder
from zortium.providers.base import ProviderFatalError, ProviderSkippedError, VLMProvider


class OpenAICompatibleProvider(VLMProvider):

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        rate_limit_policy: RateLimitPolicy = RateLimitPolicy.SKIP,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        )

        self._prompt_builder = OpenAIPromptBuilder()

        self._rate_limit_policy = rate_limit_policy
        self._rate_limiter = RateLimitResolver()

        self._connection_ok = False

        # Token accounting — accumulated across every call this provider makes.
        # Public so the scan service can read totals after a run. Guarded by a
        # lock because stream_results drives calls from a ThreadPoolExecutor.
        self.input_tokens = 0
        self.output_tokens = 0
        self._token_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _complete(self, make_call: Callable[[], ChatCompletion], *, bad_request_error: Exception) -> str:
        try:
            response = self._rate_limiter.execute_with_retry(make_call, policy=self._rate_limit_policy)
        except APIConnectionError as e:
            # A drop/reset after we've already reached the endpoint is transient —
            # skip this case and keep scanning. If we've never connected, the Base
            # URL is likely wrong, so abort the whole scan with a clear message.
            if self._connection_ok:
                raise ProviderSkippedError("Connection error — this test case was skipped.") from e
            raise ProviderFatalError(
                f"Could not connect to {self._base_url} — check that your Base URL is reachable."
            ) from e

        except AuthenticationError as e:
            raise ProviderFatalError("Authentication failed — check that your API key is correct.") from e

        except PermissionDeniedError as e:
            raise ProviderFatalError(
                "Access denied — your API key may not have permission for this model or endpoint."
            ) from e

        except NotFoundError as e:
            raise ProviderFatalError(
                f"Endpoint or model not found — check your Base URL and Model name. ({self._base_url}, {self._model})"
            ) from e

        except BadRequestError as e:
            raise bad_request_error from e

        self._connection_ok = True

        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._token_lock:
                self.input_tokens += usage.prompt_tokens or 0
                self.output_tokens += usage.completion_tokens or 0

        return response.choices[0].message.content or ""

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
                    model=self._model, messages=messages, response_format=response_format
                ),
                bad_request_error=ProviderSkippedError("structured_output_unsupported"),
            )
        # A 400 here is per-request, not per-scan: genuine misconfiguration surfaces
        # as 401/403 (auth) or 404 (not-found), which stay fatal above. A 400 means
        # the endpoint accepted the URL+model but rejected THIS body — an input
        # guardrail, a content filter, or an adversarial payload (e.g. a GCG suffix
        # a provider's tokenizer/moderation rejects). Skip the case and keep going;
        # aborting the whole scan over one bad input is wrong.
        return self._complete(
            lambda: self._client.chat.completions.create(model=self._model, messages=messages),
            bad_request_error=ProviderSkippedError("bad_request"),
        )
