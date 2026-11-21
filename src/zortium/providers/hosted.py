"""
Shared plumbing for SDK-backed hosted providers (OpenAI-compatible, Anthropic).

Owns the parts that are identical across SDKs: rate-limit/connection retry,
mapping SDK errors to ProviderFatalError / ProviderSkippedError, and thread-safe
token accounting. Subclasses only build the request and read the response.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from typing import Generic, TypeVar
from collections.abc import Callable

import openai
import anthropic

from zortium.constants import RateLimitPolicy
from zortium.providers.ratelimit import CONNECTION_ERRORS, RateLimitResolver
from zortium.providers.base import ProviderFatalError, ProviderSkippedError, VLMProvider

ResponseT = TypeVar("ResponseT")

# Caps output only (input/context untouched). Uncapped, a runaway generation took 387s
# for one call and left scans stuck at N-1/N. Refusal-vs-compliance is clear well within this.
MAX_OUTPUT_TOKENS = 4096

BAD_REQUEST = "bad_request"
STRUCTURED_OUTPUT_UNSUPPORTED = "structured_output_unsupported"

AUTH_ERRORS = (openai.AuthenticationError, anthropic.AuthenticationError)
NOT_FOUND_ERRORS = (openai.NotFoundError, anthropic.NotFoundError)
BAD_REQUEST_ERRORS = (openai.BadRequestError, anthropic.BadRequestError)
PERMISSION_ERRORS = (openai.PermissionDeniedError, anthropic.PermissionDeniedError)


class HostedProvider(VLMProvider, Generic[ResponseT]):

    def __init__(self, base_url: str, model: str, *, rate_limit_policy: RateLimitPolicy) -> None:
        self._base_url = base_url
        self._model = model

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

    @abstractmethod
    def _read_response(self, response: ResponseT) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens) from an SDK response."""
        ...

    def _complete(self, make_call: Callable[[], ResponseT], *, bad_request_error: Exception) -> str:
        try:
            response = self._rate_limiter.execute_with_retry(make_call, policy=self._rate_limit_policy)
        except CONNECTION_ERRORS as e:
            # A drop/reset after we've already reached the endpoint is transient —
            # skip this case and keep scanning. If we've never connected, the Base
            # URL is likely wrong, so abort the whole scan with a clear message.
            if self._connection_ok:
                raise ProviderSkippedError("Connection error — this test case was skipped.") from e
            raise ProviderFatalError(
                f"Could not connect to {self._base_url} — check that your Base URL is reachable."
            ) from e

        except AUTH_ERRORS as e:
            raise ProviderFatalError("Authentication failed — check that your API key is correct.") from e

        except PERMISSION_ERRORS as e:
            raise ProviderFatalError(
                "Access denied — your API key may not have permission for this model or endpoint."
            ) from e

        except NOT_FOUND_ERRORS as e:
            raise ProviderFatalError(
                f"Endpoint or model not found — check your Base URL and Model name. ({self._base_url}, {self._model})"
            ) from e

        except BAD_REQUEST_ERRORS as e:
            raise bad_request_error from e

        self._connection_ok = True

        text, used_input, used_output = self._read_response(response)
        with self._token_lock:
            self.input_tokens += used_input
            self.output_tokens += used_output

        return text

