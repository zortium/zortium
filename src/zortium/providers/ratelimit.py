from __future__ import annotations

import time
import random
from typing import TypeVar
from datetime import datetime, timezone
from collections.abc import Mapping, Callable
from email.utils import parsedate_to_datetime

from openai import RateLimitError, APIConnectionError

from zortium.logging import get_logger
from zortium.constants import RateLimitPolicy
from zortium.providers.base import ProviderSkippedError

logger = get_logger("ratelimit")

T = TypeVar("T")

RETRY_AFTER_HEADER = "retry-after"
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0
JITTER_SECONDS = 1.0
MAX_RETRIES = 8
MAX_TOTAL_WAIT_SECONDS = 600.0
CONNECT_MAX_RETRIES = 3
CONNECT_BACKOFF_SECONDS = 1.0
SKIP_MESSAGE = "Rate limit reached — this test case was skipped."
CAP_MESSAGE = "Rate limit reached — wait cap exceeded, test case skipped."


class RateLimitResolver:
    """
    Computes how long to wait after a 429 (honoring the Retry-After header,
    falling back to capped exponential backoff + jitter) and runs the retry
    loop under a given RateLimitPolicy. Stateless — safe to share one instance.
    """

    @staticmethod
    def __parse_retry_after(value: str) -> float | None:
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return float(raw)
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

    @staticmethod
    def __backoff(attempt: int) -> float:
        return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt))

    @staticmethod
    def __headers_of(error: RateLimitError) -> Mapping[str, str]:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        return headers or {}

    @staticmethod
    def __attempt(make_call: Callable[[], T], sleep: Callable[[float], None]) -> T:
        # A connection reset/timeout is usually transient (keep-alive pool reuse,
        # a brief blip). The OpenAI SDK would normally retry it, but we run with
        # max_retries=0 to own rate-limit handling — so retry it here instead.
        attempt = 0
        while True:
            try:
                return make_call()
            except APIConnectionError:
                attempt += 1
                if attempt >= CONNECT_MAX_RETRIES:
                    raise
                wait = CONNECT_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"connection error (attempt {attempt}); retrying in {wait:.1f}s")
                sleep(wait)

    def resolve_wait_seconds(self, headers: Mapping[str, str], attempt: int) -> float:
        header_value = headers.get(RETRY_AFTER_HEADER) if headers else None
        base = None
        if header_value is not None:
            base = self.__parse_retry_after(str(header_value))
        if base is None:
            base = self.__backoff(attempt)
        return base + random.uniform(0.0, JITTER_SECONDS)

    def execute_with_retry(
        self,
        make_call: Callable[[], T],
        *,
        policy: RateLimitPolicy,
        max_retries: int = MAX_RETRIES,
        max_total_wait: float = MAX_TOTAL_WAIT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> T:
        if policy != RateLimitPolicy.WAIT:
            try:
                return self.__attempt(make_call, sleep)
            except RateLimitError as e:
                logger.warning("rate limit reached; skipping test case")
                raise ProviderSkippedError(SKIP_MESSAGE) from e

        elapsed = 0.0
        last_error: RateLimitError | None = None
        for attempt in range(max_retries):
            try:
                return self.__attempt(make_call, sleep)
            except RateLimitError as e:
                last_error = e
                if attempt + 1 >= max_retries or elapsed >= max_total_wait:
                    break
                wait = self.resolve_wait_seconds(self.__headers_of(e), attempt)
                wait = min(wait, max_total_wait - elapsed)
                logger.warning(f"rate limited (attempt {attempt + 1}); waiting {wait:.1f}s before retry")
                sleep(wait)
                elapsed += wait

        logger.warning(f"rate limit persisted past cap ({max_retries} attempts, {elapsed:.0f}s); skipping test case")
        raise ProviderSkippedError(CAP_MESSAGE) from last_error
