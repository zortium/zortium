from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty


class ProviderFatalError(Exception):
    """
    Raised by a VLMProvider when an error is unrecoverable for the current
    scan — wrong API key, wrong URL, model not found, etc.  The runner
    propagates this upward instead of silently marking test cases as passed.
    The message is user-facing: keep it concise and actionable.
    """


class ProviderSkippedError(Exception):
    """
    Raised by a VLMProvider when a single test case cannot be completed but
    the scan should continue — e.g. rate limit hit.  The runner records a
    skipped result (metadata role='skipped') excluded from ASR calculation.
    """


class VLMProvider(ABC):
    """
    Base class for all VLM provider adapters.
    Normalizes the interface for sending an image + text prompt
    to any model, regardless of the underlying API.
    """

    @abstractproperty
    def model(self) -> str:
        """Human-readable model identifier e.g. 'gpt-4o'"""
        ...

    @abstractmethod
    def send(
        self,
        image: bytes | None,
        prompt: str,
        schema: dict | None = None,
        *,
        schema_name: str = "zortium_schema",
    ) -> str:
        """
        Send an optional image and text prompt to the model.
        If schema is provided, the provider must use structured/constrained output mode.
        Raise ProviderSkippedError("structured_output_unsupported") if the endpoint does not support it.
        """
        ...
