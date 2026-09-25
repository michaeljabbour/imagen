"""Custom exception hierarchy for imagen.

Provides structured, user-safe error handling with provider context
and MCP error code mapping.
"""

from __future__ import annotations

import re


def _sanitize_message(msg: str) -> str:
    """Remove potential credentials from error messages."""
    # OpenAI keys have changed shape over time (for example ``sk-proj-...``),
    # so redact the stable prefix even when a truncated key appears in an
    # upstream exception.
    msg = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", msg)
    # Strip long alphanumeric strings that could be API keys
    msg = re.sub(r"[A-Za-z0-9_-]{32,}", "[REDACTED]", msg)
    # Strip URL query params that might contain keys
    msg = re.sub(r"key=[^&\s]+", "key=[REDACTED]", msg)
    return msg


class ImagenError(Exception):
    """Base exception for imagen."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or _sanitize_message(str(message))


class ConfigurationError(ImagenError):
    """Missing API key, invalid settings, etc."""


class ProviderError(ImagenError):
    """API call failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message, user_message=user_message)
        self.provider = provider
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class AuthenticationError(ProviderError):
    """Invalid or expired API key (401/403)."""


class RateLimitError(ProviderError):
    """Rate limit exceeded (429). Includes retry_after if available."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str,
        status_code: int | None = None,
        request_id: str | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            status_code=status_code,
            code="rate_limit_exceeded",
            request_id=request_id,
            retryable=True,
            user_message=user_message or "Rate limit exceeded. Please wait and try again.",
        )
        self.retry_after = retry_after


class GenerationError(ProviderError):
    """Model-level failure (content filter, invalid prompt, etc.)."""


class ValidationError(ImagenError):
    """Invalid prompt, bad parameters, unsupported size, etc."""
