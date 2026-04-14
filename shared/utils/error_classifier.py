# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Error classification utility for chat errors.

Classifies raw LLM SDK exceptions and error strings into structured
error codes that the frontend can use to display user-friendly messages
and actionable solutions.
"""

from enum import Enum
from typing import Union


class ChatErrorCode(str, Enum):
    """Structured error codes for chat errors."""

    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMIT = "rate_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONTAINER_OOM = "container_oom"
    CONTAINER_ERROR = "container_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    LLM_UNSUPPORTED = "llm_unsupported"
    FORBIDDEN = "forbidden"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    INVALID_PARAMETER = "invalid_parameter"
    GENERIC_ERROR = "generic_error"


# Keyword patterns for string-based classification.
# Order matters: more specific patterns are checked first.
_CLASSIFICATION_RULES: list[tuple[ChatErrorCode, list[str]]] = [
    # Context length exceeded
    (
        ChatErrorCode.CONTEXT_LENGTH_EXCEEDED,
        [
            "prompt is too long",
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "max_tokens",
            "token limit exceeded",
            "tokens exceeds the model",
            "input is too long",
            "request too large",
            "maximum number of tokens",
        ],
    ),
    # Container OOM
    (
        ChatErrorCode.CONTAINER_OOM,
        [
            "out of memory",
            "oom killed",
            "memory allocation",
        ],
    ),
    # Container errors
    (
        ChatErrorCode.CONTAINER_ERROR,
        [
            "container",
            "executor",
            "docker",
            "disappeared unexpectedly",
            "no ports mapped",
            "crashed unexpectedly",
            "exit code",
        ],
    ),
    # Quota exceeded (check before rate_limit — more specific)
    (
        ChatErrorCode.QUOTA_EXCEEDED,
        [
            "quota exceeded",
            "insufficient_quota",
            "billing",
            "credit balance",
            "payment required",
            "account balance",
            "insufficient funds",
            "exceeded your current quota",
        ],
    ),
    # Rate limit (temporary throttling)
    (
        ChatErrorCode.RATE_LIMIT,
        [
            "rate limit",
            "rate_limit",
            "too many requests",
            "throttl",
        ],
    ),
    # Forbidden / auth errors
    (
        ChatErrorCode.FORBIDDEN,
        [
            "forbidden",
            "not allowed",
            "unauthorized",
            "403",
        ],
    ),
    # Model unsupported (multi-modal, incompatible request)
    (
        ChatErrorCode.LLM_UNSUPPORTED,
        [
            "multi-modal",
            "multimodal",
            "do not support",
            "does not support",
            "not support image",
        ],
    ),
    # Model unavailable
    (
        ChatErrorCode.MODEL_UNAVAILABLE,
        [
            "model not found",
            "model unavailable",
            "model_not_found",
            "model error",
            "overloaded",
            "llm request failed",
            "llm api error",
            "llm call failed",
            "llm service error",
        ],
    ),
    # Invalid parameter
    (
        ChatErrorCode.INVALID_PARAMETER,
        [
            "invalid parameter",
            "invalid_parameter",
        ],
    ),
    # Payload too large
    (
        ChatErrorCode.PAYLOAD_TOO_LARGE,
        [
            "413",
            "payload too large",
        ],
    ),
    # Network errors
    (
        ChatErrorCode.NETWORK_ERROR,
        [
            "network",
            "connection refused",
            "connection reset",
            "connection error",
            "not connected",
        ],
    ),
    # Timeout
    (
        ChatErrorCode.TIMEOUT_ERROR,
        [
            "timeout",
            "timed out",
        ],
    ),
]

# Map known SDK exception class names to error codes.
# Uses class name strings to avoid hard dependencies on SDK packages.
_EXCEPTION_CLASS_MAP: dict[str, ChatErrorCode] = {
    # OpenAI SDK
    "RateLimitError": ChatErrorCode.RATE_LIMIT,
    "AuthenticationError": ChatErrorCode.FORBIDDEN,
    "PermissionDeniedError": ChatErrorCode.FORBIDDEN,
    "NotFoundError": ChatErrorCode.MODEL_UNAVAILABLE,
    "InternalServerError": ChatErrorCode.MODEL_UNAVAILABLE,
    # Anthropic SDK
    "RateLimitError": ChatErrorCode.RATE_LIMIT,
    "AuthenticationError": ChatErrorCode.FORBIDDEN,
    "PermissionDeniedError": ChatErrorCode.FORBIDDEN,
    "NotFoundError": ChatErrorCode.MODEL_UNAVAILABLE,
    "OverloadedError": ChatErrorCode.MODEL_UNAVAILABLE,
    # Google SDK
    "ResourceExhausted": ChatErrorCode.RATE_LIMIT,
    "PermissionDenied": ChatErrorCode.FORBIDDEN,
    "NotFound": ChatErrorCode.MODEL_UNAVAILABLE,
}


def _classify_by_exception_type(error: Exception) -> ChatErrorCode | None:
    """Classify error by exception class hierarchy."""
    class_name = type(error).__name__
    code = _EXCEPTION_CLASS_MAP.get(class_name)
    if code is not None:
        # For BadRequestError, refine based on message content
        if class_name == "BadRequestError":
            return _classify_by_message(str(error))
        return code
    return None


def _classify_by_message(message: str) -> ChatErrorCode:
    """Classify error by keyword matching on the message string."""
    lower = message.lower()
    for code, patterns in _CLASSIFICATION_RULES:
        for pattern in patterns:
            if pattern in lower:
                return code
    return ChatErrorCode.GENERIC_ERROR


def classify_error(error: Union[Exception, str]) -> str:
    """Classify an error into a structured error code.

    Checks exception class hierarchy first (for typed SDK exceptions),
    then falls back to keyword-based string matching.

    Args:
        error: Exception instance or error message string.

    Returns:
        Error code string (ChatErrorCode value).
    """
    if isinstance(error, Exception):
        # Try exception class-based classification first
        code = _classify_by_exception_type(error)
        if code is not None:
            return code.value
        # Fall back to message-based classification
        return _classify_by_message(str(error)).value

    # String input — keyword matching only
    return _classify_by_message(error).value
