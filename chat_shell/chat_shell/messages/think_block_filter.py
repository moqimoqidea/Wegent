"""Filter provider-specific reasoning blocks from conversation history.

When users switch models mid-conversation (e.g. Claude → GPT → Gemini),
historical messages may contain think/reasoning blocks in formats that the
new model's API rejects.  This module provides a single public function,
:func:`strip_foreign_reasoning_blocks`, which removes or retains reasoning
blocks based on whether the message originated from the same provider as
the current request target.
"""

from __future__ import annotations

import copy
from typing import Any

# Canonical block type used for all normalized reasoning content.
_REASONING_TYPE = "reasoning"

# Provider-specific block types found in legacy (pre-normalization) data.
_LEGACY_ANTHROPIC_TYPE = "thinking"


def _infer_provider(msg: dict[str, Any]) -> str | None:
    """Heuristically infer the originating provider from a message dict.

    Used for legacy messages that lack a ``model_info`` field.

    Returns:
        Provider string (``"anthropic"``, ``"openai"``) or ``None`` if
        no think blocks are present and provider cannot be determined.
    """
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == _LEGACY_ANTHROPIC_TYPE:
                return "anthropic"
            if block_type == _REASONING_TYPE and "summary" in block:
                # OpenAI Responses API format (pre-normalization legacy)
                return "openai"
            if (
                block_type == _REASONING_TYPE
                and isinstance(block.get("extras"), dict)
                and block["extras"].get("signature")
            ):
                # Canonical reasoning with Anthropic signature
                return "anthropic"

    # DeepSeek/Kimi: reasoning_content in additional_kwargs
    additional_kwargs = msg.get("additional_kwargs")
    if isinstance(additional_kwargs, dict) and additional_kwargs.get(
        "reasoning_content"
    ):
        return "openai"

    return None


def _strip_reasoning_from_content(content: list) -> list:
    """Remove all reasoning blocks from a content block list.

    Returns:
        The filtered list, or a single empty text block if all were removed.
    """
    filtered = [
        block
        for block in content
        if not (
            isinstance(block, dict)
            and block.get("type") in (_REASONING_TYPE, _LEGACY_ANTHROPIC_TYPE)
        )
    ]
    if not filtered:
        return [{"type": "text", "text": ""}]
    return filtered


def _denormalize_for_anthropic(content: list) -> list:
    """Convert canonical reasoning blocks back to Anthropic native thinking format.

    Transforms ``{"type": "reasoning", "reasoning": "...", "extras": {"signature": "..."}}``
    back to ``{"type": "thinking", "thinking": "...", "signature": "..."}``.

    Non-reasoning blocks are passed through unchanged.
    """
    result: list = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != _REASONING_TYPE:
            result.append(block)
            continue

        thinking_block: dict[str, Any] = {
            "type": "thinking",
            "thinking": block.get("reasoning", ""),
        }
        # Promote extras entries (e.g. signature) to top-level
        extras = block.get("extras")
        if isinstance(extras, dict):
            for k, v in extras.items():
                thinking_block[k] = v
        result.append(thinking_block)
    return result


def _denormalize_for_openai_responses(content: list) -> list:
    """Convert canonical reasoning blocks back to OpenAI Responses API format.

    Transforms exploded canonical blocks::

        {"type": "reasoning", "reasoning": "text",
         "extras": {"id": "rs_...", "encrypted_content": "gAAAA...", ...}}

    back to the original Responses API structure::

        {"type": "reasoning", "id": "rs_...",
         "summary": [{"type": "summary_text", "text": "text"}],
         "encrypted_content": "gAAAA..."}

    Blocks without ``extras.id`` or ``extras.encrypted_content`` (i.e. not
    originating from the Responses API) are passed through unchanged.
    """
    result: list = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != _REASONING_TYPE:
            result.append(block)
            continue

        extras = block.get("extras")
        if not isinstance(extras, dict) or (
            "id" not in extras and "encrypted_content" not in extras
        ):
            # Not an exploded Responses API reasoning block
            result.append(block)
            continue

        # Reconstruct the original Responses API format
        rebuilt: dict[str, Any] = {"type": _REASONING_TYPE}
        if "id" in extras:
            rebuilt["id"] = extras["id"]

        reasoning_text = block.get("reasoning", "")
        rebuilt["summary"] = [{"type": "summary_text", "text": reasoning_text}]

        if "encrypted_content" in extras:
            rebuilt["encrypted_content"] = extras["encrypted_content"]

        result.append(rebuilt)
    return result


def strip_foreign_reasoning_blocks(
    messages: list[dict[str, Any]],
    target_provider: str,
) -> list[dict[str, Any]]:
    """Remove reasoning blocks from messages produced by a different provider.

    For **same-provider** messages, reasoning blocks (including ``extras``
    like ``signature``) are preserved to maintain multi-turn reasoning
    continuity.

    For **cross-provider** messages, reasoning blocks are stripped because:

    1. Each provider's API rejects foreign think block types.
    2. Provider-specific data (``signature``, ``encrypted_content``) is
       meaningless cross-provider.
    3. DeepSeek/Kimi explicitly forbid sending ``reasoning_content`` back.

    For legacy messages without ``model_info``, the provider is inferred
    heuristically from the content block types.

    Args:
        messages: Conversation history as a list of message dicts.
        target_provider: The provider of the current request
            (e.g. ``"anthropic"``, ``"openai"``, ``"google"``).

    Returns:
        A new list of message dicts with foreign reasoning blocks removed.
        Original dicts are not mutated.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        # Determine source provider
        model_info = msg.get("model_info")
        if isinstance(model_info, dict):
            source_provider = model_info.get("provider", "")
        else:
            source_provider = _infer_provider(msg) or ""

        content = msg.get("content")

        # Same provider: keep everything (denormalize to native format)
        if source_provider == target_provider:
            if target_provider == "anthropic" and isinstance(content, list):
                denormalized = copy.deepcopy(msg)
                denormalized["content"] = _denormalize_for_anthropic(content)
                # Inject response_metadata as a top-level key so
                # convert_to_messages passes it to AIMessage.response_metadata.
                # LangChain Anthropic's _format_messages checks this to keep
                # thinking blocks from the same provider.
                denormalized["response_metadata"] = {"model_provider": "anthropic"}
                result.append(denormalized)
            elif target_provider == "openai" and isinstance(content, list):
                denormalized = copy.deepcopy(msg)
                denormalized["content"] = _denormalize_for_openai_responses(content)
                result.append(denormalized)
            else:
                result.append(msg)
            continue

        # Different provider (or unknown): strip reasoning blocks
        needs_strip = False

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    _REASONING_TYPE,
                    _LEGACY_ANTHROPIC_TYPE,
                ):
                    needs_strip = True
                    break

        # Also check legacy additional_kwargs.reasoning_content
        additional_kwargs = msg.get("additional_kwargs")
        has_legacy_reasoning = isinstance(additional_kwargs, dict) and bool(
            additional_kwargs.get("reasoning_content")
        )

        if not needs_strip and not has_legacy_reasoning:
            result.append(msg)
            continue

        # Create a deep copy and strip reasoning
        stripped = copy.deepcopy(msg)

        if isinstance(content, list) and needs_strip:
            stripped["content"] = _strip_reasoning_from_content(content)

        if has_legacy_reasoning:
            new_kwargs = dict(additional_kwargs)  # type: ignore[arg-type]
            del new_kwargs["reasoning_content"]
            if new_kwargs:
                stripped["additional_kwargs"] = new_kwargs
            else:
                stripped.pop("additional_kwargs", None)

        result.append(stripped)

    return result
