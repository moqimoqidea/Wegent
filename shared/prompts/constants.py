# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt constants used across backend and chat_shell modules."""

import json
from typing import Any

# Marker that separates attachment/KB context from the actual user question.
# Used when context (attachments, documents, knowledge base) is prepended to
# a user message as a single string.  Both the producers (context preprocessing)
# and consumers (history loader, image/video agents) must use the same marker
# to ensure prefix-cache consistency across conversation turns.
USER_QUESTION_MARKER = "[User Question]:"


def parse_prompt_blocks(raw_prompt: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse a stored prompt value into its text content and extra blocks.

    When ``inject_datetime=True`` the user message is stored as a JSON array
    of content blocks::

        [{"type": "text", "text": "<user text>"},
         {"type": "text", "text": "<system-reminder>..."}]

    This helper extracts the first ``type=text`` block as the primary text
    content, and collects the remaining text blocks (e.g. system-reminder)
    into *extra_blocks*.  ``image_url`` blocks are deliberately skipped
    because image data lives in ``SubtaskContext``.

    For plain-text prompts the original string is returned with an empty
    extra-blocks list.

    Args:
        raw_prompt: Raw ``Subtask.prompt`` value from the database.

    Returns:
        A tuple of ``(text_content, extra_blocks)``.
    """
    try:
        parsed = json.loads(raw_prompt)
        if isinstance(parsed, list) and all(isinstance(b, dict) for b in parsed):
            text_content = raw_prompt  # fallback if no text block found
            extra_blocks: list[dict[str, Any]] = []
            first_text_found = False
            # Support both LangChain format ("text") and Responses API format
            # ("input_text").  The first text block is the user's message; the
            # rest (e.g. system-reminder) are extra blocks.  Non-text blocks
            # (image_url, input_image) are deliberately skipped.
            _TEXT_TYPES = {"text", "input_text"}
            for block in parsed:
                if block.get("type") in _TEXT_TYPES:
                    if not first_text_found:
                        text_content = block.get("text", "")
                        first_text_found = True
                    else:
                        extra_blocks.append(block)
            return text_content, extra_blocks
    except (json.JSONDecodeError, ValueError):
        pass

    return raw_prompt, []


def extract_user_question(text: str) -> str:
    """Extract user-visible question from a context-wrapped prompt.

    The chat context preprocessor may build prompts like::

        "<attachment>...metadata...</attachment>\\n\\n[User Question]:\\n<message>"

    This function splits on the ``USER_QUESTION_MARKER`` and returns only the
    user's own text, stripping surrounding whitespace.  If the marker is absent
    the full text is returned as-is (stripped).

    Args:
        text: Raw prompt string, possibly containing context + marker + question.

    Returns:
        The user question portion of the prompt.
    """
    if not isinstance(text, str):
        return str(text)

    if USER_QUESTION_MARKER in text:
        after = text.split(USER_QUESTION_MARKER, 1)[1]
        return after.lstrip("\n").strip()

    return text.strip()
