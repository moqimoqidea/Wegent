# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for normalising user prompt values for display.

When chat mode sends a message with `inject_datetime=True`, the user message is
stored in ``Subtask.prompt`` as a JSON array of content blocks:

    [{"type": "text", "text": "<user text>"}, {"type": "text", "text": "<system-remember>..."}]

The system-remember block carries the current time for the LLM but must not be
shown to users in the chat UI.  This module provides a single helper that strips
internal blocks and returns only the human-visible portion of the prompt.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_display_prompt(prompt: Optional[str]) -> Optional[str]:
    """Return the user-visible text from a stored prompt value.

    * Plain-text prompts are returned as-is.
    * JSON-array prompts (multi-block format) return only the text of the first
      ``type=text`` block, which is always the user's original message.

    Args:
        prompt: Raw ``Subtask.prompt`` value from the database.

    Returns:
        The human-readable text, or ``None`` if the input is ``None`` / empty.
    """
    if not prompt:
        return prompt

    # Fast path: if the string doesn't start with '[' it's definitely plain text
    stripped = prompt.lstrip()
    if not stripped.startswith("["):
        return prompt

    try:
        parsed = json.loads(prompt)
        if isinstance(parsed, list) and all(isinstance(b, dict) for b in parsed):
            for block in parsed:
                if block.get("type") == "text":
                    return block.get("text", "")
    except (json.JSONDecodeError, ValueError):
        pass

    return prompt
