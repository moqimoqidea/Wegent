# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt constants used across backend and chat_shell modules."""

# Marker that separates attachment/KB context from the actual user question.
# Used when context (attachments, documents, knowledge base) is prepended to
# a user message as a single string.  Both the producers (context preprocessing)
# and consumers (history loader, image/video agents) must use the same marker
# to ensure prefix-cache consistency across conversation turns.
USER_QUESTION_MARKER = "[User Question]:"


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
