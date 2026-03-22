# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ChatOpenAI subclass that captures reasoning_content from OpenAI-compatible APIs.

Some OpenAI-compatible providers (e.g. Kimi, DeepSeek) return a non-standard
``reasoning_content`` field in streaming deltas. The default ChatOpenAI drops
this field during chunk conversion. This module provides a thin subclass that
post-processes each chunk to preserve reasoning_content in
``additional_kwargs``, making it available to downstream consumers like
graph_builder's ``stream_tokens``.

Usage:
    Use ``ChatOpenAIWithReasoning`` as a drop-in replacement for ``ChatOpenAI``
    when think_config is present for OpenAI-compatible providers.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import ChatGenerationChunk

logger = logging.getLogger(__name__)


class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI variant that extracts reasoning_content from streaming deltas.

    Overrides ``_convert_chunk_to_generation_chunk`` to capture the
    ``reasoning_content`` field from the raw API response delta and inject it
    into the AIMessageChunk's ``additional_kwargs``.  All other behavior is
    identical to the base ``ChatOpenAI``.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None

        # Extract reasoning_content from the raw delta
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content and isinstance(
                generation_chunk.message, AIMessageChunk
            ):
                generation_chunk.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )

        return generation_chunk
