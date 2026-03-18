# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ChatOpenAI subclass that preserves reasoning_content through tool call loops.

Some OpenAI-compatible reasoning models (e.g., kimi-k2.5, DeepSeek R1) return
``reasoning_content`` alongside regular content and tool calls. The upstream
``langchain-openai`` library does not capture or re-inject this field, causing
errors like:

    "thinking is enabled but reasoning_content is missing in assistant tool call message"

This module provides ``ReasoningAwareChatOpenAI`` which:
1. Captures ``reasoning_content`` from streaming deltas into ``additional_kwargs``
2. Captures ``reasoning_content`` from non-streaming responses
3. Re-injects ``reasoning_content`` into the request payload for the next LLM call
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_REASONING_KEY = "reasoning_content"


class ReasoningAwareChatOpenAI(ChatOpenAI):
    """ChatOpenAI that preserves ``reasoning_content`` across multi-turn tool loops."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None:
            return None

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get(_REASONING_KEY)
            if reasoning:
                gen_chunk.message.additional_kwargs[_REASONING_KEY] = reasoning

        return gen_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        choices = response_dict.get("choices") or []
        for i, choice in enumerate(choices):
            msg_dict = choice.get("message") or {}
            reasoning = msg_dict.get(_REASONING_KEY)
            if reasoning and i < len(result.generations):
                gen = result.generations[i]
                if isinstance(gen.message, AIMessage):
                    gen.message.additional_kwargs[_REASONING_KEY] = reasoning

        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "messages" not in payload:
            return payload

        lc_messages = self._convert_input(input_).to_messages()
        msg_dicts = payload["messages"]
        if len(lc_messages) != len(msg_dicts):
            return payload

        for lc_msg, msg_dict in zip(lc_messages, msg_dicts):
            if not isinstance(lc_msg, AIMessage):
                continue
            reasoning = lc_msg.additional_kwargs.get(_REASONING_KEY)
            if reasoning:
                msg_dict[_REASONING_KEY] = reasoning

        return payload
