# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for think_block_filter module."""

import pytest

from chat_shell.messages.think_block_filter import (
    _infer_provider,
    strip_foreign_reasoning_blocks,
)


class TestStripForeignReasoningBlocks:
    """Tests for strip_foreign_reasoning_blocks."""

    def test_same_provider_preserves_reasoning(self):
        """Anthropic same-provider reasoning blocks are denormalized to thinking format."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "thinking...",
                        "extras": {"signature": "abc"},
                    },
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude-sonnet"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        content = result[0]["content"]
        assert content[0] == {
            "type": "thinking",
            "thinking": "thinking...",
            "signature": "abc",
        }
        assert content[1] == {"type": "text", "text": "answer"}

    def test_cross_provider_strips_reasoning(self):
        """Reasoning blocks from a different provider are removed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "thinking..."},
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude-sonnet"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[0]["content"] == [{"type": "text", "text": "answer"}]

    def test_user_messages_untouched(self):
        """Non-assistant messages are passed through without modification."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are a bot"},
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result == messages

    def test_no_think_blocks_untouched(self):
        """Messages without reasoning blocks are passed through."""
        messages = [
            {
                "role": "assistant",
                "content": "plain text",
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result == messages

    def test_all_reasoning_stripped_becomes_empty_text_block(self):
        """If all content blocks are reasoning, content becomes an empty text block."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "only thinking"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[0]["content"] == [{"type": "text", "text": ""}]

    def test_legacy_anthropic_thinking_blocks_inferred(self):
        """Legacy Claude thinking blocks (no model_info) are detected heuristically."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "old format"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
        # Same provider (anthropic inferred) -> keep
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        assert len(result[0]["content"]) == 2

        # Different provider -> strip
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[0]["content"] == [{"type": "text", "text": "answer"}]

    def test_legacy_reasoning_content_in_additional_kwargs(self):
        """Legacy additional_kwargs.reasoning_content is detected and stripped."""
        messages = [
            {
                "role": "assistant",
                "content": "answer",
                "additional_kwargs": {"reasoning_content": "deep thinking"},
            },
        ]
        # Inferred as openai -> same provider keeps it
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert "additional_kwargs" in result[0]

        # Different provider -> strip
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        assert "additional_kwargs" not in result[0]

    def test_mixed_history_selective_stripping(self):
        """Mixed provider history: cross-provider reasoning stripped, same-provider denormalized."""
        messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "claude thought",
                        "extras": {"signature": "sig1"},
                    },
                    {"type": "text", "text": "A1"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
            {"role": "user", "content": "Q2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "gpt thought"},
                    {"type": "text", "text": "A2"},
                ],
                "model_info": {"provider": "openai", "model": "gpt-5"},
            },
        ]
        # Target is openai: claude reasoning stripped, gpt reasoning kept
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[1]["content"] == [{"type": "text", "text": "A1"}]
        assert len(result[3]["content"]) == 2  # both blocks preserved

        # Target is anthropic: claude reasoning denormalized to thinking, gpt stripped
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        assert result[1]["content"][0] == {
            "type": "thinking",
            "thinking": "claude thought",
            "signature": "sig1",
        }
        assert result[3]["content"] == [{"type": "text", "text": "A2"}]

    def test_original_messages_not_mutated(self):
        """The original message dicts are not modified in-place."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "thinking"},
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
        ]
        original_content = list(messages[0]["content"])
        strip_foreign_reasoning_blocks(messages, "openai")
        assert messages[0]["content"] == original_content

    def test_unknown_provider_no_model_info_no_reasoning(self):
        """Messages without model_info and without reasoning are passed through."""
        messages = [
            {"role": "assistant", "content": "plain answer"},
        ]
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        assert result == messages

    def test_anthropic_same_provider_sets_response_metadata(self):
        """Denormalized Anthropic messages include response_metadata for LangChain."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "thought",
                        "extras": {"signature": "sig"},
                    },
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "anthropic")
        # response_metadata is injected as top-level key for convert_to_messages
        assert result[0]["response_metadata"] == {"model_provider": "anthropic"}

    def test_anthropic_same_provider_does_not_mutate_original(self):
        """Denormalization creates a deep copy, original is untouched."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "thought",
                        "extras": {"signature": "sig"},
                    },
                ],
                "model_info": {"provider": "anthropic", "model": "claude"},
            },
        ]
        original_content = list(messages[0]["content"])
        strip_foreign_reasoning_blocks(messages, "anthropic")
        assert messages[0]["content"] == original_content
        assert "response_metadata" not in messages[0]

    def test_non_anthropic_same_provider_passes_through(self):
        """Non-Anthropic same-provider messages without Responses API extras are not denormalized."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "thinking"},
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "openai", "model": "gpt-5"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        # Plain reasoning blocks (no extras.id) pass through unchanged
        assert result[0]["content"][0]["type"] == "reasoning"
        assert result[0]["content"][0].get("reasoning") == "thinking"
        assert "additional_kwargs" not in result[0]

    def test_openai_same_provider_denormalizes_reasoning(self):
        """OpenAI same-provider reasoning blocks with Responses API extras are reconstructed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "I considered the options...",
                        "extras": {
                            "id": "rs_abc123",
                            "encrypted_content": "gAAAA_encrypted_data",
                            "index": 0,
                        },
                    },
                    {
                        "id": "msg_xyz789",
                        "type": "text",
                        "text": "Final answer",
                        "index": 1,
                    },
                ],
                "model_info": {"provider": "openai", "model": "gpt-5.4"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        reasoning_block = result[0]["content"][0]
        assert reasoning_block == {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [{"type": "summary_text", "text": "I considered the options..."}],
            "encrypted_content": "gAAAA_encrypted_data",
        }
        # Text block unchanged
        assert result[0]["content"][1]["type"] == "text"

    def test_openai_same_provider_without_extras_id_passes_through(self):
        """OpenAI reasoning blocks without extras.id/encrypted_content are not reconstructed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "plain thinking"},
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "openai", "model": "gpt-5"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[0]["content"][0] == {
            "type": "reasoning",
            "reasoning": "plain thinking",
        }

    def test_openai_same_provider_does_not_mutate_original(self):
        """OpenAI denormalization creates a deep copy, original is untouched."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "thought",
                        "extras": {"id": "rs_1", "encrypted_content": "enc"},
                    },
                ],
                "model_info": {"provider": "openai", "model": "gpt-5"},
            },
        ]
        original_content = list(messages[0]["content"])
        strip_foreign_reasoning_blocks(messages, "openai")
        assert messages[0]["content"] == original_content

    def test_openai_same_provider_multiple_reasoning_blocks(self):
        """Multiple exploded reasoning blocks are each reconstructed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "Step 1",
                        "extras": {"id": "rs_1", "encrypted_content": "enc1"},
                    },
                    {
                        "type": "reasoning",
                        "reasoning": "Step 2",
                        "extras": {"id": "rs_1", "encrypted_content": "enc1"},
                    },
                    {"type": "text", "text": "answer"},
                ],
                "model_info": {"provider": "openai", "model": "gpt-5"},
            },
        ]
        result = strip_foreign_reasoning_blocks(messages, "openai")
        assert result[0]["content"][0] == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Step 1"}],
            "encrypted_content": "enc1",
        }
        assert result[0]["content"][1] == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Step 2"}],
            "encrypted_content": "enc1",
        }
        assert result[0]["content"][2] == {"type": "text", "text": "answer"}


class TestInferProvider:
    """Tests for _infer_provider heuristic."""

    def test_legacy_thinking_block_returns_anthropic(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "..."}],
        }
        assert _infer_provider(msg) == "anthropic"

    def test_canonical_reasoning_with_signature_returns_anthropic(self):
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "reasoning": "...",
                    "extras": {"signature": "abc"},
                }
            ],
        }
        assert _infer_provider(msg) == "anthropic"

    def test_reasoning_with_summary_returns_openai(self):
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "x"}],
                }
            ],
        }
        assert _infer_provider(msg) == "openai"

    def test_canonical_reasoning_without_extras_returns_none(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "reasoning", "reasoning": "..."}],
        }
        assert _infer_provider(msg) is None

    def test_additional_kwargs_reasoning_content_returns_openai(self):
        msg = {
            "role": "assistant",
            "content": "answer",
            "additional_kwargs": {"reasoning_content": "deep thinking"},
        }
        assert _infer_provider(msg) == "openai"

    def test_plain_text_returns_none(self):
        msg = {"role": "assistant", "content": "plain answer"}
        assert _infer_provider(msg) is None
