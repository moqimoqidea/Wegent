# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for think_block_filter module."""

import pytest

from chat_shell.messages.think_block_filter import (
    strip_foreign_reasoning_blocks,
)


class TestStripForeignReasoningBlocks:
    """Tests for strip_foreign_reasoning_blocks."""

    def test_same_provider_preserves_reasoning(self):
        """Reasoning blocks from the same provider are kept."""
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
        assert result[0]["content"] == messages[0]["content"]

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

    def test_all_reasoning_stripped_becomes_empty_string(self):
        """If all content blocks are reasoning, content becomes empty string."""
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
        assert result[0]["content"] == ""

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
        """Mixed provider history: only cross-provider reasoning is stripped."""
        messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "claude thought"},
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
