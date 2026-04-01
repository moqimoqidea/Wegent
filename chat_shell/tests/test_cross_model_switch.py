# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for cross-model switching with think block compatibility.

Simulates the full serialize → store → load → filter → convert cycle when
switching between different LLM providers mid-conversation.
"""

from langchain_core.messages import AIMessage, ToolMessage

from chat_shell.agents.graph_builder import (
    _convert_validated_messages,
    _serialize_messages_chain,
)


class TestCrossModelSwitch:
    """End-to-end tests for cross-model think block round-trips."""

    def test_claude_thinking_to_gpt(self):
        """Claude thinking blocks are stripped when loading for GPT."""
        # Step 1: Simulate Claude response with thinking
        claude_msg = AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "Let me analyze...",
                    "signature": "sig123",
                },
                {"type": "text", "text": "The answer is 42"},
            ]
        )
        chain = _serialize_messages_chain(
            [claude_msg], provider="anthropic", model_id="claude-sonnet"
        )

        # Step 2: Simulate history loading for GPT
        history = [{"role": "user", "content": "What is 6*7?"}] + chain
        lc_messages = _convert_validated_messages(
            history, context="test", target_provider="openai"
        )

        # Verify: no thinking blocks in the converted messages
        for msg in lc_messages:
            if hasattr(msg, "content") and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        assert (
                            block.get("type") != "thinking"
                        ), "thinking block should be stripped for openai"
                        # Canonical reasoning blocks should also be stripped
                        assert (
                            block.get("type") != "reasoning"
                        ), "reasoning block should be stripped for cross-provider"

    def test_deepseek_reasoning_to_claude(self):
        """DeepSeek reasoning_content is stripped when loading for Claude."""
        deepseek_msg = AIMessage(
            content="The answer is 42",
            additional_kwargs={"reasoning_content": "Step 1: multiply..."},
        )
        chain = _serialize_messages_chain(
            [deepseek_msg], provider="openai", model_id="deepseek-r1"
        )

        history = [{"role": "user", "content": "What is 6*7?"}] + chain
        lc_messages = _convert_validated_messages(
            history, context="test", target_provider="anthropic"
        )

        # Verify: no reasoning blocks in the anthropic-targeted messages
        for msg in lc_messages:
            if hasattr(msg, "content") and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        assert block.get("type") != "reasoning"

    def test_openai_responses_reasoning_to_gemini(self):
        """OpenAI Responses API reasoning summary is stripped for Gemini."""
        oai_msg = AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "I considered..."}],
                    "id": "rs_abc",
                },
                {"type": "text", "text": "Final answer"},
            ]
        )
        chain = _serialize_messages_chain(
            [oai_msg], provider="openai", model_id="gpt-5"
        )

        history = [{"role": "user", "content": "Question?"}] + chain
        lc_messages = _convert_validated_messages(
            history, context="test", target_provider="google"
        )

        for msg in lc_messages:
            if hasattr(msg, "content") and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        assert block.get("type") != "reasoning"

    def test_same_provider_preserves_reasoning(self):
        """Claude → Claude preserves thinking blocks (as canonical reasoning)."""
        claude_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Analysis...", "signature": "sig"},
                {"type": "text", "text": "Answer"},
            ]
        )
        chain = _serialize_messages_chain(
            [claude_msg], provider="anthropic", model_id="claude-sonnet"
        )

        history = [{"role": "user", "content": "Question"}] + chain
        lc_messages = _convert_validated_messages(
            history, context="test", target_provider="anthropic"
        )

        # Verify: reasoning blocks are preserved
        has_reasoning = False
        for msg in lc_messages:
            if hasattr(msg, "content") and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "reasoning":
                        has_reasoning = True
        assert has_reasoning, "Reasoning blocks should be preserved for same provider"

    def test_tool_call_sequence_preserved_across_model_switch(self):
        """Tool call sequences remain valid after cross-model reasoning strip."""
        msgs = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "I should search"},
                ],
                tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
            ),
            ToolMessage(content="Search result", tool_call_id="call_1", name="search"),
            AIMessage(content="Based on the search..."),
        ]
        chain = _serialize_messages_chain(msgs, provider="anthropic", model_id="claude")

        history = [{"role": "user", "content": "Find something"}] + chain
        # Should not raise InvalidToolMessageSequenceError
        lc_messages = _convert_validated_messages(
            history, context="test", target_provider="openai"
        )
        assert len(lc_messages) == 4  # user + 3 from chain
