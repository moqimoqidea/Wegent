# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for think-block handling in non-streaming graph builder paths.

These tests cover the two wrappers around
``LangGraphAgentBuilder._collect_final_state_from_events()``:
- ``execute()``
- ``stream_events_with_state()``

They ensure ``target_provider`` is passed through so cross-provider reasoning
blocks are stripped and same-provider Anthropic reasoning is denormalized.
"""

from unittest.mock import MagicMock, patch

from chat_shell.agents.graph_builder import LangGraphAgentBuilder


class RecordingAgent:
    """Minimal fake LangGraph agent that records converted input messages."""

    def __init__(self):
        self.received_messages = None
        self.received_config = None
        self.received_version = None

    async def astream_events(self, inputs, config=None, version=None):
        self.received_messages = inputs["messages"]
        self.received_config = config
        self.received_version = version
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": self.received_messages}},
        }


class TestGraphBuilderNonStreamingThinkBlocks:
    """Regression tests for non-streaming think-block adaptation paths."""

    async def test_execute_strips_foreign_reasoning_blocks(self):
        """execute() strips cross-provider reasoning before agent execution."""
        mock_llm = MagicMock()
        mock_llm._wegent_provider = "openai"
        builder = LangGraphAgentBuilder(llm=mock_llm)
        fake_agent = RecordingAgent()

        messages = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "Claude thought",
                        "extras": {"signature": "sig_123"},
                    },
                    {"type": "text", "text": "Answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude-sonnet"},
            },
        ]

        with patch.object(builder, "_build_agent", return_value=fake_agent):
            final_state = await builder.execute(messages)

        assert fake_agent.received_version == "v2"
        assert final_state["messages"] == fake_agent.received_messages

        assistant_msg = fake_agent.received_messages[1]
        if isinstance(assistant_msg.content, list):
            for block in assistant_msg.content:
                if isinstance(block, dict):
                    assert block.get("type") != "reasoning"
                    assert block.get("type") != "thinking"
            assert any(
                isinstance(block, dict) and block.get("type") == "text"
                for block in assistant_msg.content
            )
        else:
            assert assistant_msg.content == "Answer"

    async def test_stream_events_with_state_denormalizes_anthropic_reasoning(self):
        """stream_events_with_state() restores Anthropic thinking blocks."""
        mock_llm = MagicMock()
        mock_llm._wegent_provider = "anthropic"
        builder = LangGraphAgentBuilder(llm=mock_llm)
        fake_agent = RecordingAgent()

        messages = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "Anthropic thought",
                        "extras": {"signature": "sig_abc"},
                    },
                    {"type": "text", "text": "Answer"},
                ],
                "model_info": {"provider": "anthropic", "model": "claude-sonnet"},
            },
        ]

        with patch.object(builder, "_build_agent", return_value=fake_agent):
            final_state, all_events = await builder.stream_events_with_state(messages)

        assert fake_agent.received_version == "v2"
        assert len(all_events) == 1
        assert final_state["messages"] == fake_agent.received_messages

        assistant_msg = fake_agent.received_messages[1]
        assert assistant_msg.response_metadata["model_provider"] == "anthropic"
        assert isinstance(assistant_msg.content, list)
        assert assistant_msg.content[0] == {
            "type": "thinking",
            "thinking": "Anthropic thought",
            "signature": "sig_abc",
        }
        assert assistant_msg.content[1] == {"type": "text", "text": "Answer"}
