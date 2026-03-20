# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MessageConverter - message building and conversion utilities.

This module tests the core message conversion functionality including:
- Message building with/without datetime injection
- Vision message handling
- Username prefixing for group chat
"""

import base64
from datetime import datetime

import pytest

# Import directly from the module to avoid triggering __init__.py dependencies
from chat_shell.messages.converter import MessageConverter


class TestMessageConverterBuildMessages:
    """Tests for MessageConverter.build_messages method."""

    def test_build_messages_basic(self):
        """Test basic message building."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="You are helpful.",
            inject_datetime=False,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_messages_with_datetime_injection(self):
        """Test datetime is injected as a system-reminder block when inject_datetime=True."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="",
            inject_datetime=True,
        )

        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        # Content should now be a list of two blocks
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2
        # First block: plain user text
        assert user_msg["content"][0] == {"type": "text", "text": "Hello"}
        # Second block: system-reminder with current time
        system_block = user_msg["content"][1]
        assert system_block["type"] == "text"
        assert "<system-reminder>" in system_block["text"]
        assert "[Current time:" in system_block["text"]
        assert datetime.now().strftime("%Y-%m-%d") in system_block["text"]

    def test_build_messages_without_datetime_injection(self):
        """Test datetime is NOT injected when inject_datetime=False.

        This is the expected behavior for API calls without wegent_chat_bot tool.
        """
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="",
            inject_datetime=False,
        )

        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert "[Current time:" not in user_msg["content"]
        assert user_msg["content"] == "Hello"

    def test_build_messages_default_injects_datetime(self):
        """Test that default behavior injects datetime as system-reminder block (backward compatibility)."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="",
            # inject_datetime defaults to True
        )

        user_msg = messages[-1]
        content = user_msg["content"]
        # New format: list with system-reminder block
        assert isinstance(content, list)
        system_block_texts = [b["text"] for b in content if b.get("type") == "text"]
        assert any("[Current time:" in t for t in system_block_texts)

    def test_build_messages_with_history(self):
        """Test that history is preserved."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        messages = MessageConverter.build_messages(
            history=history,
            current_message="How are you?",
            system_prompt="Be helpful",
            inject_datetime=False,
        )

        assert len(messages) == 4  # system + 2 history + current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["content"] == "Hello!"
        assert messages[3]["content"] == "How are you?"

    def test_build_messages_with_username(self):
        """Test username prefix for group chat."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello everyone",
            system_prompt="",
            username="Alice",
            inject_datetime=False,
        )

        user_msg = messages[-1]
        assert "User[Alice]:" in user_msg["content"]
        assert "Hello everyone" in user_msg["content"]

    def test_build_messages_without_system_prompt(self):
        """Test building messages without system prompt."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="",  # Empty
            inject_datetime=False,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_build_messages_with_dynamic_context_injection(self):
        """Test that dynamic_context is injected as a human message."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        messages = MessageConverter.build_messages(
            history=history,
            current_message="Question",
            system_prompt="SYS",
            inject_datetime=False,
            dynamic_context="DYNAMIC",
        )

        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "user",  # dynamic_context
            "user",  # current
        ]
        assert messages[3]["content"] == "DYNAMIC"
        assert messages[4]["content"] == "Question"

    def test_build_messages_dynamic_context_position(self):
        """Test that dynamic_context is inserted after history and before current message."""
        history = [
            {"role": "user", "content": "H1"},
            {"role": "assistant", "content": "A1"},
        ]

        messages = MessageConverter.build_messages(
            history=history,
            current_message="CUR",
            system_prompt="SYS",
            inject_datetime=False,
            dynamic_context="CTX",
        )

        assert messages[0] == {"role": "system", "content": "SYS"}
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3] == {"role": "user", "content": "CTX"}
        assert messages[4] == {"role": "user", "content": "CUR"}

    def test_build_messages_empty_dynamic_context(self):
        """Test that empty dynamic_context does not create an extra message."""
        messages_none = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="SYS",
            inject_datetime=False,
            dynamic_context=None,
        )
        messages_empty = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="SYS",
            inject_datetime=False,
            dynamic_context="",
        )

        assert len(messages_none) == 2
        assert len(messages_empty) == 2

    def test_build_messages_with_all_parameters(self):
        """Test message building with system_prompt, history, dynamic_context and username."""
        history = [
            {"role": "user", "content": "H1"},
            {"role": "assistant", "content": "A1"},
        ]

        messages = MessageConverter.build_messages(
            history=history,
            current_message="Hello",
            system_prompt="SYS",
            username="Alice",
            inject_datetime=False,
            dynamic_context="KB_META",
        )

        assert messages[0]["role"] == "system"
        assert messages[3] == {"role": "user", "content": "KB_META"}
        assert messages[4]["role"] == "user"
        assert messages[4]["content"].startswith("User[Alice]:")
        assert "Hello" in messages[4]["content"]

    def test_build_messages_vision_message(self):
        """Test building vision messages with OpenAI Responses API format."""
        # Create a tiny valid image (1x1 red PNG)
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        image_b64 = base64.b64encode(tiny_png).decode()

        # OpenAI Responses API format: list of content blocks
        vision_data = [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
        ]

        messages = MessageConverter.build_messages(
            history=[],
            current_message=vision_data,
            system_prompt="",
            inject_datetime=False,
        )

        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        # Should have text and image blocks (converted to LangChain format)
        text_block = next(
            (b for b in user_msg["content"] if b.get("type") == "text"), None
        )
        image_block = next(
            (b for b in user_msg["content"] if b.get("type") == "image_url"), None
        )
        assert text_block is not None
        assert image_block is not None
        assert "What is this?" in text_block["text"]
        # Verify image_url is in LangChain format (nested dict)
        assert "url" in image_block["image_url"]


class TestMessageConverterExtractText:
    """Tests for MessageConverter.extract_text method."""

    def test_extract_text_from_string(self):
        """Test extracting text from string message."""
        result = MessageConverter.extract_text("Hello world")
        assert result == "Hello world"

    def test_extract_text_from_dict(self):
        """Test extracting text from dict message."""
        message = {"role": "user", "content": "Hello world"}
        result = MessageConverter.extract_text(message)
        assert result == "Hello world"

    def test_extract_text_from_multipart(self):
        """Test extracting text from multipart content."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "world"},
            ],
        }
        result = MessageConverter.extract_text(message)
        assert "Hello" in result
        assert "world" in result


class TestMessageConverterIsVisionMessage:
    """Tests for MessageConverter.is_vision_message method."""

    def test_is_vision_message_true(self):
        """Test detecting vision message."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
        assert MessageConverter.is_vision_message(message) is True

    def test_is_vision_message_false_text_only(self):
        """Test non-vision text message."""
        message = {"role": "user", "content": "Hello"}
        assert MessageConverter.is_vision_message(message) is False

    def test_is_vision_message_false_list_without_image(self):
        """Test multipart message without image."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        assert MessageConverter.is_vision_message(message) is False


class TestConvertResponsesAPIToLangchain:
    """Tests for _convert_responses_api_to_langchain with new username/time_block changes."""

    def test_username_applied_to_last_text_block(self):
        """Username prefix is applied to the LAST text block (user's question), not the first."""
        blocks = [
            {"type": "input_text", "text": "<attachment>\nmetadata\n</attachment>"},
            {"type": "input_text", "text": "What is this?"},
        ]
        result = MessageConverter._convert_responses_api_to_langchain(
            blocks, username="Alice"
        )
        content = result["content"]
        # The first block (attachment metadata) must NOT have the prefix
        assert content[0]["text"] == "<attachment>\nmetadata\n</attachment>"
        # The second block (user question) should have the prefix
        assert content[1]["text"] == "User[Alice]: What is this?"

    def test_time_block_appended_at_end(self):
        """System-reminder time block is appended after all content blocks."""
        blocks = [{"type": "input_text", "text": "Hello"}]
        time_block = {
            "type": "text",
            "text": "<system-reminder>[time]</system-reminder>",
        }
        result = MessageConverter._convert_responses_api_to_langchain(
            blocks, time_block=time_block
        )
        content = result["content"]
        assert len(content) == 2
        assert content[0]["text"] == "Hello"
        assert content[1] == time_block

    def test_username_and_time_block_combined(self):
        """Username prefix on last text + time block at end."""
        blocks = [
            {"type": "input_text", "text": "attachment data"},
            {"type": "input_text", "text": "My question"},
        ]
        time_block = {
            "type": "text",
            "text": "<system-reminder>[now]</system-reminder>",
        }
        result = MessageConverter._convert_responses_api_to_langchain(
            blocks, username="Bob", time_block=time_block
        )
        content = result["content"]
        assert len(content) == 3
        assert content[0]["text"] == "attachment data"
        assert content[1]["text"] == "User[Bob]: My question"
        assert content[2] == time_block

    def test_no_username_no_time_block(self):
        """Without username and time_block, content is passed through."""
        blocks = [{"type": "input_text", "text": "Hello"}]
        result = MessageConverter._convert_responses_api_to_langchain(blocks)
        content = result["content"]
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello"}

    def test_vision_message_with_username(self):
        """Vision message: username applied to last text block, images preserved."""
        blocks = [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ]
        result = MessageConverter._convert_responses_api_to_langchain(
            blocks, username="Alice"
        )
        content = result["content"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "User[Alice]: What is this?"
        assert len(image_blocks) == 1

    def test_single_text_block_gets_username(self):
        """With only one text block, it gets the username prefix."""
        blocks = [{"type": "input_text", "text": "Hello"}]
        result = MessageConverter._convert_responses_api_to_langchain(
            blocks, username="X"
        )
        assert result["content"][0]["text"] == "User[X]: Hello"


class TestBuildMessagesPlainTextWithTimeBlock:
    """Tests for plain text message building with the new time block format."""

    def test_plain_text_with_datetime_produces_list_content(self):
        """Plain text + datetime injection produces a list of two blocks."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hi",
            system_prompt="",
            inject_datetime=True,
        )
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "Hi"
        assert "<system-reminder>" in user_msg["content"][1]["text"]

    def test_plain_text_with_username_and_datetime(self):
        """Plain text + username + datetime: username in first block text."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hi",
            system_prompt="",
            username="Alice",
            inject_datetime=True,
        )
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["text"] == "User[Alice]: Hi"

    def test_plain_text_no_datetime_stays_string(self):
        """Without datetime injection, content remains a plain string."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hi",
            system_prompt="",
            inject_datetime=False,
        )
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], str)
        assert user_msg["content"] == "Hi"


class TestCacheBreakpoints:
    """Tests for Anthropic explicit cache breakpoint support."""

    _CC = {"type": "ephemeral"}

    def test_no_breakpoints_by_default(self):
        """Cache breakpoints should not be added unless explicitly applied."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="System",
            inject_datetime=False,
        )
        # System prompt stays a plain string
        assert messages[0]["content"] == "System"
        assert "cache_control" not in messages[0]

    def test_breakpoint_on_system_prompt(self):
        """System prompt should get a cache_control breakpoint."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="System",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=False,
            has_dynamic_context=False,
        )
        sys_msg = messages[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["text"] == "System"
        assert sys_msg["content"][0]["cache_control"] == self._CC

    def test_breakpoint_on_last_history_message(self):
        """Last history message should get a cache_control breakpoint."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        messages = MessageConverter.build_messages(
            history=history,
            current_message="Next question",
            system_prompt="System",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=True,
            has_dynamic_context=False,
        )
        # messages[0]=system, messages[1]=user(history), messages[2]=assistant(history), messages[3]=current
        # The assistant message (last in history) should have cache_control
        assistant_msg = messages[2]
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][-1]["cache_control"] == self._CC

    def test_breakpoint_on_dynamic_context(self):
        """Dynamic context message should get a cache_control breakpoint."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Tell me about X",
            system_prompt="System",
            dynamic_context="KB context here",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=False,
            has_dynamic_context=True,
        )
        # messages[0]=system, messages[1]=dynamic_context, messages[2]=current
        dc_msg = messages[1]
        assert dc_msg["role"] == "user"
        assert isinstance(dc_msg["content"], list)
        assert dc_msg["content"][-1]["cache_control"] == self._CC

    def test_no_breakpoint_on_current_message(self):
        """The current user message should NOT have a cache_control breakpoint."""
        messages = MessageConverter.build_messages(
            history=[],
            current_message="Hello",
            system_prompt="System",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=False,
            has_dynamic_context=False,
        )
        user_msg = messages[-1]
        if isinstance(user_msg["content"], list):
            for block in user_msg["content"]:
                assert "cache_control" not in block
        elif isinstance(user_msg["content"], str):
            pass  # no cache_control possible on a string

    def test_breakpoint_with_multiblock_history(self):
        """History messages that already have list content get breakpoint on last block."""
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "text", "text": "<system-reminder>time</system-reminder>"},
                ],
            },
            {"role": "assistant", "content": "Hello there!"},
        ]
        messages = MessageConverter.build_messages(
            history=history,
            current_message="Follow up",
            system_prompt="System",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=True,
            has_dynamic_context=False,
        )
        # The assistant message should have cache_control
        assistant_msg = messages[2]
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][-1]["cache_control"] == self._CC

    def test_all_breakpoints_with_full_context(self):
        """System, history, and dynamic context should all get breakpoints."""
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]
        messages = MessageConverter.build_messages(
            history=history,
            current_message="New question",
            system_prompt="You are helpful.",
            dynamic_context="Important KB content",
            inject_datetime=False,
        )
        MessageConverter.apply_cache_breakpoints(
            messages,
            has_history=True,
            has_dynamic_context=True,
        )
        # messages: [system, user(hist), assistant(hist), dynamic_ctx, current]
        assert len(messages) == 5

        # System has breakpoint
        sys_content = messages[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[-1]["cache_control"] == self._CC

        # Last history (assistant) has breakpoint
        hist_content = messages[2]["content"]
        assert isinstance(hist_content, list)
        assert hist_content[-1]["cache_control"] == self._CC

        # Dynamic context has breakpoint
        dc_content = messages[3]["content"]
        assert isinstance(dc_content, list)
        assert dc_content[-1]["cache_control"] == self._CC

        # Current message does NOT have breakpoint
        current_msg = messages[4]
        if isinstance(current_msg["content"], str):
            assert "cache_control" not in current_msg
        elif isinstance(current_msg["content"], list):
            for block in current_msg["content"]:
                assert "cache_control" not in block
