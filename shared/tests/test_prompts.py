# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for shared prompts module.

Tests that KB prompt constants are properly exported and accessible.
"""

import pytest


class TestKBPromptConstants:
    """Test KB prompt constant exports."""

    def test_kb_prompt_strict_importable(self):
        """Should be able to import KB_PROMPT_STRICT from shared.prompts."""
        from shared.prompts import KB_PROMPT_STRICT

        assert KB_PROMPT_STRICT is not None
        assert isinstance(KB_PROMPT_STRICT, str)
        assert len(KB_PROMPT_STRICT) > 0

    def test_kb_prompt_relaxed_importable(self):
        """Should be able to import KB_PROMPT_RELAXED from shared.prompts."""
        from shared.prompts import KB_PROMPT_RELAXED

        assert KB_PROMPT_RELAXED is not None
        assert isinstance(KB_PROMPT_RELAXED, str)
        assert len(KB_PROMPT_RELAXED) > 0

    def test_kb_prompt_strict_contains_required_content(self):
        """KB_PROMPT_STRICT should contain strict mode instructions."""
        from shared.prompts import KB_PROMPT_STRICT

        # Check for key phrases in strict mode (Intent Routing approach)
        assert "MUST NOT" in KB_PROMPT_STRICT  # Critical rule for strict mode
        assert "knowledge_base_search" in KB_PROMPT_STRICT
        assert "ONLY" in KB_PROMPT_STRICT or "only" in KB_PROMPT_STRICT
        assert "Intent Routing" in KB_PROMPT_STRICT  # New routing approach

    def test_kb_prompt_relaxed_contains_required_content(self):
        """KB_PROMPT_RELAXED should contain relaxed mode instructions."""
        from shared.prompts import KB_PROMPT_RELAXED

        # Check for key phrases in relaxed mode
        assert "knowledge_base_search" in KB_PROMPT_RELAXED
        # Relaxed mode must allow fallback to general knowledge
        assert "general knowledge" in KB_PROMPT_RELAXED
        # All KB prompts use Intent Routing pattern
        assert "Intent Routing" in KB_PROMPT_RELAXED

    def test_prompts_are_different(self):
        """Strict and relaxed prompts should be different."""
        from shared.prompts import KB_PROMPT_RELAXED, KB_PROMPT_STRICT

        assert KB_PROMPT_STRICT != KB_PROMPT_RELAXED

    def test_prompts_module_all_export(self):
        """shared.prompts module should export KB_PROMPT_STRICT, KB_PROMPT_RELAXED and KB_PROMPT_NO_RAG in __all__."""
        from shared import prompts

        assert hasattr(prompts, "__all__")
        assert "KB_PROMPT_STRICT" in prompts.__all__
        assert "KB_PROMPT_RELAXED" in prompts.__all__
        assert "KB_PROMPT_NO_RAG" in prompts.__all__

    def test_kb_prompt_no_rag_importable(self):
        """Should be able to import KB_PROMPT_NO_RAG from shared.prompts."""
        from shared.prompts import KB_PROMPT_NO_RAG

        assert KB_PROMPT_NO_RAG is not None
        assert isinstance(KB_PROMPT_NO_RAG, str)
        assert len(KB_PROMPT_NO_RAG) > 0

    def test_kb_prompt_no_rag_contains_required_content(self):
        """KB_PROMPT_NO_RAG should contain no-RAG mode instructions."""
        from shared.prompts import KB_PROMPT_NO_RAG

        # Check for key phrases in no-RAG mode
        assert "Exploration Mode" in KB_PROMPT_NO_RAG
        assert "kb_ls" in KB_PROMPT_NO_RAG
        assert "kb_head" in KB_PROMPT_NO_RAG
        assert "RAG retrieval is NOT configured" in KB_PROMPT_NO_RAG
        assert "Intent Routing" in KB_PROMPT_NO_RAG


class TestUserQuestionMarker:
    """Test USER_QUESTION_MARKER constant and extract_user_question utility."""

    def test_marker_importable(self):
        from shared.prompts import USER_QUESTION_MARKER

        assert USER_QUESTION_MARKER == "[User Question]:"

    def test_extract_plain_text(self):
        from shared.prompts.constants import extract_user_question

        assert extract_user_question("Hello world") == "Hello world"

    def test_extract_with_attachment_context(self):
        from shared.prompts.constants import extract_user_question

        text = "<attachment>\nsome metadata\n</attachment>\n\n[User Question]:\nWhat is this?"
        assert extract_user_question(text) == "What is this?"

    def test_extract_with_kb_and_attachment_context(self):
        from shared.prompts.constants import extract_user_question

        text = (
            "<knowledge_base>\nKB content\n</knowledge_base>\n\n"
            "<attachment>\ndoc content\n</attachment>\n\n"
            "[User Question]:\nSummarize"
        )
        assert extract_user_question(text) == "Summarize"

    def test_extract_strips_whitespace(self):
        from shared.prompts.constants import extract_user_question

        assert extract_user_question("  Hello  ") == "Hello"

    def test_extract_non_string_returns_str(self):
        from shared.prompts.constants import extract_user_question

        assert extract_user_question(123) == "123"

    def test_extract_preserves_multiline_question(self):
        from shared.prompts.constants import extract_user_question

        text = "<attachment>\ndata\n</attachment>\n\n[User Question]:\nLine1\nLine2"
        assert extract_user_question(text) == "Line1\nLine2"


class TestParsePromptBlocks:
    """Tests for parse_prompt_blocks utility."""

    def test_plain_text_returns_as_is(self):
        from shared.prompts.constants import parse_prompt_blocks

        text_content, extra = parse_prompt_blocks("Hello world")
        assert text_content == "Hello world"
        assert extra == []

    def test_json_array_two_text_blocks(self):
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(
            [
                {"type": "text", "text": "User question"},
                {"type": "text", "text": "<system-reminder>time</system-reminder>"},
            ]
        )
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "User question"
        assert len(extra) == 1
        assert extra[0]["text"] == "<system-reminder>time</system-reminder>"

    def test_json_array_single_block(self):
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps([{"type": "text", "text": "Only block"}])
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "Only block"
        assert extra == []

    def test_json_array_skips_image_url_blocks(self):
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(
            [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "<system-reminder>t</system-reminder>"},
            ]
        )
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "describe"
        assert len(extra) == 1

    def test_invalid_json_returns_raw(self):
        from shared.prompts.constants import parse_prompt_blocks

        raw = "[not json"
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == raw
        assert extra == []

    def test_json_non_array_returns_raw(self):
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps({"type": "text", "text": "hello"})
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == raw
        assert extra == []

    def test_json_array_of_non_dicts_returns_raw(self):
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(["a", "b"])
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == raw
        assert extra == []

    def test_empty_string(self):
        from shared.prompts.constants import parse_prompt_blocks

        text_content, extra = parse_prompt_blocks("")
        assert text_content == ""
        assert extra == []

    def test_importable_from_init(self):
        from shared.prompts import parse_prompt_blocks

        text, _extra = parse_prompt_blocks("test")
        assert text == "test"

    def test_input_text_type_blocks(self):
        """Old format: <attachment> + [User Question]: blocks → extracts user message."""
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(
            [
                {"type": "input_text", "text": "<attachment>metadata</attachment>"},
                {"type": "input_image", "image_url": "data:image/png;base64,..."},
                {"type": "input_text", "text": "[User Question]:\nDescribe this"},
            ]
        )
        text_content, extra = parse_prompt_blocks(raw)
        # User message extracted from [User Question]: block
        assert text_content == "Describe this"
        # Old <attachment> block is discarded; no extra_blocks
        assert extra == []

    def test_new_format_user_msg_and_system_reminder(self):
        """New format: user message + system-reminder."""
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(
            [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "text",
                    "text": "<system-reminder><attachment>meta</attachment>[time]</system-reminder>",
                },
            ]
        )
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "Describe this image"
        assert len(extra) == 1
        assert "<system-reminder>" in extra[0]["text"]

    def test_old_format_string_with_marker(self):
        """Old plain-text format with [User Question]: extracts the actual question."""
        from shared.prompts.constants import parse_prompt_blocks

        raw = "<attachment>\ndoc content\n</attachment>\n\n[User Question]:\nWhat is this?"
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "What is this?"
        assert extra == []

    def test_input_text_single_block(self):
        """Single input_text block after image stripping."""
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps([{"type": "input_text", "text": "Hello"}])
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "Hello"
        assert extra == []

    def test_mixed_text_and_input_text(self):
        """Mixed LangChain and Responses API format blocks."""
        import json

        from shared.prompts.constants import parse_prompt_blocks

        raw = json.dumps(
            [
                {"type": "text", "text": "User message"},
                {"type": "input_text", "text": "Extra context"},
            ]
        )
        text_content, extra = parse_prompt_blocks(raw)
        assert text_content == "User message"
        assert len(extra) == 1
        assert extra[0]["text"] == "Extra context"
