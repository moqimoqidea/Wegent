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
