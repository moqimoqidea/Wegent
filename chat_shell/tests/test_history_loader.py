# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from chat_shell.history.loader import (
    _build_knowledge_base_text_prefix,
    _strip_selected_documents,
)


class TestHistoryLoaderRestrictedKnowledgeBase:
    def test_restricted_kb_context_is_not_injected_into_history(self):
        context = SimpleNamespace(
            id=10,
            name="KB",
            knowledge_id=123,
            extracted_text="sensitive text",
            type_data={"rag_result": {"restricted_mode": True}},
        )

        assert _build_knowledge_base_text_prefix(context) == ""


class TestStripSelectedDocuments:
    """Tests for _strip_selected_documents which removes document content
    from system-reminder blocks before DB persistence."""

    def test_strips_selected_documents_keeps_current_time(self):
        """The typical case: system-reminder has both documents and time."""
        content = [
            {"type": "text", "text": "hello"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<selected_documents>Large doc content here</selected_documents>"
                    "<CurrentTime>2026-03-24 00:22</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_selected_documents(content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "hello"}
        assert result[1] == {
            "type": "text",
            "text": (
                "<system-reminder>"
                "<CurrentTime>2026-03-24 00:22</CurrentTime>"
                "</system-reminder>"
            ),
        }

    def test_drops_block_when_only_selected_documents(self):
        """If system-reminder only had documents, the block is removed entirely."""
        content = [
            {"type": "text", "text": "hello"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<selected_documents>doc content</selected_documents>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_selected_documents(content)
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "hello"}

    def test_no_system_reminder_unchanged(self):
        """Content without system-reminder blocks passes through unchanged."""
        content = [
            {"type": "text", "text": "user message"},
        ]
        result = _strip_selected_documents(content)
        assert result == content

    def test_multiline_document_content(self):
        """Documents with newlines are fully stripped."""
        doc = "line1\nline2\nline3\n" * 100
        content = [
            {"type": "text", "text": "question"},
            {
                "type": "text",
                "text": (
                    f"<system-reminder>"
                    f"<selected_documents>{doc}</selected_documents>"
                    f"<CurrentTime>2026-03-24 12:00</CurrentTime>"
                    f"</system-reminder>"
                ),
            },
        ]
        result = _strip_selected_documents(content)
        assert len(result) == 2
        assert "<selected_documents>" not in result[1]["text"]
        assert "<CurrentTime>" in result[1]["text"]

    def test_preserves_non_text_blocks(self):
        """Non-text blocks (e.g. image_url) are kept as-is."""
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<selected_documents>doc</selected_documents>"
                    "<CurrentTime>2026-03-24</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_selected_documents(content)
        assert len(result) == 3
        assert result[1]["type"] == "image_url"

    def test_empty_list(self):
        assert _strip_selected_documents([]) == []

    def test_system_reminder_without_selected_documents(self):
        """System-reminder without documents is kept unchanged."""
        content = [
            {"type": "text", "text": "msg"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<CurrentTime>2026-03-24 10:00</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_selected_documents(content)
        assert result == content
