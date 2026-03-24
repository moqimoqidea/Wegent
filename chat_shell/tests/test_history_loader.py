# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from chat_shell.history.loader import (
    _build_knowledge_base_text_prefix,
    _extract_time_text,
    _strip_context_from_reminder,
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


class TestStripContextFromReminder:
    """Tests for _strip_context_from_reminder which removes SubtaskContext-backed
    content (selected_documents, attachment, knowledge_base) from system-reminder
    blocks before DB persistence."""

    def test_strips_selected_documents_keeps_current_time(self):
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
        result = _strip_context_from_reminder(content)
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

    def test_strips_attachment_keeps_current_time(self):
        attachment_text = "[Attachment 1]\nfilename.xlsx | ID: 123\n" + "data " * 10000
        content = [
            {"type": "text", "text": "analyse this"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    f"<attachment>{attachment_text}</attachment>"
                    "<CurrentTime>2026-03-24 10:00</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 2
        assert "<attachment>" not in result[1]["text"]
        assert "<CurrentTime>2026-03-24 10:00</CurrentTime>" in result[1]["text"]

    def test_strips_knowledge_base(self):
        content = [
            {"type": "text", "text": "question"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<knowledge_base>KB content here</knowledge_base>"
                    "<CurrentTime>2026-03-24 12:00</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 2
        assert "<knowledge_base>" not in result[1]["text"]
        assert "<CurrentTime>" in result[1]["text"]

    def test_strips_all_context_tags_simultaneously(self):
        """Attachment + selected_documents + knowledge_base all stripped."""
        content = [
            {"type": "text", "text": "hello"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<attachment>file content</attachment>"
                    "<selected_documents>doc content</selected_documents>"
                    "<knowledge_base>kb content</knowledge_base>"
                    "<CurrentTime>2026-03-24 14:00</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 2
        text = result[1]["text"]
        assert "<attachment>" not in text
        assert "<selected_documents>" not in text
        assert "<knowledge_base>" not in text
        assert "<CurrentTime>2026-03-24 14:00</CurrentTime>" in text

    def test_drops_block_when_only_context_tags(self):
        """If system-reminder only has context content, the block is removed."""
        content = [
            {"type": "text", "text": "hello"},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<attachment>file</attachment>"
                    "<selected_documents>doc</selected_documents>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "hello"}

    def test_no_system_reminder_unchanged(self):
        content = [{"type": "text", "text": "user message"}]
        result = _strip_context_from_reminder(content)
        assert result == content

    def test_multiline_attachment_content(self):
        doc = "line1\nline2\nline3\n" * 100
        content = [
            {"type": "text", "text": "question"},
            {
                "type": "text",
                "text": (
                    f"<system-reminder>"
                    f"<attachment>{doc}</attachment>"
                    f"<CurrentTime>2026-03-24 12:00</CurrentTime>"
                    f"</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 2
        assert "<attachment>" not in result[1]["text"]
        assert "<CurrentTime>" in result[1]["text"]

    def test_preserves_non_text_blocks(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<attachment>file</attachment>"
                    "<CurrentTime>2026-03-24</CurrentTime>"
                    "</system-reminder>"
                ),
            },
        ]
        result = _strip_context_from_reminder(content)
        assert len(result) == 3
        assert result[1]["type"] == "image_url"

    def test_empty_list(self):
        assert _strip_context_from_reminder([]) == []

    def test_system_reminder_without_context_tags(self):
        """System-reminder with only CurrentTime is kept unchanged."""
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
        result = _strip_context_from_reminder(content)
        assert result == content


class TestExtractTimeText:
    """Tests for _extract_time_text which extracts <CurrentTime> from extra_blocks."""

    def test_extracts_current_time_from_system_reminder(self):
        blocks = [
            {
                "type": "text",
                "text": (
                    "<system-reminder>"
                    "<CurrentTime>2026-03-24 14:30</CurrentTime>"
                    "</system-reminder>"
                ),
            }
        ]
        assert (
            _extract_time_text(blocks) == "<CurrentTime>2026-03-24 14:30</CurrentTime>"
        )

    def test_returns_none_when_no_time(self):
        blocks = [{"type": "text", "text": "some other content"}]
        assert _extract_time_text(blocks) is None

    def test_returns_none_for_empty_list(self):
        assert _extract_time_text([]) is None

    def test_ignores_non_system_reminder_blocks(self):
        blocks = [
            {"type": "text", "text": "user text with <CurrentTime>fake</CurrentTime>"},
        ]
        assert _extract_time_text(blocks) is None
