# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for extract_display_prompt in prompt_utils module."""

import json

import pytest

from app.utils.prompt_utils import extract_display_prompt


class TestExtractDisplayPrompt:
    """Tests for the extract_display_prompt utility."""

    def test_none_returns_none(self):
        assert extract_display_prompt(None) is None

    def test_empty_string_returns_empty(self):
        assert extract_display_prompt("") == ""

    def test_plain_text_returned_as_is(self):
        assert extract_display_prompt("Hello world") == "Hello world"

    def test_json_array_returns_first_text_block(self):
        prompt = json.dumps(
            [
                {"type": "text", "text": "User question"},
                {
                    "type": "text",
                    "text": "<system-reminder>\n[Current time: 2025-01-01 12:00]\n</system-reminder>",
                },
            ]
        )
        assert extract_display_prompt(prompt) == "User question"

    def test_json_array_single_text_block(self):
        """Single-element array still extracts the text."""
        prompt = json.dumps([{"type": "text", "text": "Only block"}])
        assert extract_display_prompt(prompt) == "Only block"

    def test_json_array_with_image_url_blocks(self):
        """image_url blocks are ignored; first text block is extracted."""
        prompt = json.dumps(
            [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {
                    "type": "text",
                    "text": "<system-reminder>[Current time: 2025-01-01]</system-reminder>",
                },
            ]
        )
        assert extract_display_prompt(prompt) == "Describe this image"

    def test_invalid_json_returns_original(self):
        prompt = "[not valid json"
        assert extract_display_prompt(prompt) == prompt

    def test_json_non_array_returns_original(self):
        """JSON object (not array) is treated as plain text."""
        prompt = json.dumps({"type": "text", "text": "hello"})
        assert extract_display_prompt(prompt) == prompt

    def test_json_array_of_non_dicts_returns_original(self):
        """JSON array of non-dict items is treated as plain text."""
        prompt = json.dumps(["string1", "string2"])
        assert extract_display_prompt(prompt) == prompt

    def test_whitespace_preserved_for_plain_text(self):
        """Leading/trailing whitespace in plain text prompts is not stripped."""
        prompt = "  Hello world  "
        assert extract_display_prompt(prompt) == prompt

    def test_json_array_empty_first_text(self):
        """Empty first text block returns empty string."""
        prompt = json.dumps(
            [
                {"type": "text", "text": ""},
                {"type": "text", "text": "system block"},
            ]
        )
        assert extract_display_prompt(prompt) == ""
