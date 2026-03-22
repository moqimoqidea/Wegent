# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for model_resolver - thinkingConfig extraction.

Tests that _extract_model_config correctly extracts thinkingConfig
from the model spec and maps it to think_config in the output dict.
"""

from unittest.mock import patch

import pytest

from app.services.chat.config.model_resolver import _extract_model_config


class TestExtractModelConfigThinkingConfig:
    """Tests for thinkingConfig extraction in _extract_model_config."""

    @staticmethod
    def _make_model_spec(thinking_config=None, **overrides):
        """Build a minimal model spec dict for testing."""
        spec = {
            "modelConfig": {
                "env": {
                    "api_key": "sk-test",
                    "base_url": "https://api.example.com/v1",
                    "model_id": "gpt-4",
                    "model": "openai",
                },
            },
            **overrides,
        }
        if thinking_config is not None:
            spec["thinkingConfig"] = thinking_config
        return spec

    @patch("app.services.chat.config.model_resolver.decrypt_api_key", side_effect=lambda k: k)
    def test_thinking_config_extracted(self, _decrypt):
        """thinkingConfig in spec → think_config in output."""
        spec = self._make_model_spec(
            thinking_config={"reasoning_effort": "medium"}
        )
        result = _extract_model_config(spec)
        assert result["think_config"] == {"reasoning_effort": "medium"}

    @patch("app.services.chat.config.model_resolver.decrypt_api_key", side_effect=lambda k: k)
    def test_thinking_config_absent(self, _decrypt):
        """No thinkingConfig in spec → think_config is None."""
        spec = self._make_model_spec()
        result = _extract_model_config(spec)
        assert result["think_config"] is None

    @patch("app.services.chat.config.model_resolver.decrypt_api_key", side_effect=lambda k: k)
    def test_thinking_config_complex_object(self, _decrypt):
        """Complex thinkingConfig → preserved as-is."""
        tc = {"thinking": {"type": "enabled", "budget_tokens": 10000}}
        spec = self._make_model_spec(thinking_config=tc)
        result = _extract_model_config(spec)
        assert result["think_config"] == tc

    @patch("app.services.chat.config.model_resolver.decrypt_api_key", side_effect=lambda k: k)
    def test_thinking_config_empty_dict(self, _decrypt):
        """Empty dict thinkingConfig → empty dict in output."""
        spec = self._make_model_spec(thinking_config={})
        result = _extract_model_config(spec)
        assert result["think_config"] == {}
