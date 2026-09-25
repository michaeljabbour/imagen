"""Shared pytest fixtures for the imagen test suite.

The single most important fixture here is ``fake_env`` (autouse): it gives
every test a deterministic environment with both provider API keys present
and an isolated ``OUTPUT_DIR``, then clears the ``lru_cache`` on
``get_settings()`` / ``get_provider_registry()`` so fresh instances pick up
the patched environment. This replaces the per-module
``os.environ.setdefault(...)`` calls that previously left provider
availability dependent on import order.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.config.settings import get_settings
from src.providers.base import ImageResult
from src.providers.registry import get_provider_registry

# 1x1 red-pixel PNG, base64-encoded — reused across provider/save tests.
SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def fake_env(monkeypatch, tmp_path):
    """Provide a deterministic environment for every test.

    Sets both API keys and an isolated output directory, and clears the
    settings/registry caches before and after so no state leaks between
    tests. Tests that need the *absence* of a key (e.g. fallback paths)
    can still ``monkeypatch.delenv(...)`` in their own body — that runs
    after this fixture and takes precedence.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("IMAGEN_MCP_ALLOWED_INPUT_ROOTS", str(tmp_path))

    get_settings.cache_clear()
    get_provider_registry.cache_clear()
    _reset_rate_limiter()
    yield
    get_settings.cache_clear()
    get_provider_registry.cache_clear()
    _reset_rate_limiter()


def _reset_rate_limiter() -> None:
    """Drop the rate-limiter singleton so each test gets a fresh, isolated one."""
    import src.services.rate_limiter as rl

    rl._limiter = None


@pytest.fixture
def settings():
    """Return a freshly-loaded Settings instance for the patched env.

    Relies on the autouse ``fake_env`` fixture having already patched the
    environment and cleared the settings cache.
    """
    return get_settings()


@pytest.fixture
def conversation_store(tmp_path):
    """An isolated ConversationStore backed by a temp SQLite file."""
    from src.services.conversation_store import ConversationStore

    return ConversationStore(tmp_path / "test_conversations.db")


@pytest.fixture
def mock_chat_completion() -> dict[str, Any]:
    """A mocked OpenAI /chat/completions response with a forced tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "generate_image",
                                "arguments": json.dumps(
                                    {"prompt": "A beautiful sunset", "size": "1024x1024"}
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }


@pytest.fixture
def mock_image_response() -> dict[str, Any]:
    """A mocked OpenAI /images/generations response (single image)."""
    return {"data": [{"b64_json": SAMPLE_IMAGE_B64}]}


@pytest.fixture
def mock_image_response_batch() -> dict[str, Any]:
    """A mocked /images/generations response with multiple images (n>1)."""
    return {
        "data": [
            {"b64_json": SAMPLE_IMAGE_B64},
            {"b64_json": SAMPLE_IMAGE_B64},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 200, "total_tokens": 210},
    }


@pytest.fixture
def sample_image_result() -> ImageResult:
    """A minimal successful ImageResult for formatting tests."""
    return ImageResult(
        success=True,
        provider="openai",
        model="gpt-image-2",
        prompt="A sunset",
        size="1024x1024",
    )
