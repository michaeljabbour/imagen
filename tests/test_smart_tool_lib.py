"""Tests for src.smart_tool.lib -- the smart tool's capability surface.

These exercise the deterministic capabilities directly and verify that the
model-backed capabilities route through the resolved model with
``allow_unknown_model=True`` (no provider client construction is mocked out
here; the two /providers unit tests already at ``tests/test_providers.py``
cover ``allow_unknown`` at the provider layer -- this file checks the smart
tool wires it through).
"""

from __future__ import annotations

import pytest

from src.smart_tool import lib


def test_manifest_returns_frontmatter_and_body():
    result = lib.manifest()
    assert result["frontmatter"]["name"] == "imagen"
    assert result["frontmatter"]["smart_tool_format"] == 1
    assert "imagen-mcp" in result["body"]


def test_list_providers_reports_both_when_keys_present():
    result = lib.list_providers()
    assert set(result["providers"]) == {"openai", "gemini"}
    assert result["providers"]["openai"]["available"] is True
    assert result["providers"]["gemini"]["available"] is True
    assert set(result["available"]) == {"openai", "gemini"}


def test_list_providers_reports_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(lib, "_keychain_api_key", lambda env_var: None)
    result = lib.list_providers()
    assert result["available"] == []
    assert result["providers"]["openai"]["available"] is False


def test_estimate_cost_openai():
    result = lib.estimate_cost(provider="openai", size="1024x1024", quality="medium")
    assert result["provider"] == "openai"
    assert result["per_image_usd"] == 0.053


def test_estimate_cost_unknown_model_returns_note_not_error():
    result = lib.estimate_cost(provider="gemini", model="gemini-9000-nobody-has-heard-of")
    assert result["total_usd"] is None
    assert result["note"]


def test_estimate_cost_no_provider_uses_selector(monkeypatch):
    monkeypatch.setattr(lib, "_available_providers", lambda: ["openai"])
    result = lib.estimate_cost(prompt="a menu with text and a logo")
    assert result["provider"] == "openai"


async def test_generate_image_requires_a_configured_provider(monkeypatch):
    monkeypatch.setattr(lib, "_available_providers", lambda: [])
    with pytest.raises(lib.SmartToolError, match="No image-generation provider"):
        await lib.generate_image("a cat")


async def test_generate_image_rejects_unavailable_explicit_provider(monkeypatch):
    monkeypatch.setattr(lib, "_available_providers", lambda: ["gemini"])
    with pytest.raises(lib.SmartToolError, match="not available"):
        await lib.generate_image("a cat", provider="openai")


async def test_edit_image_requires_openai_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(lib, "_keychain_api_key", lambda env_var: None)
    with pytest.raises(lib.SmartToolError, match="OpenAI is not configured"):
        await lib.edit_image("add a hat", "/tmp/does-not-matter.png")


async def test_generate_image_passes_resolved_model_and_allow_unknown(monkeypatch):
    """The smart tool always resolves the model itself and always passes
    allow_unknown_model=True -- it is the resolver's job (explicit choice,
    config, or discovery), not a static allowlist, that decides validity."""
    captured: dict[str, object] = {}

    class _FakeResult:
        success = True
        error = None

        def to_dict(self):
            return {"ok": True}

    class _FakeProvider:
        async def generate_image(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return _FakeResult()

    class _FakeRegistry:
        def get_provider(self, name, *, api_key=None):
            captured["provider_name"] = name
            return _FakeProvider()

    monkeypatch.setattr(lib, "_available_providers", lambda: ["openai"])
    monkeypatch.setattr(lib, "get_provider_registry", lambda: _FakeRegistry())

    async def _fake_resolved_model(provider, requested_model, *, family="flash"):
        return "gpt-image-99-from-discovery"

    monkeypatch.setattr(lib, "_resolved_model", _fake_resolved_model)

    result = await lib.generate_image("a menu with text", provider="openai")

    assert result == {"ok": True}
    assert captured["provider_name"] == "openai"
    assert captured["kwargs"]["openai_model"] == "gpt-image-99-from-discovery"
    assert captured["kwargs"]["allow_unknown_model"] is True
