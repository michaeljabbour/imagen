"""Tests for src.smart_tool.model_resolution.

Covers the resolution order (explicit > config > discovery > cache >
hardcoded fallback) with the provider listing APIs mocked via respx, plus
the two scenarios called out explicitly in the task: "a newer model appears
-> it is selected" and "discovery fails -> fallback".
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.smart_tool import model_resolution as mr


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Give every test its own config file and cache file paths."""
    monkeypatch.setenv("IMAGEN_MCP_MODELS_CONFIG", str(tmp_path / "models.yaml"))
    monkeypatch.setenv("IMAGEN_MCP_DISCOVERY_CACHE", str(tmp_path / "cache.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def test_load_config_missing_file_returns_empty():
    assert mr.load_config() == {}


def test_load_config_parses_yaml(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "openai: latest\n"
        "gemini: gemini-3-pro-image\n"
        "aliases:\n"
        "  fast: gemini-3.1-flash-lite-image\n"
    )
    config = mr.load_config(path)
    assert config["openai"] == "latest"
    assert config["gemini"] == "gemini-3-pro-image"
    assert config["aliases"]["fast"] == "gemini-3.1-flash-lite-image"


def test_load_config_invalid_yaml_returns_empty(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("not: [valid: yaml: at all")
    assert mr.load_config(path) == {}


# ---------------------------------------------------------------------------
# OpenAI discovery ranking
# ---------------------------------------------------------------------------


def test_newest_openai_image_model_picks_highest_version():
    models = [
        {"id": "gpt-image-1", "created": 100},
        {"id": "gpt-image-1.5", "created": 200},
        {"id": "gpt-image-2", "created": 300},
        {"id": "gpt-5.1", "created": 999},  # not an image model -- must be ignored
    ]
    assert mr.newest_openai_image_model(models) == "gpt-image-2"


def test_newest_openai_image_model_a_newer_model_appears():
    """The core requirement: a model this tool has never seen wins on ranking."""
    models = [
        {"id": "gpt-image-2", "created": 100},
        {"id": "gpt-image-3", "created": 200},  # hypothetical future release
    ]
    assert mr.newest_openai_image_model(models) == "gpt-image-3"


def test_newest_openai_image_model_no_match_returns_none():
    assert mr.newest_openai_image_model([{"id": "gpt-5.1", "created": 1}]) is None


# ---------------------------------------------------------------------------
# Gemini discovery ranking
# ---------------------------------------------------------------------------


def test_newest_gemini_image_model_picks_highest_version():
    models = [
        {
            "name": "models/gemini-3.1-flash-image",
            "supportedGenerationMethods": ["generateContent"],
        },
        {
            "name": "models/gemini-3.2-flash-image",
            "supportedGenerationMethods": ["generateContent"],
        },
    ]
    assert mr.newest_gemini_image_model(models, family="flash") == "gemini-3.2-flash-image"


def test_newest_gemini_image_model_a_newer_model_appears():
    models = [
        {
            "name": "models/gemini-3.1-flash-image",
            "supportedGenerationMethods": ["generateContent"],
        },
        {
            "name": "models/gemini-4.0-flash-image",
            "supportedGenerationMethods": ["generateContent"],
        },
    ]
    assert mr.newest_gemini_image_model(models, family="flash") == "gemini-4.0-flash-image"


def test_newest_gemini_image_model_respects_family():
    models = [
        {
            "name": "models/gemini-3.1-flash-image",
            "supportedGenerationMethods": ["generateContent"],
        },
        {
            "name": "models/gemini-3.1-flash-lite-image",
            "supportedGenerationMethods": ["generateContent"],
        },
        {"name": "models/gemini-3-pro-image", "supportedGenerationMethods": ["generateContent"]},
    ]
    assert mr.newest_gemini_image_model(models, family="flash") == "gemini-3.1-flash-image"
    assert (
        mr.newest_gemini_image_model(models, family="flash-lite") == "gemini-3.1-flash-lite-image"
    )
    assert mr.newest_gemini_image_model(models, family="pro") == "gemini-3-pro-image"


def test_newest_gemini_image_model_filters_unsupported_method():
    models = [
        {"name": "models/gemini-3.1-flash-image", "supportedGenerationMethods": ["embedContent"]}
    ]
    assert mr.newest_gemini_image_model(models, family="flash") is None


# ---------------------------------------------------------------------------
# resolve_model: explicit / config / discovery / fallback
# ---------------------------------------------------------------------------


async def test_resolve_model_explicit_choice_passes_through_unknown():
    """Requirement: unknown new model strings are never rejected here."""
    result = await mr.resolve_model("openai", "gpt-image-99-nobody-has-heard-of")
    assert result == "gpt-image-99-nobody-has-heard-of"


async def test_resolve_model_explicit_choice_resolves_alias(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text("aliases:\n  fast: gemini-3.1-flash-lite-image\n")
    result = await mr.resolve_model("gemini", "fast", config_path=config_path)
    assert result == "gemini-3.1-flash-lite-image"


async def test_resolve_model_config_literal_pin(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text("openai: gpt-image-1\n")
    result = await mr.resolve_model("openai", None, config_path=config_path)
    assert result == "gpt-image-1"


@respx.mock
async def test_resolve_model_discovery_selects_newer_model():
    """'A newer model appears -> it is selected', via the full resolve path."""
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-image-2", "created": 100},
                    {"id": "gpt-image-3", "created": 200},
                ]
            },
        )
    )
    result = await mr.resolve_model("openai", "latest", api_key="sk-test")
    assert result == "gpt-image-3"


@respx.mock
async def test_resolve_model_discovery_failure_falls_back_to_cache(tmp_path):
    """'Discovery fails -> fallback', preferring a prior cached success."""
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"openai": {"model": "gpt-image-2", "ts": 1_700_000_000.0}}))

    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    result = await mr.resolve_model("openai", "latest", api_key="sk-test")
    assert result == "gpt-image-2"


@respx.mock
async def test_resolve_model_discovery_failure_no_cache_falls_back_to_hardcoded():
    respx.get("https://api.openai.com/v1/models").mock(side_effect=httpx.ConnectError("no route"))
    result = await mr.resolve_model("openai", "latest", api_key="sk-test")
    assert result == "gpt-image-2"  # _HARDCODED_FALLBACK["openai"]


async def test_resolve_model_no_api_key_falls_back_to_hardcoded():
    result = await mr.resolve_model("gemini", "latest", api_key=None)
    assert result == "gemini-3.1-flash-image"


def test_resolve_model_rejects_unknown_provider():
    import asyncio

    with pytest.raises(ValueError, match="Unknown provider"):
        asyncio.run(mr.resolve_model("bogus", None))


# ---------------------------------------------------------------------------
# Disk cache round-trip
# ---------------------------------------------------------------------------


@respx.mock
async def test_successful_discovery_is_cached_for_next_call(tmp_path):
    cache_path = tmp_path / "cache.json"
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-image-2", "created": 1}]})
    )
    await mr.resolve_model("openai", "latest", api_key="sk-test")
    assert cache_path.is_file()
    cached = json.loads(cache_path.read_text())
    assert cached["openai"]["model"] == "gpt-image-2"


# ---------------------------------------------------------------------------
# Post-rename path fallback (imagen-mcp -> imagen)
#
# The autouse `_isolated_paths` fixture above sets the env var overrides for
# every other test in this file; these tests explicitly unset them to
# exercise the actual default (non-override) path resolution, which is where
# the pre-rename fallback logic lives.
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_home(tmp_path, monkeypatch):
    """Unset the config/cache env overrides and point Path.home() at a temp dir."""
    monkeypatch.delenv("IMAGEN_MCP_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("IMAGEN_MCP_DISCOVERY_CACHE", raising=False)
    monkeypatch.setattr(mr.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_default_config_path_prefers_new_location_when_both_exist(_fake_home):
    new_path = _fake_home / ".config" / "imagen" / "models.yaml"
    old_path = _fake_home / ".config" / "imagen-mcp" / "models.yaml"
    new_path.parent.mkdir(parents=True)
    old_path.parent.mkdir(parents=True)
    new_path.write_text("openai: latest\n")
    old_path.write_text("openai: latest\n")
    assert mr.default_config_path() == new_path


def test_default_config_path_falls_back_to_old_location(_fake_home):
    old_path = _fake_home / ".config" / "imagen-mcp" / "models.yaml"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("openai: gpt-image-1\n")
    assert mr.default_config_path() == old_path
    # And the fallback is actually readable end-to-end.
    assert mr.load_config() == {"openai": "gpt-image-1"}


def test_default_config_path_prefers_new_location_when_neither_exists(_fake_home):
    new_path = _fake_home / ".config" / "imagen" / "models.yaml"
    assert mr.default_config_path() == new_path


def test_cache_read_path_falls_back_to_old_location(_fake_home):
    old_path = _fake_home / ".cache" / "imagen-mcp" / "model_discovery_cache.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps({"openai": {"model": "gpt-image-1", "ts": 1_700_000_000.0}}))
    assert mr._cache_read_path() == old_path
    assert mr._read_cache()["openai"]["model"] == "gpt-image-1"


def test_cache_read_path_prefers_new_location_when_both_exist(_fake_home):
    new_path = _fake_home / ".cache" / "imagen" / "model_discovery_cache.json"
    old_path = _fake_home / ".cache" / "imagen-mcp" / "model_discovery_cache.json"
    new_path.parent.mkdir(parents=True)
    old_path.parent.mkdir(parents=True)
    new_path.write_text(json.dumps({"openai": {"model": "gpt-image-2", "ts": 1_700_000_000.0}}))
    old_path.write_text(json.dumps({"openai": {"model": "gpt-image-1", "ts": 1_700_000_000.0}}))
    assert mr._cache_read_path() == new_path
    assert mr._read_cache()["openai"]["model"] == "gpt-image-2"


def test_cache_write_path_always_targets_new_location_even_if_old_exists(_fake_home):
    old_path = _fake_home / ".cache" / "imagen-mcp" / "model_discovery_cache.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps({"openai": {"model": "gpt-image-1", "ts": 1_700_000_000.0}}))

    new_path = _fake_home / ".cache" / "imagen" / "model_discovery_cache.json"
    assert mr._cache_write_path() == new_path

    mr._write_cache({"gemini": {"model": "gemini-3.1-flash-image", "ts": 1_700_000_001.0}})
    assert new_path.is_file()
    # The old file is untouched by the write.
    old_contents = json.loads(old_path.read_text())
    assert old_contents == {"openai": {"model": "gpt-image-1", "ts": 1_700_000_000.0}}
