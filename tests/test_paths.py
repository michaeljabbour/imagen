"""Tests for output path handling utilities."""

import os
import stat
from pathlib import Path

import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+9pQAAAAASUVORK5CYII="
)


def test_resolve_output_path_expands_user_directory(tmp_path: Path, monkeypatch):
    from src.config.paths import resolve_output_path

    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_output_path("~/images/", default_filename="test.png")
    assert resolved == tmp_path / "images" / "test.png"
    assert resolved.parent.is_dir()
    assert stat.S_IMODE(resolved.parent.stat().st_mode) == 0o700


def test_resolve_output_path_existing_directory_with_suffix_is_directory(
    tmp_path: Path,
):
    from src.config.paths import resolve_output_path

    output_dir = tmp_path / "my.images"
    output_dir.mkdir()

    resolved = resolve_output_path(str(output_dir), default_filename="test.png")
    assert resolved == output_dir / "test.png"


def test_default_output_directory_uses_output_dir_env(tmp_path: Path, monkeypatch):
    from src.config.paths import get_base_output_directory, get_provider_output_directory
    from src.config.settings import get_settings

    output_dir = tmp_path / "custom-output"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    get_settings.cache_clear()

    resolved = get_base_output_directory()
    assert resolved == output_dir
    assert resolved.is_dir()
    assert stat.S_IMODE(resolved.stat().st_mode) == 0o700

    provider_dir = get_provider_output_directory("openai")
    assert provider_dir == output_dir / "openai"
    assert provider_dir.is_dir()
    assert stat.S_IMODE(provider_dir.stat().st_mode) == 0o700


async def test_provider_save_image_uses_output_dir_env(tmp_path: Path, monkeypatch):
    from src.config.settings import get_settings
    from src.providers.openai_provider import OpenAIProvider

    output_dir = tmp_path / "custom-output"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    get_settings.cache_clear()

    provider = OpenAIProvider()
    saved_path = await provider._save_image(TINY_PNG_BASE64, "Test prompt")
    assert saved_path.parent == output_dir / "openai"
    assert saved_path.is_file()
    assert stat.S_IMODE(saved_path.stat().st_mode) == 0o600


async def test_provider_save_image_output_path_directory_creates_file(tmp_path: Path):
    from src.providers.gemini_provider import GeminiProvider

    output_dir = tmp_path / "outputs"
    provider = GeminiProvider()
    saved_path = await provider._save_image(
        TINY_PNG_BASE64, "Test prompt", output_path=str(output_dir)
    )

    assert saved_path.parent == output_dir
    assert output_dir.is_dir()
    assert saved_path.is_file()
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved_path.stat().st_mode) == 0o600


def test_resolve_output_path_preserves_existing_directory_mode(tmp_path: Path):
    from src.config.paths import resolve_output_path

    output_dir = tmp_path / "shared-output"
    output_dir.mkdir(mode=0o755)
    output_dir.chmod(0o755)

    resolved = resolve_output_path(str(output_dir), default_filename="test.png")

    assert resolved == output_dir / "test.png"
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o755


async def test_provider_save_image_refuses_existing_file(tmp_path: Path):
    from src.providers.openai_provider import OpenAIProvider

    destination = tmp_path / "existing.png"
    destination.write_bytes(b"original")
    provider = OpenAIProvider()

    with pytest.raises(FileExistsError, match="File exists"):
        await provider._save_image(
            TINY_PNG_BASE64,
            "Test prompt",
            output_path=str(destination),
        )

    assert destination.read_bytes() == b"original"
