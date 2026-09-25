"""Tests for MCP server."""

import json
import os

import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


class TestServerImports:
    """Tests for server module imports."""

    def test_server_imports(self):
        """Server module should import without errors."""
        from src.server import mcp

        assert mcp is not None

    def test_config_imports(self):
        """Config modules should import without errors."""
        from src.config.constants import (
            DEFAULT_OPENAI_IMAGE_MODEL,
            GEMINI_SIZES,
            OPENAI_QUALITY_OPTIONS,
            OPENAI_SIZES,
        )
        from src.config.settings import get_settings

        # gpt-image-2 supports constrained arbitrary sizes; these are examples.
        assert len(OPENAI_SIZES) >= 6
        assert "1024x1024" in OPENAI_SIZES
        assert "2560x1440" in OPENAI_SIZES
        assert "3840x2160" in OPENAI_SIZES
        assert DEFAULT_OPENAI_IMAGE_MODEL == "gpt-image-2"
        assert "high" in OPENAI_QUALITY_OPTIONS
        assert len(GEMINI_SIZES) == 4
        assert get_settings() is not None

    def test_provider_imports(self):
        """Provider modules should import without errors."""
        from src.providers import (
            ImageProvider,
            ImageResult,
            get_provider_registry,
        )

        assert ImageProvider is not None
        assert ImageResult is not None
        assert get_provider_registry() is not None

    def test_model_imports(self):
        """Model modules should import without errors."""
        from src.models import ConversationalImageInput, ImageGenerationInput

        assert ImageGenerationInput is not None
        assert ConversationalImageInput is not None

    def test_canonical_package_imports(self):
        """The installed package name should expose the server and subpackages."""
        from imagen_mcp.providers.openai_provider import OpenAIProvider

        from imagen_mcp import __version__
        from imagen_mcp.server import mcp

        assert __version__ == "0.5.0"
        assert OpenAIProvider is not None
        assert mcp is not None

    def test_secret_overrides_remain_programmatically_usable(self):
        """Schema hiding must not break trusted direct callers."""
        from src.models.input_models import ImageGenerationInput

        params = ImageGenerationInput(
            prompt="x", openai_api_key="openai-secret", gemini_api_key="gemini-secret"
        )
        assert params.openai_api_key == "openai-secret"
        assert params.gemini_api_key == "gemini-secret"
        assert "openai_api_key" not in params.model_dump()
        assert "gemini_api_key" not in params.model_dump()

    async def test_mcp_tool_schemas_do_not_expose_secret_fields(self):
        """FastMCP tools/list must never advertise request-scoped credentials."""
        from imagen_mcp.server import mcp

        schemas = json.dumps([tool.inputSchema for tool in await mcp.list_tools()])
        assert "openai_api_key" not in schemas
        assert "gemini_api_key" not in schemas


class TestSettings:
    """Tests for settings configuration."""

    def test_settings_from_env(self):
        """Settings should load from environment."""
        from src.config.settings import Settings

        settings = Settings.from_env()
        assert settings.default_provider == "auto"
        assert settings.default_openai_size == "1024x1024"
        assert settings.default_gemini_size == "1K"

    def test_settings_has_keys(self):
        """Settings should detect API keys."""
        from src.config.settings import get_settings

        settings = get_settings()
        # With test keys set, both should be available
        assert settings.has_openai_key()
        assert settings.has_gemini_key()

    def test_available_providers(self):
        """Settings should list available providers."""
        from src.config.settings import get_settings

        settings = get_settings()
        providers = settings.available_providers()
        assert "openai" in providers
        assert "gemini" in providers


class TestInputModels:
    """Tests for Pydantic input models."""

    def test_image_generation_input(self):
        """ImageGenerationInput should validate correctly."""
        from src.models import ImageGenerationInput

        # Valid input
        input_data = ImageGenerationInput(prompt="A sunset over mountains")
        assert input_data.prompt == "A sunset over mountains"
        assert input_data.provider is not None
        assert input_data.provider.value == "auto"  # default is auto
        assert input_data.size is None  # default

    def test_image_generation_with_provider(self):
        """ImageGenerationInput should accept provider."""
        from src.models import ImageGenerationInput
        from src.models.input_models import Provider

        input_data = ImageGenerationInput(prompt="A sunset", provider=Provider("gemini"), size="4K")
        assert input_data.provider == Provider("gemini")
        assert input_data.size == "4K"

    def test_conversational_input(self):
        """ConversationalImageInput should validate correctly."""
        from src.models import ConversationalImageInput

        input_data = ConversationalImageInput(
            prompt="Make it more colorful", conversation_id="test-123"
        )
        assert input_data.prompt == "Make it more colorful"
        assert input_data.conversation_id == "test-123"

    def test_conversational_input_rejects_unknown_fields(self):
        from pydantic import ValidationError

        from src.models import ConversationalImageInput

        with pytest.raises(ValidationError, match="conversationd_id"):
            ConversationalImageInput(prompt="Refine", conversationd_id="typo")
