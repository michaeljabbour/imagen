"""Tests for list_gemini_models, the cost no-provider branch, and main()."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx
from mcp.server.fastmcp.exceptions import ToolError

from src import server
from src.models.input_models import CostEstimateInput, OutputFormat, Provider

MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class TestListGeminiModels:
    @pytest.mark.asyncio
    @respx.mock
    async def test_lists_image_models(self):
        respx.get(MODELS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3-pro-image",
                            "supportedGenerationMethods": ["generateContent"],
                            "description": "Nano Banana Pro image model",
                        },
                        {
                            "name": "models/gemini-1.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                            "description": "text only",
                        },
                    ]
                },
            )
        )
        out = await server.list_gemini_models()
        assert "gemini-3-pro-image" in out
        # Non-image model filtered out.
        assert "gemini-1.5-flash" not in out

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_image_models(self):
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"models": []}))
        out = await server.list_gemini_models()
        assert "No Image Models Found" in out

    @pytest.mark.asyncio
    async def test_no_api_key(self, monkeypatch):
        from src.config.settings import get_settings

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        get_settings.cache_clear()

        with pytest.raises(ToolError, match="No Gemini API key configured"):
            await server.list_gemini_models()

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_handled(self):
        respx.get(MODELS_URL).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(ToolError, match="Server error '500 Internal Server Error'"):
            await server.list_gemini_models()


class TestEstimateCostNoProvider:
    @pytest.mark.asyncio
    async def test_estimate_without_keys_still_works(self, monkeypatch):
        from src.config.settings import get_settings

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        get_settings.cache_clear()

        out = await server.estimate_cost(
            CostEstimateInput(
                prompt="A poster",
                provider=Provider.OPENAI,
                quality="medium",
                output_format=OutputFormat.JSON,
            )
        )
        import json

        data = json.loads(out)
        assert data["provider"] == "openai"
        assert data["total_usd"] is not None


class TestMain:
    def test_main_stdio_default(self, monkeypatch):
        run = MagicMock()
        monkeypatch.setattr(server.mcp, "run", run)
        monkeypatch.setattr(server, "get_settings", server.get_settings)
        monkeypatch.setattr("src.config.dotenv.load_dotenv", lambda **kw: {})
        monkeypatch.delenv("IMAGEN_MCP_TRANSPORT", raising=False)

        server.main()
        run.assert_called_once_with()

    def test_main_streamable_http(self, monkeypatch):
        run = MagicMock()
        monkeypatch.setattr(server.mcp, "run", run)
        monkeypatch.setattr("src.config.dotenv.load_dotenv", lambda **kw: {})
        monkeypatch.setenv("IMAGEN_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("IMAGEN_MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("IMAGEN_MCP_PORT", "9000")
        monkeypatch.setenv("IMAGEN_MCP_ALLOW_INSECURE_REMOTE", "true")

        server.main()

        run.assert_called_once_with(transport="streamable-http")
        assert server.mcp.settings.host == "0.0.0.0"
        assert server.mcp.settings.port == 9000

    def test_main_rejects_unauthenticated_remote_bind(self, monkeypatch):
        monkeypatch.setattr("src.config.dotenv.load_dotenv", lambda **kw: {})
        monkeypatch.setenv("IMAGEN_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("IMAGEN_MCP_HOST", "0.0.0.0")
        monkeypatch.delenv("IMAGEN_MCP_ALLOW_INSECURE_REMOTE", raising=False)

        with pytest.raises(RuntimeError, match="Refusing non-loopback"):
            server.main()

    def test_main_rejects_unknown_transport(self, monkeypatch):
        monkeypatch.setattr("src.config.dotenv.load_dotenv", lambda **kw: {})
        monkeypatch.setenv("IMAGEN_MCP_TRANSPORT", "websocket")

        with pytest.raises(ValueError, match="Unsupported IMAGEN_MCP_TRANSPORT"):
            server.main()
