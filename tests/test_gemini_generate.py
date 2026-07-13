"""Tests for GeminiProvider.generate_image with a mocked google-genai SDK."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.genai import errors as genai_errors
from PIL import Image as PILImage

from src.providers import gemini_provider
from src.providers.gemini_provider import GeminiProvider


class FakePart:
    """Mimics a google-genai response part."""

    def __init__(
        self, *, data: bytes | None = None, text: str | None = None, thought: bool = False
    ):
        self.thought = thought
        self.inline_data = MagicMock(data=data) if data is not None else None
        self.text = text


class FakeResponse:
    def __init__(self, parts):
        self.parts = parts


@pytest.fixture
def mocked_sdk(monkeypatch):
    """Stub out the lazily-imported google-genai globals."""
    fake_genai = MagicMock()
    fake_types = MagicMock()

    monkeypatch.setattr(gemini_provider, "genai", fake_genai)
    monkeypatch.setattr(gemini_provider, "types", fake_types)
    monkeypatch.setattr(gemini_provider, "Image", PILImage)
    # Prevent _import_dependencies from overwriting our stubs.
    monkeypatch.setattr(gemini_provider, "_import_dependencies", lambda: None)
    # Avoid real backoff sleeps in error paths.
    import src.providers.base as base

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)
    return fake_genai


def _set_response(fake_genai, response):
    client = MagicMock()
    client.models.generate_content.return_value = response
    fake_genai.Client.return_value = client
    return client


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _encoded_image_bytes(image_format: str) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), "blue").save(buffer, format=image_format)
    return buffer.getvalue()


class TestGeminiGenerate:
    def test_reference_decoder_enforces_dimension_and_format_limits(self, monkeypatch):
        monkeypatch.setattr(gemini_provider, "Image", PILImage)
        monkeypatch.setattr(gemini_provider, "_MAX_REFERENCE_DIMENSION", 2)
        buffer = io.BytesIO()
        PILImage.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(ValueError, match="8192px dimension limit"):
            provider._decode_images(None, [encoded])

        gif_buffer = io.BytesIO()
        PILImage.new("RGB", (1, 1), "blue").save(gif_buffer, format="GIF")
        gif_encoded = base64.b64encode(gif_buffer.getvalue()).decode()
        monkeypatch.setattr(gemini_provider, "_MAX_REFERENCE_DIMENSION", 8192)
        with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
            provider._decode_images(None, [gif_encoded])

    def test_reference_decoder_enforces_aggregate_byte_limit(self, monkeypatch):
        monkeypatch.setattr(gemini_provider, "Image", PILImage)
        buffer = io.BytesIO()
        PILImage.new("RGB", (2, 2), "blue").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode()
        monkeypatch.setattr(
            gemini_provider,
            "_MAX_REFERENCE_TOTAL_BYTES",
            len(image_bytes) + 1,
        )
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(ValueError, match="aggregate limit"):
            provider._decode_images(None, [encoded, encoded])

    @pytest.mark.asyncio
    async def test_single_image_success(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image(
            "A photorealistic portrait", size="2K", output_path=str(tmp_path)
        )

        assert result.success is True
        assert result.provider == "gemini"
        assert result.image_path is not None
        assert result.image_path.is_file()
        assert result.additional_paths is None
        assert result.output_format == "png"
        with PILImage.open(result.image_path) as saved:
            assert saved.format == "PNG"

    @pytest.mark.asyncio
    async def test_jpeg_response_is_validated_and_normalized_to_png(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse([FakePart(data=_encoded_image_bytes("JPEG"))]),
        )
        destination = tmp_path / "gemini-output.png"

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A blue square", output_path=str(destination))

        assert result.success is True
        assert result.image_path == destination
        assert result.output_format == "png"
        with PILImage.open(destination) as saved:
            assert saved.format == "PNG"
            assert saved.size == (8, 8)

    @pytest.mark.asyncio
    async def test_invalid_response_image_fails_without_writing(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=b"not-an-image")]))
        destination = tmp_path / "invalid.png"

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(destination))

        assert result.success is False
        assert result.error == "Gemini returned invalid image data for artifact 1."
        assert not destination.exists()

    @pytest.mark.asyncio
    async def test_one_shot_generation_does_not_persist_history(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")
        provider._store_conversation_turn = AsyncMock()

        result = await provider.generate_image("One shot", output_path=str(tmp_path))

        assert result.success is True
        assert result.conversation_id is None
        provider._store_conversation_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_multiple_images_saved(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse([FakePart(data=PNG_BYTES), FakePart(data=PNG_BYTES)]),
        )

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("Two variants", size="1K", output_path=str(tmp_path))

        assert result.success is True
        assert result.additional_paths is not None
        assert len(result.additional_paths) == 1
        assert result.additional_paths[0].is_file()

    @pytest.mark.asyncio
    async def test_batch_explicit_file_is_safely_indexed(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse([FakePart(data=PNG_BYTES), FakePart(data=PNG_BYTES)]),
        )
        destination = tmp_path / "variant.png"

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("Two variants", output_path=str(destination))

        assert result.success is True
        assert result.image_path == destination
        assert result.additional_paths == [tmp_path / "variant_2.png"]

    @pytest.mark.asyncio
    async def test_invalid_reference_fails_before_generation(self, mocked_sdk, tmp_path):
        client = _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        result = await provider.generate_image(
            "A scene", reference_images=["not base64!"], output_path=str(tmp_path)
        )

        assert result.success is False
        assert "Reference image 1" in (result.error or "")
        client.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_conversation_image_fails_closed(self, mocked_sdk, tmp_path):
        client = _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")
        provider._get_last_image_from_conversation = AsyncMock(return_value=None)

        result = await provider.generate_image(
            "Refine it",
            conversation_id="missing-thread",
            output_path=str(tmp_path),
        )

        assert result.success is False
        assert "no prior image" in (result.error or "")
        client.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_corrupt_conversation_image_fails_closed(self, mocked_sdk, tmp_path):
        client = _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")
        provider._get_last_image_from_conversation = AsyncMock(return_value="not-base64")

        result = await provider.generate_image(
            "Refine it",
            conversation_id="corrupt-thread",
            output_path=str(tmp_path),
        )

        assert result.success is False
        assert "Conversation history image" in (result.error or "")
        client.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_and_thought_parts_ignored_for_image(self, mocked_sdk, tmp_path):
        _set_response(
            mocked_sdk,
            FakeResponse(
                [
                    FakePart(text="some reasoning", thought=True),
                    FakePart(text="caption"),
                    FakePart(data=PNG_BYTES),
                ]
            ),
        )

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is True
        assert result.image_path is not None

    def test_thought_content_is_reduced_to_safe_telemetry(self):
        provider = GeminiProvider(api_key="test-key")
        extraction = provider._extract_content(
            FakeResponse(
                [
                    FakePart(text="private chain of thought", thought=True),
                    FakePart(data=b"private-thought-image", thought=True),
                    FakePart(data=PNG_BYTES),
                ]
            )
        )

        serialized = str(extraction["thoughts"])
        assert "private chain of thought" not in serialized
        assert "private-thought-image" not in serialized
        assert extraction["thoughts"] == [
            {"type": "text", "index": 0, "character_count": 24},
            {"type": "image", "index": 1, "byte_length": 21},
        ]

    def test_grounding_extracts_search_suggestions_sources_and_citations(self):
        provider = GeminiProvider(api_key="test-key")
        response = FakeResponse([FakePart(data=PNG_BYTES)])
        response.candidates = [
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    search_entry_point=SimpleNamespace(
                        rendered_content="<div>Google Search Suggestions</div>"
                    ),
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(uri="https://example.com/source", title="Source")
                        )
                    ],
                    grounding_supports=[
                        SimpleNamespace(
                            segment=SimpleNamespace(text="grounded claim"),
                            grounding_chunk_indices=[0],
                        )
                    ],
                )
            )
        ]

        extraction = provider._extract_content(response)

        assert extraction["grounding_metadata"] == {
            "search_suggestions_html": "<div>Google Search Suggestions</div>",
            "sources": [{"uri": "https://example.com/source", "title": "Source"}],
            "citations": [{"text": "grounded claim", "source_indices": [0]}],
        }

    @pytest.mark.asyncio
    async def test_search_fails_closed_without_search_suggestions(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        result = await provider.generate_image(
            "Current weather",
            enable_google_search=True,
            output_path=str(tmp_path),
        )

        assert result.success is False
        assert "Search Suggestions" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_image_in_response_fails(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(text="only text")]))

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_api_exception_returns_failure(self, mocked_sdk, tmp_path):
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("API down")
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))
        assert result.success is False
        assert result.error == "Gemini image generation failed."

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 403])
    async def test_client_errors_are_not_retried(self, mocked_sdk, tmp_path, status_code):
        client = MagicMock()
        client.models.generate_content.side_effect = genai_errors.ClientError(
            status_code, {"error": {"message": "bad request"}}
        )
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is False
        assert client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_is_retried(self, mocked_sdk, tmp_path):
        client = MagicMock()
        client.models.generate_content.side_effect = genai_errors.ClientError(
            429, {"error": {"message": "rate limited"}}
        )
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is False
        assert client.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [500, 503])
    async def test_server_errors_are_retried(self, mocked_sdk, tmp_path, status_code):
        client = MagicMock()
        client.models.generate_content.side_effect = genai_errors.ServerError(
            status_code, {"error": {"message": "unavailable"}}
        )
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is False
        assert client.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_transport_errors_are_retried(self, mocked_sdk, tmp_path):
        client = MagicMock()
        request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
        client.models.generate_content.side_effect = httpx.ConnectError(
            "connection reset", request=request
        )
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is False
        assert client.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_is_not_retried(self, mocked_sdk, tmp_path):
        client = MagicMock()
        client.models.generate_content.side_effect = TimeoutError("render timed out")
        mocked_sdk.Client.return_value = client

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is False
        assert client.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_size_returns_failure(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate_image("A scene", size="8K", output_path=str(tmp_path))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_default_model_and_size_are_ga_and_1k(self, mocked_sdk, tmp_path):
        client = _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        result = await provider.generate_image("A scene", output_path=str(tmp_path))

        assert result.success is True
        assert result.model == "gemini-3.1-flash-image"
        assert result.size == "1K"
        assert client.models.generate_content.call_args.kwargs["model"] == (
            "gemini-3.1-flash-image"
        )

    @pytest.mark.asyncio
    async def test_flash_forwards_strict_thinking_level(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        result = await provider.generate_image(
            "A complex scene",
            thinking_level="high",
            output_path=str(tmp_path),
        )

        assert result.success is True
        gemini_provider.types.ThinkingConfig.assert_called_once_with(thinking_level="HIGH")

    @pytest.mark.asyncio
    async def test_pro_rejects_unexposed_thinking_level(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        result = await provider.generate_image(
            "A complex scene",
            model="gemini-3-pro-image",
            thinking_level="minimal",
            output_path=str(tmp_path),
        )

        assert result.success is False
        assert "provider-managed only" in (result.error or "")

    @pytest.mark.asyncio
    async def test_half_k_is_flash_only(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        flash = await provider.generate_image(
            "A scene", size="0.5K", output_path=str(tmp_path / "flash")
        )
        pro = await provider.generate_image(
            "A scene",
            size="0.5K",
            model="gemini-3-pro-image",
            output_path=str(tmp_path / "pro"),
        )

        assert flash.success is True
        assert pro.success is False
        assert "Supported sizes" in (pro.error or "")

    @pytest.mark.asyncio
    async def test_lite_is_1k_only_and_rejects_search(self, mocked_sdk, tmp_path):
        _set_response(mocked_sdk, FakeResponse([FakePart(data=PNG_BYTES)]))
        provider = GeminiProvider(api_key="test-key")

        success = await provider.generate_image(
            "A scene",
            model="gemini-3.1-flash-lite-image",
            size="1K",
            output_path=str(tmp_path / "success"),
        )
        bad_size = await provider.generate_image(
            "A scene",
            model="gemini-3.1-flash-lite-image",
            size="2K",
            output_path=str(tmp_path / "bad-size"),
        )
        bad_search = await provider.generate_image(
            "A scene",
            model="gemini-3.1-flash-lite-image",
            enable_google_search=True,
            output_path=str(tmp_path / "bad-search"),
        )

        assert success.success is True
        assert bad_size.success is False
        assert "Supported sizes: 1K" in (bad_size.error or "")
        assert bad_search.success is False
        assert "not supported" in (bad_search.error or "")
