"""Tests for OpenAIProvider generate/edit paths with mocked HTTP."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from src.exceptions import ProviderError
from src.providers.openai_provider import OpenAIProvider

SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _img_response(count: int = 1) -> dict:
    return {
        "data": [{"b64_json": SAMPLE_IMAGE_B64} for _ in range(count)],
        "usage": {"input_tokens": 5, "output_tokens": 100, "total_tokens": 105},
    }


class TestGenerateDirect:
    @pytest.mark.asyncio
    async def test_direct_path_success(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,  # direct /images/generations path
                output_path=str(tmp_path),
            )
        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.is_file()
        assert result.usage_tokens == {
            "input_tokens": 5,
            "output_tokens": 100,
            "total_tokens": 105,
        }
        # Direct path makes exactly one API call.
        assert mock_req.await_count == 1

    @pytest.mark.asyncio
    async def test_result_reports_actual_derived_aspect_ratio(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.generate_image(
                "A poster",
                aspect_ratio="4:3",
                enable_enhancement=False,
                output_path=str(tmp_path),
            )

        assert result.success is True
        assert result.size == "1408x1056"
        assert result.aspect_ratio == "4:3"

    @pytest.mark.asyncio
    async def test_batch_n_populates_additional_paths(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response(3))):
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, n=3, output_path=str(tmp_path)
            )
        assert result.success is True
        assert result.additional_paths is not None
        assert len(result.additional_paths) == 2

    @pytest.mark.asyncio
    async def test_batch_explicit_file_is_safely_indexed(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        destination = tmp_path / "variant.webp"
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response(3))):
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,
                n=3,
                openai_output_format="webp",
                output_path=str(destination),
            )

        assert result.success is True
        assert result.image_path == destination
        assert result.additional_paths == [
            tmp_path / "variant_2.webp",
            tmp_path / "variant_3.webp",
        ]
        assert all(path.is_file() for path in [result.image_path, *result.additional_paths])

    @pytest.mark.asyncio
    async def test_output_encoding_controls_generated_extension(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,
                openai_output_format="jpeg",
                output_path=str(tmp_path),
            )

        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.suffix == ".jpeg"

    @pytest.mark.asyncio
    async def test_mismatched_explicit_extension_fails_before_api(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,
                openai_output_format="jpeg",
                output_path=str(tmp_path / "poster.png"),
            )

        assert result.success is False
        assert "does not match" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existing_explicit_output_fails_before_api(self, tmp_path):
        destination = tmp_path / "poster.png"
        destination.write_bytes(b"keep")
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, output_path=str(destination)
            )

        assert result.success is False
        assert destination.read_bytes() == b"keep"
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_image_response_is_failure(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value={"data": []})):
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, output_path=str(tmp_path)
            )

        assert result.success is False
        assert "no usable image data" in (result.error or "")

    @pytest.mark.asyncio
    async def test_reference_images_are_rejected_before_api(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=False,
                reference_images=[SAMPLE_IMAGE_B64],
                output_path=str(tmp_path),
            )

        assert result.success is False
        assert "cannot apply reference_images" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_second_conversation_turn_uses_image_edits_wire_payload(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        provider._store_conversation_turn = AsyncMock()
        provider._get_last_image_from_conversation = AsyncMock(return_value=SAMPLE_IMAGE_B64)

        with patch.object(
            provider,
            "_make_api_request",
            AsyncMock(side_effect=[_img_response(), _img_response()]),
        ) as mock_req:
            first = await provider.generate_image(
                "A poster",
                enable_enhancement=False,
                persist_conversation=True,
                output_path=str(tmp_path / "first"),
            )
            second = await provider.generate_image(
                "Make the title blue",
                conversation_id=first.conversation_id,
                enable_enhancement=False,
                output_path=str(tmp_path / "second"),
            )

        assert first.success is True
        assert second.success is True
        assert second.conversation_id == first.conversation_id
        assert [call.kwargs["endpoint"] for call in mock_req.await_args_list] == [
            "/images/generations",
            "/images/edits",
        ]
        edit_call = mock_req.await_args_list[1].kwargs
        assert edit_call["data"]["prompt"] == "Make the title blue"
        assert edit_call["files"]["image"][1].startswith(b"\x89PNG")
        assert "json_data" not in edit_call

    @pytest.mark.asyncio
    async def test_one_shot_generation_does_not_persist_history(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        provider._store_conversation_turn = AsyncMock()
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.generate_image(
                "One shot", enable_enhancement=False, output_path=str(tmp_path)
            )

        assert result.success is True
        assert result.conversation_id is None
        provider._store_conversation_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_conversation_persistence_failure_is_not_reported_resumable(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        provider._store_conversation_turn = AsyncMock(side_effect=OSError("disk full"))
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.generate_image(
                "Conversation",
                enable_enhancement=False,
                persist_conversation=True,
                output_path=str(tmp_path),
            )

        assert result.success is False
        assert result.conversation_id is None

    @pytest.mark.asyncio
    async def test_api_error_returns_failure(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(side_effect=ValueError("OpenAI API error"))
        ):
            result = await provider.generate_image(
                "A poster", enable_enhancement=False, output_path=str(tmp_path)
            )
        assert result.success is False
        assert "OpenAI API error" in (result.error or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-5.1", "gpt-4o", "gpt-image-future"])
    async def test_non_image_models_fail_before_api_call(self, model, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.generate_image(
                "A poster",
                openai_model=model,
                enable_enhancement=False,
                output_path=str(tmp_path),
            )
        assert result.success is False
        assert "Unsupported OpenAI image model" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_assistant_model_fails_before_api_call(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.generate_image(
                "A poster",
                assistant_model="gpt-arbitrary",
                enable_enhancement=True,
                output_path=str(tmp_path),
            )
        assert result.success is False
        assert "Unsupported OpenAI assistant model" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ("{", "malformed JSON"),
            ("[]", "JSON object"),
            ('{"prompt": "   "}', "empty or invalid prompt"),
            (json.dumps({"prompt": "x" * 32_001}), "exceeds the OpenAI maximum"),
            ('{"prompt": "refined", "quality": "ultra"}', "Invalid quality"),
            ('{"prompt": "refined", "surprise": true}', "unsupported fields"),
        ],
    )
    async def test_enhancement_rejects_untrusted_tool_arguments_before_image_call(
        self, arguments, message, tmp_path
    ):
        provider = OpenAIProvider(api_key="k")
        chat_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "generate_image",
                                    "arguments": arguments,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=chat_response)
        ) as mock_req:
            result = await provider.generate_image(
                "A poster",
                enable_enhancement=True,
                output_path=str(tmp_path),
            )

        assert result.success is False
        assert message in (result.error or "")
        assert mock_req.await_count == 1


class TestValidateParams:
    @pytest.mark.asyncio
    async def test_accepts_openai_prompt_up_to_32000_characters(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x" * 32_000)
        assert len(validated["prompt"]) == 32_000

    @pytest.mark.asyncio
    async def test_rejects_openai_prompt_above_32000_characters(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="32000"):
            await provider.validate_params("x" * 32_001)

    @pytest.mark.asyncio
    async def test_uses_configured_default_size(self, monkeypatch):
        from src.config.settings import get_settings

        monkeypatch.setenv("DEFAULT_OPENAI_SIZE", "1536x1024")
        get_settings.cache_clear()
        provider = OpenAIProvider(api_key="k")

        validated = await provider.validate_params("x")

        assert validated["size"] == "1536x1024"

    @pytest.mark.asyncio
    async def test_rejects_invalid_background(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid background"):
            await provider.validate_params("x", background="rainbow")

    @pytest.mark.asyncio
    async def test_rejects_invalid_output_format(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid output_format"):
            await provider.validate_params("x", openai_output_format="gif")

    @pytest.mark.asyncio
    async def test_rejects_invalid_moderation(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid moderation"):
            await provider.validate_params("x", moderation="strict")

    @pytest.mark.asyncio
    async def test_rejects_n_out_of_range(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="n must be between"):
            await provider.validate_params("x", n=99)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("quality", ["standard", "hd"])
    async def test_rejects_dall_e_quality_values(self, quality):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid quality"):
            await provider.validate_params("x", quality=quality)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("style", ["vivid", "natural"])
    async def test_rejects_dall_e_style_parameter(self, style):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="only supported by DALL-E 3"):
            await provider.validate_params("x", style=style)

    @pytest.mark.asyncio
    async def test_gemini_size_mapped(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", size="4K")
        assert validated["size"] == "3840x2160"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("aspect_ratio", "expected_size"),
        [
            ("1:1", "1024x1024"),
            ("2:3", "1024x1536"),
            ("3:2", "1536x1024"),
            ("3:4", "1056x1408"),
            ("4:3", "1408x1056"),
            ("4:5", "1024x1280"),
            ("5:4", "1280x1024"),
            ("9:16", "1152x2048"),
            ("16:9", "2048x1152"),
            ("21:9", "2464x1056"),
            ("portrait", "1024x1536"),
            ("landscape", "1536x1024"),
            ("square", "1024x1024"),
        ],
    )
    async def test_aspect_ratio_to_exact_valid_size(self, aspect_ratio, expected_size):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", aspect_ratio=aspect_ratio)
        assert validated["size"] == expected_size

    @pytest.mark.asyncio
    @pytest.mark.parametrize("aspect_ratio", ["1:4", "8:1", "banana", "0:1"])
    async def test_rejects_unsupported_aspect_ratio_without_square_fallback(self, aspect_ratio):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="Unsupported OpenAI aspect_ratio"):
            await provider.validate_params("x", aspect_ratio=aspect_ratio)

    @pytest.mark.asyncio
    async def test_rejects_conflicting_explicit_size_and_aspect_ratio(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="conflicts with aspect_ratio"):
            await provider.validate_params("x", size="1024x1024", aspect_ratio="16:9")

    @pytest.mark.asyncio
    async def test_accepts_consistent_explicit_size_and_aspect_ratio(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", size="2560x1440", aspect_ratio="16:9")
        assert validated["size"] == "2560x1440"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "size",
        ["1024x640", "2048x1024", "2560x1440", "2880x2880", "3840x2160"],
    )
    async def test_gpt_image_2_accepts_constrained_custom_sizes(self, size):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params("x", size=size)
        assert validated["size"] == size

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("size", "message"),
        [
            ("1025x1024", "multiples of 16"),
            ("3856x2160", "at most 3840"),
            ("3088x1024", "3:1"),
            ("512x512", "total pixels"),
            ("3008x3008", "total pixels"),
        ],
    )
    async def test_gpt_image_2_rejects_out_of_contract_sizes(self, size, message):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match=message):
            await provider.validate_params("x", size=size)

    @pytest.mark.asyncio
    async def test_gpt_image_2_rejects_transparent_background(self):
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(ValueError, match="does not support transparent"):
            await provider.validate_params("x", background="transparent")

    @pytest.mark.asyncio
    async def test_legacy_model_keeps_enumerated_sizes_and_transparency(self):
        provider = OpenAIProvider(api_key="k")
        validated = await provider.validate_params(
            "x",
            size="1024x1536",
            background="transparent",
            model="gpt-image-1.5",
        )
        assert validated["size"] == "1024x1536"
        assert validated["background"] == "transparent"
        with pytest.raises(ValueError, match="Invalid size"):
            await provider.validate_params("x", size="2048x1024", model="gpt-image-1.5")


class TestEditImage:
    @pytest.mark.asyncio
    async def test_edit_success(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)

        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value=_img_response())):
            result = await provider.edit_image(
                prompt="change the sky to sunset",
                image_path=str(src),
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.is_file()

    @pytest.mark.asyncio
    async def test_gpt_image_2_edit_omits_input_fidelity(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.edit_image(
                prompt="x",
                image_path=str(src),
                size="2048x1024",
                input_fidelity="low",
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        assert mock_req.call_args.kwargs["data"]["size"] == "2048x1024"
        assert "input_fidelity" not in mock_req.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_legacy_edit_preserves_input_fidelity(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)
        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.edit_image(
                prompt="x",
                image_path=str(src),
                openai_model="gpt-image-1.5",
                background="transparent",
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        assert mock_req.call_args.kwargs["data"]["input_fidelity"] == "high"
        assert mock_req.call_args.kwargs["data"]["background"] == "transparent"

    @pytest.mark.asyncio
    async def test_edit_with_mask(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)
        mask = tmp_path / "mask.png"
        Image.new("L", (4, 4), 255).save(mask)

        provider = OpenAIProvider(api_key="k")
        with patch.object(
            provider, "_make_api_request", AsyncMock(return_value=_img_response())
        ) as mock_req:
            result = await provider.edit_image(
                prompt="inpaint",
                image_path=str(src),
                mask_path=str(mask),
                output_path=str(tmp_path / "out"),
            )
        assert result.success is True
        # The multipart files dict should include the mask.
        _, kwargs = mock_req.call_args
        assert "mask" in kwargs["files"]

    @pytest.mark.asyncio
    async def test_edit_missing_source_fails(self, tmp_path):
        provider = OpenAIProvider(api_key="k")
        result = await provider.edit_image(prompt="x", image_path=str(tmp_path / "missing.png"))
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_edit_rejects_input_outside_allowed_roots_before_api(self, tmp_path):
        secret = tmp_path.parent / f"{tmp_path.name}-secret.png"
        Image.new("RGB", (4, 4), "red").save(secret)
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.edit_image(prompt="x", image_path=str(secret))

        assert result.success is False
        assert "outside IMAGEN_MCP_ALLOWED_INPUT_ROOTS" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_rejects_non_image_bytes_before_api(self, tmp_path):
        source = tmp_path / "fake.png"
        source.write_text("not an image")
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.edit_image(prompt="x", image_path=str(source))

        assert result.success is False
        assert "not a valid image" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_rejects_mask_dimension_mismatch_before_api(self, tmp_path):
        source = tmp_path / "source.png"
        mask = tmp_path / "mask.png"
        Image.new("RGB", (8, 8), "blue").save(source)
        Image.new("L", (4, 4), 255).save(mask)
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.edit_image(
                prompt="x", image_path=str(source), mask_path=str(mask)
            )

        assert result.success is False
        assert "dimensions must match" in (result.error or "")
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_refuses_to_overwrite_source_before_api(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)
        original = src.read_bytes()
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock()) as mock_req:
            result = await provider.edit_image(
                prompt="x", image_path=str(src), output_path=str(src)
            )

        assert result.success is False
        assert "must not overwrite" in (result.error or "")
        assert src.read_bytes() == original
        mock_req.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_edit_response_is_failure(self, tmp_path):
        src = tmp_path / "in.png"
        Image.new("RGB", (4, 4), "blue").save(src)
        provider = OpenAIProvider(api_key="k")
        with patch.object(provider, "_make_api_request", AsyncMock(return_value={"data": []})):
            result = await provider.edit_image(
                prompt="x", image_path=str(src), output_path=str(tmp_path / "out")
            )

        assert result.success is False
        assert "no usable edited image data" in (result.error or "")

    @pytest.mark.asyncio
    async def test_edit_invalid_size_fails(self, tmp_path):
        src = tmp_path / "in.png"
        src.write_bytes(b"fake")
        provider = OpenAIProvider(api_key="k")
        result = await provider.edit_image(
            prompt="x",
            image_path=str(src),
            size="3856x2160",  # exceeds the inclusive max edge
        )
        assert result.success is False
        assert "at most 3840" in (result.error or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("quality", ["standard", "hd"])
    async def test_edit_rejects_dall_e_quality_values(self, quality, tmp_path):
        src = tmp_path / "in.png"
        src.write_bytes(b"fake")
        provider = OpenAIProvider(api_key="k")
        result = await provider.edit_image(
            prompt="x",
            image_path=str(src),
            quality=quality,
        )
        assert result.success is False
        assert "Invalid quality" in (result.error or "")


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_is_safe_without_client(self):
        provider = OpenAIProvider(api_key="k")
        await provider.close()  # no client created yet — must not raise

    def test_ensure_client_reused(self):
        provider = OpenAIProvider(api_key="k")
        c1 = provider._ensure_client()
        c2 = provider._ensure_client()
        assert c1 is c2

    def test_client_uses_configured_timeout(self, monkeypatch):
        from src.config.settings import get_settings

        monkeypatch.setenv("REQUEST_TIMEOUT", "300")
        get_settings.cache_clear()

        provider = OpenAIProvider(api_key="k")
        client = provider._ensure_client()
        # Generous read ceiling, short connect timeout.
        assert client.timeout.read == 300.0
        assert client.timeout.connect == 10.0


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_timeout_is_not_retried(self):
        provider = OpenAIProvider(api_key="k")
        calls = 0

        async def boom():
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("render too slow")

        with pytest.raises(httpx.ReadTimeout):
            await provider._retry_with_backoff(boom, non_retryable=(httpx.TimeoutException,))
        assert calls == 1  # failed once, no ×3 retry storm

    @pytest.mark.asyncio
    async def test_other_errors_still_retried(self, monkeypatch):
        import src.providers.base as base

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)

        provider = OpenAIProvider(api_key="k")
        calls = 0

        async def boom():
            nonlocal calls
            calls += 1
            raise ValueError("transient")

        with pytest.raises(ValueError):
            await provider._retry_with_backoff(
                boom, max_retries=3, non_retryable=(httpx.TimeoutException,)
            )
        assert calls == 3  # retried up to max_retries

    @pytest.mark.asyncio
    async def test_retry_log_never_exposes_exception_secret(self, monkeypatch, caplog):
        import src.providers.base as base

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)
        provider = OpenAIProvider(api_key="k")
        secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"

        async def boom():
            raise ValueError(f"transient failure with {secret}")

        with pytest.raises(ValueError):
            await provider._retry_with_backoff(boom, max_retries=2)

        assert secret not in caplog.text
        assert "ValueError" in caplog.text

    @pytest.mark.asyncio
    async def test_conversation_read_failure_propagates_instead_of_looking_absent(self):
        provider = OpenAIProvider(api_key="k")
        store = MagicMock()
        store.get_last_image.side_effect = sqlite3.DatabaseError("database is corrupt")

        with patch("src.services.conversation_store.get_conversation_store", return_value=store):
            with pytest.raises(sqlite3.DatabaseError, match="database is corrupt"):
                await provider._get_last_image_from_conversation("conversation-1")

    @pytest.mark.asyncio
    async def test_make_api_request_maps_timeout_without_retrying(self, monkeypatch):
        import src.providers.base as base

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)

        provider = OpenAIProvider(api_key="k")
        post_calls = 0

        class _Client:
            is_closed = False

            async def post(self, *a, **k):
                nonlocal post_calls
                post_calls += 1
                raise httpx.ReadTimeout("slow")

        monkeypatch.setattr(provider, "_ensure_client", lambda: _Client())

        with pytest.raises(ProviderError, match="timed out"):
            await provider._make_api_request("/images/generations", "k", json_data={})
        assert post_calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "expected_calls"),
        [(400, 1), (403, 1), (422, 1), (429, 3), (500, 3), (503, 3)],
    )
    async def test_http_retry_classification(self, monkeypatch, status_code, expected_calls):
        import src.providers.base as base

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)
        provider = OpenAIProvider(api_key="k")
        post_calls = 0

        class _Client:
            is_closed = False

            async def post(self, *args, **kwargs):
                nonlocal post_calls
                post_calls += 1
                request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
                return httpx.Response(status_code, request=request, text="failure")

        monkeypatch.setattr(provider, "_ensure_client", lambda: _Client())

        with pytest.raises(ProviderError):
            await provider._make_api_request("/images/generations", "k", json_data={})
        assert post_calls == expected_calls

    @pytest.mark.asyncio
    async def test_network_transport_error_is_retried(self, monkeypatch):
        import src.providers.base as base

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(base.asyncio, "sleep", _no_sleep)
        provider = OpenAIProvider(api_key="k")
        post_calls = 0

        class _Client:
            is_closed = False

            async def post(self, *args, **kwargs):
                nonlocal post_calls
                post_calls += 1
                request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
                raise httpx.ConnectError("connection reset", request=request)

        monkeypatch.setattr(provider, "_ensure_client", lambda: _Client())

        with pytest.raises(ProviderError, match="transport failure"):
            await provider._make_api_request("/images/generations", "k", json_data={})
        assert post_calls == 3

    @pytest.mark.asyncio
    async def test_moderation_error_has_safe_structured_diagnostics(self, monkeypatch):
        provider = OpenAIProvider(api_key="k")

        class _Client:
            is_closed = False

            async def post(self, *args, **kwargs):
                request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
                return httpx.Response(
                    400,
                    request=request,
                    headers={"x-request-id": "req_safe_123"},
                    json={
                        "error": {
                            "code": "content_policy_violation",
                            "message": "raw classifier details must stay private",
                        }
                    },
                )

        monkeypatch.setattr(provider, "_ensure_client", lambda: _Client())

        with pytest.raises(ProviderError) as exc_info:
            await provider._make_api_request("/images/generations", "k", json_data={})

        error = exc_info.value
        assert error.code == "moderation_blocked"
        assert error.request_id == "req_safe_123"
        assert "raw classifier" not in error.user_message
