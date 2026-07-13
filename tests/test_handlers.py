"""Tests for the MCP tool handlers in src/server.py.

These exercise each tool function directly (the FastMCP decorator returns
the original callable), with the provider registry mocked so no network
calls happen.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent
from PIL import Image

from src import server
from src.config.settings import Settings
from src.models.input_models import (
    BatchGenerationInput,
    BatchItem,
    ConversationalImageInput,
    CostEstimateInput,
    EditImageInput,
    ImageGenerationInput,
    ListConversationsInput,
    OutputFormat,
    Provider,
)
from src.providers.base import ImageResult
from src.providers.openai_provider import OpenAIProvider
from src.providers.selector import ProviderRecommendation


def make_result(**kw) -> ImageResult:
    base = ImageResult(
        success=True,
        provider="openai",
        model="gpt-image-2",
        image_path=Path("/tmp/x.png"),
        prompt="p",
        size="1024x1024",
        generation_time_seconds=1.2,
    )
    return dataclasses.replace(base, **kw)


@pytest.fixture
def fake_registry(monkeypatch):
    """Patch server.get_provider_registry with a controllable mock."""
    reg = MagicMock()
    rec = ProviderRecommendation(provider="openai", confidence=0.9, reasoning="text rendering")
    provider = MagicMock()
    provider.generate_image = AsyncMock(return_value=make_result())
    reg.get_provider_for_prompt.return_value = (provider, rec)
    reg.is_provider_available.return_value = True
    monkeypatch.setattr(server, "get_provider_registry", lambda: reg)
    return reg, provider, rec


# --------------------------------------------------------------------------
# generate_image
# --------------------------------------------------------------------------


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_markdown_success(self, fake_registry):
        out = await server.generate_image(ImageGenerationInput(prompt="A poster with text"))
        assert "Image Generated Successfully" in out
        assert "Openai" in out

    @pytest.mark.asyncio
    async def test_json_output(self, fake_registry):
        out = await server.generate_image(
            ImageGenerationInput(prompt="A poster", output_format=OutputFormat.JSON)
        )
        data = json.loads(out)
        assert data["success"] is True
        assert data["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_failure_result_raises_tool_error(self, fake_registry):
        _, provider, _ = fake_registry
        provider.generate_image.return_value = make_result(
            success=False, image_path=None, error="boom"
        )
        with pytest.raises(ToolError, match="boom"):
            await server.generate_image(ImageGenerationInput(prompt="x"))

    @pytest.mark.asyncio
    async def test_unexpected_exception_sanitized(self, fake_registry):
        _, provider, _ = fake_registry
        provider.generate_image.side_effect = RuntimeError("token sk-ABCDEFGHIJKLMNOP")
        with pytest.raises(ToolError) as exc_info:
            await server.generate_image(ImageGenerationInput(prompt="x"))
        assert "sk-ABCDEFGHIJKLMNOP" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_progress_reported_when_ctx_present(self, fake_registry):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        await server.generate_image(ImageGenerationInput(prompt="A poster"), ctx=ctx)
        assert ctx.report_progress.await_count >= 2

    @pytest.mark.asyncio
    async def test_explicit_provider_forwarded(self, fake_registry):
        reg, _, _ = fake_registry
        await server.generate_image(ImageGenerationInput(prompt="x", provider=Provider.OPENAI))
        _, kwargs = reg.get_provider_for_prompt.call_args
        assert kwargs["explicit_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_omitted_values_use_configured_openai_defaults(self, fake_registry, monkeypatch):
        reg, provider, _ = fake_registry
        monkeypatch.setattr(
            server,
            "get_settings",
            lambda: Settings(
                default_provider="openai",
                default_openai_size="1536x1024",
                enable_prompt_enhancement=False,
                enable_google_search=False,
            ),
        )

        await server.generate_image(ImageGenerationInput(prompt="x"))

        assert reg.get_provider_for_prompt.call_args.kwargs["explicit_provider"] == "openai"
        assert reg.get_provider_for_prompt.call_args.kwargs["size"] == "1536x1024"
        call = provider.generate_image.call_args.kwargs
        assert call["size"] == "1536x1024"
        assert call["enable_enhancement"] is False
        assert call["enable_google_search"] is False

    @pytest.mark.asyncio
    async def test_explicit_false_and_size_override_server_defaults(
        self, fake_registry, monkeypatch
    ):
        _, provider, _ = fake_registry
        monkeypatch.setattr(
            server,
            "get_settings",
            lambda: Settings(
                default_openai_size="1536x1024",
                enable_prompt_enhancement=True,
                enable_google_search=True,
            ),
        )

        await server.generate_image(
            ImageGenerationInput(
                prompt="x",
                size="2048x1024",
                enhance_prompt=False,
                enable_google_search=False,
            )
        )

        call = provider.generate_image.call_args.kwargs
        assert call["size"] == "2048x1024"
        assert call["enable_enhancement"] is False
        assert call["enable_google_search"] is False

    @pytest.mark.asyncio
    async def test_search_json_fails_before_provider_call(self, fake_registry):
        reg, provider, _ = fake_registry

        with pytest.raises(ToolError, match="Search Suggestions widget"):
            await server.generate_image(
                ImageGenerationInput(
                    prompt="Current weather",
                    enable_google_search=True,
                    output_format=OutputFormat.JSON,
                )
            )

        reg.get_provider_for_prompt.assert_not_called()
        provider.generate_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gemini_controls_hard_route_to_gemini(self, fake_registry):
        reg, _, _ = fake_registry

        await server.generate_image(
            ImageGenerationInput(prompt="x", gemini_model="gemini-3.1-flash-image")
        )

        assert reg.get_provider_for_prompt.call_args.kwargs["explicit_provider"] == "gemini"

    @pytest.mark.asyncio
    async def test_conflicting_provider_specific_controls_fail_before_selection(
        self, fake_registry
    ):
        reg, provider, _ = fake_registry

        with pytest.raises(ToolError, match="Cannot combine"):
            await server.generate_image(
                ImageGenerationInput(
                    prompt="x",
                    openai_model="gpt-image-2",
                    gemini_model="gemini-3.1-flash-image",
                )
            )

        reg.get_provider_for_prompt.assert_not_called()
        provider.generate_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_provider_control_conflict_fails_closed(self, fake_registry):
        reg, _, _ = fake_registry

        with pytest.raises(ToolError, match="conflict with provider='openai'"):
            await server.generate_image(
                ImageGenerationInput(
                    prompt="x",
                    provider=Provider.OPENAI,
                    thinking_level="high",
                )
            )

        reg.get_provider_for_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_global_enhancement_default_does_not_override_explicit_gemini(
        self, fake_registry, monkeypatch
    ):
        reg, provider, rec = fake_registry
        reg.get_provider_for_prompt.return_value = (
            provider,
            dataclasses.replace(rec, provider="gemini"),
        )
        monkeypatch.setattr(
            server,
            "get_settings",
            lambda: Settings(enable_prompt_enhancement=True),
        )

        await server.generate_image(ImageGenerationInput(prompt="x", provider=Provider.GEMINI))

        assert reg.get_provider_for_prompt.call_args.kwargs["explicit_provider"] == "gemini"
        assert "enable_enhancement" not in provider.generate_image.call_args.kwargs

    @pytest.mark.asyncio
    async def test_live_mcp_result_includes_bounded_image_preview(self, fake_registry, tmp_path):
        _, provider, _ = fake_registry
        image_path = tmp_path / "wide.png"
        Image.new("RGB", (2048, 512), "navy").save(image_path)
        provider.generate_image.return_value = make_result(image_path=image_path)
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()

        tool = server.mcp._tool_manager.get_tool("generate_image")
        assert tool is not None
        assert tool.output_schema is None
        content = await tool.run(
            {
                "params": {
                    "prompt": "preview",
                    "enhance_prompt": False,
                    "include_preview": True,
                }
            },
            context=ctx,
            convert_result=True,
        )

        assert any(
            isinstance(block, TextContent) and "Image Generated Successfully" in block.text
            for block in content
        )
        preview = next(block for block in content if isinstance(block, ImageContent))
        assert preview.mimeType == "image/jpeg"
        preview_bytes = base64.b64decode(preview.data)
        assert len(preview_bytes) <= 200_000
        with Image.open(io.BytesIO(preview_bytes)) as thumbnail:
            assert max(thumbnail.size) <= 512

    @pytest.mark.asyncio
    async def test_preview_is_disabled_by_default(self, fake_registry, tmp_path):
        _, provider, _ = fake_registry
        image_path = tmp_path / "image.png"
        Image.new("RGB", (32, 32), "navy").save(image_path)
        provider.generate_image.return_value = make_result(image_path=image_path)
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()

        result = await server.generate_image(ImageGenerationInput(prompt="preview"), ctx=ctx)

        assert isinstance(result, str)


# --------------------------------------------------------------------------
# conversational_image
# --------------------------------------------------------------------------


class TestConversationalImage:
    @pytest.mark.asyncio
    async def test_skip_dialogue_generates(self, fake_registry):
        out = await server.conversational_image(
            ConversationalImageInput(prompt="a cat", skip_dialogue=True)
        )
        assert "Image Generated Successfully" in out

    @pytest.mark.asyncio
    async def test_default_conversation_can_use_explicit_gemini(self, fake_registry):
        reg, provider, rec = fake_registry
        reg.get_provider_for_prompt.return_value = (
            provider,
            dataclasses.replace(rec, provider="gemini"),
        )

        await server.conversational_image(
            ConversationalImageInput(
                prompt="a cat",
                provider=Provider.GEMINI,
                skip_dialogue=True,
            )
        )

        assert reg.get_provider_for_prompt.call_args.kwargs["explicit_provider"] == "gemini"
        assert "enable_enhancement" not in provider.generate_image.call_args.kwargs

    @pytest.mark.asyncio
    async def test_vague_prompt_returns_questions(self, fake_registry):
        # No ctx → elicitation unavailable → text dialogue questions.
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided")
        )
        assert "?" in out  # contains dialogue questions

    @pytest.mark.asyncio
    async def test_elicitation_accepted_enriches_prompt(self, fake_registry):
        _, provider, _ = fake_registry
        ctx = MagicMock()
        ctx.elicit = AsyncMock(
            return_value=MagicMock(
                action="accept",
                data=MagicMock(style="oil-painting", mood="warm", additional_details=None),
            )
        )
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided"),
            ctx=ctx,
        )
        assert "Image Generated Successfully" in out
        # The enriched prompt is what reached the provider.
        called_prompt = provider.generate_image.call_args.args[0]
        assert "oil-painting" in called_prompt
        assert "warm" in called_prompt

    @pytest.mark.asyncio
    async def test_elicitation_declined_falls_back_to_questions(self, fake_registry):
        ctx = MagicMock()
        ctx.elicit = AsyncMock(return_value=MagicMock(action="decline", data=None))
        out = await server.conversational_image(
            ConversationalImageInput(prompt="something cool", dialogue_mode="guided"),
            ctx=ctx,
        )
        assert "?" in out

    @pytest.mark.asyncio
    async def test_openai_fields_and_configured_size_are_forwarded(
        self, fake_registry, monkeypatch
    ):
        _, provider, _ = fake_registry
        monkeypatch.setattr(
            server,
            "get_settings",
            lambda: Settings(default_provider="openai", default_openai_size="1536x1024"),
        )

        await server.conversational_image(
            ConversationalImageInput(
                prompt="a cat",
                skip_dialogue=True,
                openai_model="gpt-image-1.5",
                quality="high",
                background="opaque",
            )
        )

        call = provider.generate_image.call_args.kwargs
        assert call["size"] == "1536x1024"
        assert call["openai_model"] == "gpt-image-1.5"
        assert call["quality"] == "high"
        assert call["background"] == "opaque"

    @pytest.mark.asyncio
    async def test_existing_conversation_rejects_provider_switch(self, fake_registry, monkeypatch):
        from src.services import conversation_store

        reg, _, _ = fake_registry
        store = MagicMock()
        store.get_conversation.return_value = {"id": "conv-1", "provider": "openai"}
        monkeypatch.setattr(conversation_store, "get_conversation_store", lambda: store)

        with pytest.raises(ToolError, match="locked to provider 'openai'"):
            await server.conversational_image(
                ConversationalImageInput(
                    prompt="make it blue",
                    conversation_id="conv-1",
                    provider=Provider.GEMINI,
                    skip_dialogue=True,
                )
            )

        reg.get_provider_for_prompt.assert_not_called()


# --------------------------------------------------------------------------
# edit_image
# --------------------------------------------------------------------------


class TestEditImage:
    @pytest.mark.asyncio
    async def test_edit_success(self, monkeypatch):
        reg = MagicMock()
        reg.is_provider_available.return_value = True
        op = OpenAIProvider(api_key="k")
        op.edit_image = AsyncMock(return_value=make_result(provider="openai"))
        reg.get_provider.return_value = op
        monkeypatch.setattr(server, "get_provider_registry", lambda: reg)

        out = await server.edit_image(EditImageInput(prompt="change sky", image_path="/tmp/in.png"))
        assert "Image Generated Successfully" in out
        op.edit_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_unavailable_when_no_openai(self, monkeypatch):
        reg = MagicMock()
        reg.is_provider_available.return_value = False
        monkeypatch.setattr(server, "get_provider_registry", lambda: reg)

        with pytest.raises(ToolError, match="OpenAI provider is not configured"):
            await server.edit_image(EditImageInput(prompt="change sky", image_path="/tmp/in.png"))


# --------------------------------------------------------------------------
# read-only tools (use the real registry)
# --------------------------------------------------------------------------


class TestReadOnlyTools:
    @pytest.mark.asyncio
    async def test_list_providers(self):
        out = await server.list_providers()
        assert "Provider Comparison" in out

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self):
        out = await server.list_conversations(ListConversationsInput(limit=5))
        assert "Conversations" in out


# --------------------------------------------------------------------------
# estimate_cost
# --------------------------------------------------------------------------


class TestEstimateCost:
    @pytest.mark.asyncio
    async def test_markdown(self):
        out = await server.estimate_cost(
            CostEstimateInput(prompt="A poster with bold text", quality="medium")
        )
        assert "Cost Estimate" in out
        assert "$" in out

    @pytest.mark.asyncio
    async def test_json(self):
        out = await server.estimate_cost(
            CostEstimateInput(
                prompt="A poster",
                provider=Provider.OPENAI,
                quality="high",
                size="1024x1024",
                n=2,
                output_format=OutputFormat.JSON,
            )
        )
        data = json.loads(out)
        assert data["provider"] == "openai"
        assert data["n"] == 2
        assert data["total_usd"] is not None


# --------------------------------------------------------------------------
# generate_image_batch
# --------------------------------------------------------------------------


class TestGenerateImageBatch:
    @pytest.mark.asyncio
    async def test_batch_all_succeed(self, fake_registry):
        items = [BatchItem(prompt=f"poster {i}") for i in range(3)]
        out = await server.generate_image_batch(BatchGenerationInput(items=items))
        assert "3/3 succeeded" in out
        for i in range(3):
            assert f"[{i}]" in out

    @pytest.mark.asyncio
    async def test_batch_per_item_failure_isolated(self, fake_registry):
        _, provider, _ = fake_registry
        # One of the three items raises; the batch must still return the others.
        provider.generate_image.side_effect = [
            make_result(),
            RuntimeError("boom"),
            make_result(),
        ]
        items = [BatchItem(prompt=f"poster {i}") for i in range(3)]
        out = await server.generate_image_batch(BatchGenerationInput(items=items))
        assert "2/3 succeeded" in out
        assert "Error:" in out

    @pytest.mark.asyncio
    async def test_batch_total_failure_raises_tool_error(self, fake_registry):
        _, provider, _ = fake_registry
        provider.generate_image.side_effect = RuntimeError("provider unavailable")
        items = [BatchItem(prompt="a"), BatchItem(prompt="b")]

        with pytest.raises(ToolError) as exc_info:
            await server.generate_image_batch(BatchGenerationInput(items=items))

        assert "0/2 succeeded" in str(exc_info.value)
        assert "provider unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_json(self, fake_registry):
        items = [BatchItem(prompt="a"), BatchItem(prompt="b")]
        out = await server.generate_image_batch(
            BatchGenerationInput(items=items, output_format=OutputFormat.JSON)
        )
        data = json.loads(out)
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert len(data["results"]) == 2
        # Results are ordered by index.
        assert [r["index"] for r in data["results"]] == [0, 1]

    @pytest.mark.asyncio
    async def test_batch_progress_reported(self, fake_registry):
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        items = [BatchItem(prompt=f"p{i}") for i in range(4)]
        await server.generate_image_batch(BatchGenerationInput(items=items), ctx=ctx)
        # One progress report per completed item.
        assert ctx.report_progress.await_count == 4

    @pytest.mark.asyncio
    async def test_batch_items_use_server_defaults(self, fake_registry, monkeypatch):
        reg, provider, rec = fake_registry
        gemini_rec = dataclasses.replace(rec, provider="gemini")
        reg.get_provider_for_prompt.return_value = (provider, gemini_rec)
        monkeypatch.setattr(
            server,
            "get_settings",
            lambda: Settings(
                default_provider="gemini",
                default_gemini_size="2K",
                default_gemini_aspect_ratio="16:9",
                enable_prompt_enhancement=False,
                enable_google_search=False,
            ),
        )

        await server.generate_image_batch(BatchGenerationInput(items=[BatchItem(prompt="x")]))

        selection = reg.get_provider_for_prompt.call_args.kwargs
        assert selection["explicit_provider"] == "gemini"
        assert selection["size"] == "2K"
        call = provider.generate_image.call_args.kwargs
        assert call["size"] == "2K"
        assert call["aspect_ratio"] == "16:9"
        assert "enable_enhancement" not in call
        assert call["enable_google_search"] is False

    @pytest.mark.asyncio
    async def test_batch_rejects_search_grounding_before_provider_call(self, fake_registry):
        reg, provider, _ = fake_registry

        with pytest.raises(ToolError, match="not supported in batch output"):
            await server.generate_image_batch(
                BatchGenerationInput(
                    items=[BatchItem(prompt="current weather", enable_google_search=True)]
                )
            )

        reg.get_provider_for_prompt.assert_not_called()
        provider.generate_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency_limit(self, fake_registry):
        _, provider, _ = fake_registry
        # Track concurrent in-flight calls; assert it never exceeds the cap.
        state = {"current": 0, "peak": 0}

        async def _slow(*args, **kwargs):
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0)
            state["current"] -= 1
            return make_result()

        provider.generate_image.side_effect = _slow
        items = [BatchItem(prompt=f"p{i}") for i in range(8)]
        await server.generate_image_batch(BatchGenerationInput(items=items, max_concurrency=2))
        assert state["peak"] <= 2
