"""
OpenAI gpt-image-2 (ChatGPT Images 2.0) provider implementation.

Two code paths are exposed:

- **Direct path** — a plain POST to ``/images/generations`` with the full
  gpt-image-2 parameter surface (quality, output_format, background,
  moderation, n, output_compression). Used when
  ``enable_enhancement=False`` to avoid the prompt-refinement round trip.

- **Chat Completions refinement path** — a two-stage flow where ``gpt-5.1`` (or a
  user-chosen assistant model) first refines the prompt through a
  forced function call, then the refined prompt is passed to
  ``/images/generations``. Used when ``enable_enhancement=True`` and
  for the conversational tool (preserves multi-turn context).

An ``edit_image`` entry point targets ``/images/edits``.  gpt-image-2 always
processes inputs at high fidelity, so its requests deliberately omit the
legacy ``input_fidelity`` parameter.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Any, cast

import httpx

from ..config.constants import (
    DEFAULT_OPENAI_BACKGROUND,
    DEFAULT_OPENAI_IMAGE_MODEL,
    DEFAULT_OPENAI_INPUT_FIDELITY,
    DEFAULT_OPENAI_MODERATION,
    DEFAULT_OPENAI_OUTPUT_FORMAT,
    DEFAULT_OPENAI_QUALITY,
    OPENAI_API_BASE_URL,
    OPENAI_ASPECT_RATIOS,
    OPENAI_ASSISTANT_MODELS,
    OPENAI_BACKGROUND_OPTIONS,
    OPENAI_EDIT_SIZES,
    OPENAI_GPT_IMAGE_2_BACKGROUND_OPTIONS,
    OPENAI_GPT_IMAGE_2_MAX_ASPECT_RATIO,
    OPENAI_GPT_IMAGE_2_MAX_EDGE,
    OPENAI_GPT_IMAGE_2_MAX_PIXELS,
    OPENAI_GPT_IMAGE_2_MIN_PIXELS,
    OPENAI_IMAGE_MODELS,
    OPENAI_INPUT_FIDELITY_OPTIONS,
    OPENAI_LEGACY_SIZES,
    OPENAI_MAX_N,
    OPENAI_MAX_PROMPT_LENGTH,
    OPENAI_MODERATION_OPTIONS,
    OPENAI_OUTPUT_FORMATS,
    OPENAI_QUALITY_OPTIONS,
    OPENAI_SIZES,
)
from ..config.settings import get_settings
from ..exceptions import (
    AuthenticationError,
    GenerationError,
    ProviderError,
    RateLimitError,
)
from .base import ImageProvider, ImageResult, ProviderCapabilities

logger = logging.getLogger(__name__)


class OpenAIProvider(ImageProvider):
    """
    OpenAI gpt-image-2 (ChatGPT Images 2.0) provider.

    Best for:
    - Text rendering (menus, infographics, comics)
    - UI mockups and screenshot-style renders
    - Technical diagrams and labeled illustrations
    - Marketing materials with exact text
    - Precise instruction following and world knowledge

    Limitations vs Gemini:
    - Sizes above 2560x1440 are experimental (maximum edge 3840px)
    - No transparent-background output
    - No reference image support on /images/generations
      (use the edit_image tool instead)
    - No real-time data grounding
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._capabilities: ProviderCapabilities | None = None

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it lazily.

        Uses a short connect timeout (fail fast on network issues) but a
        generous read/write/pool ceiling so slow high-quality gpt-image-2
        renders aren't cut off mid-generation. The ceiling is configurable
        via ``REQUEST_TIMEOUT`` (seconds).
        """
        if self._client is None or self._client.is_closed:
            timeout_s = float(get_settings().request_timeout)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI gpt-image-2 (ChatGPT Images 2.0)"

    @property
    def capabilities(self) -> ProviderCapabilities:
        if self._capabilities is None:
            self._capabilities = ProviderCapabilities(
                name="openai",
                display_name="OpenAI gpt-image-2",
                supported_sizes=OPENAI_SIZES,
                supported_aspect_ratios=["1:1", "2:3", "3:2", "16:9", "9:16"],
                max_resolution="3840px edge (8,294,400 pixels max)",
                supports_text_rendering=True,
                text_rendering_quality="excellent",
                supports_reference_images=False,  # via edit_image only
                max_reference_images=0,
                supports_real_time_data=False,
                supports_thinking_mode=False,
                supports_multi_turn=True,
                typical_latency_seconds=None,
                cost_tier="standard",
                best_for=[
                    "Text rendering (menus, posters, infographics)",
                    "UI mockups and labeled screenshots",
                    "Comics with dialogue and speech bubbles",
                    "Technical diagrams with precise labels",
                    "Marketing materials with text",
                    "Multi-step sequential edits (preserve-pixel editing)",
                ],
                not_recommended_for=[
                    "Transparent-background output (use downstream removal)",
                    "Multi-reference character consistency (use Gemini)",
                    "Real-time data visualization (use Gemini)",
                ],
            )
        return self._capabilities

    # ------------------------------------------------------------------
    # API key + HTTP helper
    # ------------------------------------------------------------------

    def _get_api_key(self, provided_key: str | None = None) -> str:
        """Get API key from provided value, instance, or settings."""
        api_key = provided_key or self._api_key
        if not api_key:
            settings = get_settings()
            api_key = settings.get_openai_api_key()
        return api_key

    def _resolve_model(self, openai_model: str | None, *, allow_unknown: bool = False) -> str:
        """Resolve an Image API model identifier.

        Strictly validates against the known ``OPENAI_IMAGE_MODELS`` registry
        by default, so a typo never silently bills a different model. Callers
        that already resolved ``openai_model`` themselves (explicit user
        choice, or a model discovered live from OpenAI's own models API --
        see ``src.smart_tool.model_resolution``) may pass
        ``allow_unknown=True`` to pass an unrecognized id through unchanged,
        deferring rejection to the API itself. This keeps the MCP server's
        default behavior unchanged while letting the smart tool stay
        model-agnostic.
        """
        if not openai_model:
            return DEFAULT_OPENAI_IMAGE_MODEL
        try:
            return OPENAI_IMAGE_MODELS[openai_model]
        except KeyError as exc:
            if allow_unknown:
                logger.info(
                    "Passing unrecognized OpenAI image model '%s' through to the API.",
                    openai_model,
                )
                return openai_model
            supported = ", ".join(OPENAI_IMAGE_MODELS)
            raise ValueError(
                f"Unsupported OpenAI image model '{openai_model}'. Supported: {supported}."
            ) from exc

    @staticmethod
    def _is_gpt_image_2(model: str) -> bool:
        """Return whether ``model`` uses the gpt-image-2 parameter contract."""
        return model == "gpt-image-2"

    def _validate_size(self, size: str, model: str) -> str:
        """Normalize and validate an Image API size for the selected model.

        gpt-image-2 accepts constrained arbitrary resolutions.  Earlier models
        keep the server's historical enumerated behavior for compatibility.
        """
        normalized = size.strip().replace("X", "x")
        if normalized.lower() == "auto":
            return "auto"

        # Provider-neutral shorthand is useful for callers that switch models.
        if self._is_gpt_image_2(model):
            shorthand = {
                "1K": "1024x1024",
                "2K": "2560x1440",
                "4K": "3840x2160",
            }
        else:
            shorthand = {
                "1K": "1024x1024",
                "2K": "1536x1024",
                "4K": "1536x1024",
            }

        mapped = shorthand.get(normalized.upper())
        if mapped:
            logger.info("Converting size shorthand '%s' to OpenAI size '%s'.", size, mapped)
            normalized = mapped

        if not self._is_gpt_image_2(model):
            if normalized not in OPENAI_LEGACY_SIZES:
                raise ValueError(
                    f"Invalid size '{normalized}' for {model}. Supported: "
                    f"{', '.join(OPENAI_LEGACY_SIZES)}"
                )
            return normalized

        parts = normalized.split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"Invalid size '{normalized}' for gpt-image-2. Use 'auto' or WIDTHxHEIGHT."
            )

        width, height = (int(part) for part in parts)
        if width == 0 or height == 0:
            raise ValueError("gpt-image-2 width and height must be positive.")
        if width % 16 or height % 16:
            raise ValueError("gpt-image-2 width and height must both be multiples of 16.")
        if max(width, height) > OPENAI_GPT_IMAGE_2_MAX_EDGE:
            raise ValueError(
                "gpt-image-2 width and height must each be at most "
                f"{OPENAI_GPT_IMAGE_2_MAX_EDGE}px."
            )

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > OPENAI_GPT_IMAGE_2_MAX_ASPECT_RATIO:
            raise ValueError("gpt-image-2 long-to-short edge ratio must not exceed 3:1.")

        pixels = width * height
        if not OPENAI_GPT_IMAGE_2_MIN_PIXELS <= pixels <= OPENAI_GPT_IMAGE_2_MAX_PIXELS:
            raise ValueError(
                "gpt-image-2 total pixels must be between "
                f"{OPENAI_GPT_IMAGE_2_MIN_PIXELS:,} and "
                f"{OPENAI_GPT_IMAGE_2_MAX_PIXELS:,}."
            )
        return normalized

    def _validate_background(self, background: str, model: str) -> str:
        """Validate background treatment against the selected model."""
        allowed = (
            OPENAI_GPT_IMAGE_2_BACKGROUND_OPTIONS
            if self._is_gpt_image_2(model)
            else OPENAI_BACKGROUND_OPTIONS
        )
        if background not in allowed:
            if self._is_gpt_image_2(model) and background == "transparent":
                raise ValueError(
                    "gpt-image-2 does not support transparent backgrounds; "
                    "use 'auto' or 'opaque' and remove the background downstream."
                )
            raise ValueError(
                f"Invalid background '{background}' for {model}. Supported: {', '.join(allowed)}"
            )
        return background

    @staticmethod
    def _is_retryable_api_error(error: Exception) -> bool:
        """Return whether an OpenAI request failure is safe to retry."""
        if isinstance(error, httpx.TimeoutException):
            # Image renders can run for minutes; retrying multiplies the wait
            # and may duplicate billable work.
            return False
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            return status == 429 or status >= 500
        # Connection resets, DNS/connect failures, and protocol errors are
        # transient transport failures. Other programming/data errors are not.
        return isinstance(error, httpx.TransportError)

    async def _make_api_request(
        self,
        endpoint: str,
        api_key: str,
        json_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """POST or multipart-POST to OpenAI with retry + rate limiting."""
        await self._acquire_rate_limit()

        url = f"{OPENAI_API_BASE_URL}{endpoint}"
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        if files is None:
            headers["Content-Type"] = "application/json"

        client = self._ensure_client()

        async def _do_request() -> dict[str, Any]:
            if files is not None:
                # Multipart upload (for /images/edits)
                response = await client.post(url, headers=headers, files=files, data=data)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=json_data)
            else:
                response = await client.request(method, url, headers=headers, json=json_data)
            response.raise_for_status()
            return cast("dict[str, Any]", response.json())

        try:
            return cast(
                "dict[str, Any]",
                await self._retry_with_backoff(
                    _do_request,
                    is_retryable=self._is_retryable_api_error,
                ),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            request_id = e.response.headers.get("x-request-id")
            try:
                payload = e.response.json()
                error_payload = payload.get("error", {}) if isinstance(payload, dict) else {}
            except (ValueError, TypeError):
                error_payload = {}
            raw_code = (
                error_payload.get("code") or error_payload.get("type")
                if isinstance(error_payload, dict)
                else None
            )
            code = str(raw_code or f"http_{status}")
            normalized_code = code.lower()
            logger.error(
                "OpenAI request failed status=%s code=%s request_id=%s retryable=%s",
                status,
                code,
                request_id,
                status == 429 or status >= 500,
            )
            common = {
                "provider": "openai",
                "status_code": status,
                "request_id": request_id,
            }
            if status in (401, 403):
                raise AuthenticationError(
                    "OpenAI authentication failed.",
                    code=code,
                    user_message="OpenAI authentication failed. Check the configured API key.",
                    **common,
                ) from e
            if status == 429:
                raise RateLimitError(
                    "OpenAI rate limit exceeded.",
                    **common,
                ) from e
            if any(term in normalized_code for term in ("moderation", "safety", "policy")):
                raise GenerationError(
                    "OpenAI moderation blocked the request.",
                    code="moderation_blocked",
                    user_message="The image request was blocked by the provider's safety policy.",
                    **common,
                ) from e
            if status >= 500:
                raise ProviderError(
                    "OpenAI service error.",
                    code=code,
                    retryable=True,
                    user_message="OpenAI is temporarily unavailable. Please try again.",
                    **common,
                ) from e
            raise GenerationError(
                "OpenAI rejected the image request.",
                code=code,
                user_message="OpenAI rejected the image request. Check its prompt and parameters.",
                **common,
            ) from e
        except ProviderError:
            raise
        except httpx.TimeoutException as e:
            raise ProviderError(
                "OpenAI request timed out.",
                provider="openai",
                code="timeout",
                retryable=False,
                user_message="OpenAI image generation timed out; it was not retried.",
            ) from e
        except httpx.TransportError as e:
            raise ProviderError(
                "OpenAI transport failure.",
                provider="openai",
                code="transport_error",
                retryable=True,
                user_message="Could not reach OpenAI. Please try again.",
            ) from e
        except Exception as e:
            raise ProviderError(
                "OpenAI returned an invalid response.",
                provider="openai",
                code="invalid_response",
                retryable=False,
                user_message="OpenAI returned an invalid response.",
            ) from e

    # ------------------------------------------------------------------
    # Direct /images/generations path
    # ------------------------------------------------------------------

    def _build_generate_payload(
        self,
        *,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Build the JSON body for /images/generations.

        Only includes keys that were explicitly provided so the API applies
        its own defaults for anything omitted.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            # gpt-image-2 returns b64_json in data[] by default;
            # the legacy "response_format" param is not supported.
        }
        if quality is not None:
            payload["quality"] = quality
        if output_format is not None:
            payload["output_format"] = output_format
        if output_compression is not None:
            payload["output_compression"] = output_compression
        if background is not None:
            payload["background"] = background
        if moderation is not None:
            payload["moderation"] = moderation
        if style is not None:
            raise ValueError(
                "style is only supported by DALL-E 3 and is unavailable on GPT Image models."
            )
        if n is not None and n > 1:
            payload["n"] = n
        return payload

    async def _call_images_generate_direct(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Call /images/generations directly without prompt refinement."""
        payload = self._build_generate_payload(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format,
            output_compression=output_compression,
            background=background,
            moderation=moderation,
            style=style,
            n=n,
        )
        logger.info("Calling /images/generations directly with model=%s", model)
        return await self._make_api_request(
            endpoint="/images/generations",
            api_key=api_key,
            json_data=payload,
        )

    async def _call_images_edit_bytes(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        image_b64: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Apply a conversational edit to the prior generated image bytes."""
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except ValueError as e:
            raise ValueError("Stored conversation image is not valid base64 data.") from e

        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            filename, content_type = "conversation.png", "image/png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            filename, content_type = "conversation.jpeg", "image/jpeg"
        elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            filename, content_type = "conversation.webp", "image/webp"
        else:
            raise ValueError("Stored conversation image is not a supported PNG, JPEG, or WebP.")

        form_data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }
        if not self._is_gpt_image_2(model):
            form_data["input_fidelity"] = DEFAULT_OPENAI_INPUT_FIDELITY
        if quality is not None:
            form_data["quality"] = quality
        if output_format is not None:
            form_data["output_format"] = output_format
        if output_compression is not None:
            form_data["output_compression"] = output_compression
        if background is not None:
            form_data["background"] = background
        if n is not None and n > 1:
            form_data["n"] = str(n)

        logger.info("Calling /images/edits for conversation continuation model=%s", model)
        return await self._make_api_request(
            endpoint="/images/edits",
            api_key=api_key,
            files={"image": (filename, image_bytes, content_type)},
            data=form_data,
        )

    # ------------------------------------------------------------------
    # Chat Completions prompt-refinement path
    # ------------------------------------------------------------------

    async def _call_chat_completions_refinement(
        self,
        *,
        prompt: str,
        api_key: str,
        conversation_id: str,
        assistant_model: str,
        image_model: str,
        size: str,
        quality: str | None,
        output_format: str | None,
        output_compression: int | None,
        background: str | None,
        moderation: str | None,
        style: str | None,
        n: int | None,
    ) -> dict[str, Any]:
        """Refine through /chat/completions, then call image generation."""
        if assistant_model not in OPENAI_ASSISTANT_MODELS:
            raise ValueError(
                f"Unsupported OpenAI assistant model '{assistant_model}'. Supported: "
                f"{', '.join(sorted(OPENAI_ASSISTANT_MODELS))}."
            )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        # Forced function-calling tool schema — lets the assistant model refine
        # the prompt and choose a size. We still own the actual image call.
        tool_schema_properties: dict[str, Any] = {
            "prompt": {
                "type": "string",
                "description": "Refined prompt to send to the image model",
            },
        }
        if self._is_gpt_image_2(image_model):
            tool_schema_properties["size"] = {
                "type": "string",
                "description": (
                    "Output size as WIDTHxHEIGHT: both edges must be multiples of 16 and "
                    "at most 3840px; ratio <= 3:1; 655,360-8,294,400 total pixels."
                ),
                "default": "1024x1024",
            }
        else:
            tool_schema_properties["size"] = {
                "type": "string",
                "enum": list(OPENAI_LEGACY_SIZES),
                "default": "1024x1024",
            }
        if quality is None:
            tool_schema_properties["quality"] = {
                "type": "string",
                "enum": list(OPENAI_QUALITY_OPTIONS),
            }

        payload = {
            "model": assistant_model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": (
                            f"Generate an image using {image_model}. "
                            "Refine the user's prompt before calling."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": tool_schema_properties,
                            "required": ["prompt"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "generate_image"},
            },
            "parallel_tool_calls": False,
            "max_tokens": 1000,
        }

        response = await self._make_api_request(
            endpoint="/chat/completions",
            api_key=api_key,
            json_data=payload,
        )

        if not ("choices" in response and response["choices"]):
            return response

        choice = response["choices"][0]
        if "message" not in choice:
            return response

        assistant_message = choice["message"]
        if not isinstance(assistant_message, dict):
            raise ValueError("Assistant refinement response did not contain a message object.")

        if "tool_calls" not in assistant_message:
            return response

        tool_calls = assistant_message["tool_calls"]
        if not isinstance(tool_calls, list):
            raise ValueError("Assistant refinement response contained invalid tool calls.")

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
                raise ValueError("Assistant refinement response contained an invalid tool call.")
            function = tool_call["function"]
            if function.get("name") != "generate_image":
                continue

            raw_args = function.get("arguments")
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Assistant refinement returned malformed JSON arguments."
                    ) from exc
            else:
                tool_args = raw_args
            if not isinstance(tool_args, dict):
                raise ValueError("Assistant refinement arguments must be a JSON object.")

            unknown_fields = set(tool_args) - set(tool_schema_properties)
            if unknown_fields:
                raise ValueError(
                    "Assistant refinement returned unsupported fields: "
                    f"{', '.join(sorted(str(field) for field in unknown_fields))}."
                )

            refined_prompt_value = tool_args.get("prompt")
            if not isinstance(refined_prompt_value, str) or not refined_prompt_value.strip():
                raise ValueError("Assistant refinement returned an empty or invalid prompt.")
            if len(refined_prompt_value) > OPENAI_MAX_PROMPT_LENGTH:
                raise ValueError(
                    "Assistant refinement prompt exceeds the OpenAI maximum of "
                    f"{OPENAI_MAX_PROMPT_LENGTH} characters."
                )
            refined_prompt = refined_prompt_value

            refined_size_value = tool_args.get("size", size)
            if not isinstance(refined_size_value, str):
                raise ValueError("Assistant refinement size must be a string.")
            refined_quality_value = tool_args.get("quality", quality)
            if refined_quality_value is not None and not isinstance(refined_quality_value, str):
                raise ValueError("Assistant refinement quality must be a string.")
            refined = await self.validate_params(
                refined_prompt,
                size=refined_size_value,
                quality=refined_quality_value,
                model=image_model,
            )
            refined_size = str(refined["size"])
            refined_quality = refined.get("quality", quality)

            settings = get_settings()
            if settings.log_prompts:
                logger.info("Assistant refined prompt via tool call: %s", tool_args)
            else:
                logger.info(
                    "Assistant refined prompt via tool call: "
                    "length=%d sha256=%s size=%s quality=%s",
                    len(refined_prompt),
                    hashlib.sha256(refined_prompt.encode("utf-8")).hexdigest(),
                    tool_args.get("size"),
                    tool_args.get("quality"),
                )

            image_response = await self._call_images_generate_direct(
                api_key=api_key,
                model=image_model,
                prompt=refined_prompt,
                size=refined_size,
                quality=refined_quality,
                output_format=output_format,
                output_compression=output_compression,
                background=background,
                moderation=moderation,
                style=style,
                n=n,
            )

            return {
                "conversation_id": conversation_id,
                "chat_response": response,
                "image_response": image_response,
                "tool_calls": assistant_message["tool_calls"],
                "refined_prompt": refined_prompt,
            }

        return response

    # ------------------------------------------------------------------
    # Param validation
    # ------------------------------------------------------------------

    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate and normalize parameters for OpenAI."""
        if len(prompt) > OPENAI_MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt too long. Maximum {OPENAI_MAX_PROMPT_LENGTH} characters for OpenAI."
            )

        model = self._resolve_model(
            kwargs.get("openai_model") or kwargs.get("model"),
            allow_unknown=bool(kwargs.get("allow_unknown_model", False)),
        )

        # --- Size ---
        if size:
            size = self._validate_size(size, model)
            if aspect_ratio:
                expected_size = self._size_from_aspect_ratio(aspect_ratio, model)
                if size == "auto":
                    raise ValueError("size='auto' cannot be combined with an aspect_ratio.")
                if self._is_gpt_image_2(model):
                    requested_width, requested_height = self._parse_aspect_ratio(aspect_ratio)
                    width, height = (int(part) for part in size.split("x"))
                    if width * requested_height != height * requested_width:
                        raise ValueError(
                            f"size '{size}' conflicts with aspect_ratio '{aspect_ratio}'. "
                            f"Use the exact derived size '{expected_size}' or omit one control."
                        )
                elif size != expected_size:
                    raise ValueError(
                        f"size '{size}' conflicts with aspect_ratio '{aspect_ratio}' for {model}. "
                        f"The nearest supported size is '{expected_size}'."
                    )
        else:
            inferred_size = (
                self._size_from_aspect_ratio(aspect_ratio, model)
                if aspect_ratio
                else get_settings().default_openai_size
            )
            size = self._validate_size(inferred_size, model)

        # --- Enum validation (pass through to API if valid, else raise) ---
        validated: dict[str, Any] = {"prompt": prompt, "size": size}

        quality = kwargs.get("quality")
        if quality is not None:
            if quality not in OPENAI_QUALITY_OPTIONS:
                raise ValueError(
                    f"Invalid quality '{quality}'. Supported: {', '.join(OPENAI_QUALITY_OPTIONS)}"
                )
            validated["quality"] = quality

        output_format = kwargs.get("openai_output_format")
        if output_format is not None:
            if output_format not in OPENAI_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output_format '{output_format}'. "
                    f"Supported: {', '.join(OPENAI_OUTPUT_FORMATS)}"
                )
            validated["output_format"] = output_format

        output_compression = kwargs.get("openai_output_compression")
        if output_compression is not None:
            if not (0 <= int(output_compression) <= 100):
                raise ValueError("output_compression must be between 0 and 100.")
            validated["output_compression"] = int(output_compression)

        background = kwargs.get("background")
        if background is not None:
            validated["background"] = self._validate_background(str(background), model)

        moderation = kwargs.get("moderation")
        if moderation is not None:
            if moderation not in OPENAI_MODERATION_OPTIONS:
                raise ValueError(
                    f"Invalid moderation '{moderation}'. "
                    f"Supported: {', '.join(OPENAI_MODERATION_OPTIONS)}"
                )
            validated["moderation"] = moderation

        style = kwargs.get("style")
        if style is not None:
            raise ValueError(
                "style is only supported by DALL-E 3 and is unavailable on GPT Image models."
            )

        n = kwargs.get("n")
        if n is not None:
            n = int(n)
            if not (1 <= n <= OPENAI_MAX_N):
                raise ValueError(f"n must be between 1 and {OPENAI_MAX_N}.")
            validated["n"] = n

        return validated

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
        """Return normalized ratio components or reject an unsupported ratio."""
        normalized = aspect_ratio.strip().lower()
        normalized = {
            "portrait": "2:3",
            "landscape": "3:2",
            "square": "1:1",
        }.get(normalized, normalized)
        if normalized not in OPENAI_ASPECT_RATIOS:
            raise ValueError(
                f"Unsupported OpenAI aspect_ratio '{aspect_ratio}'. Supported: "
                f"{', '.join(OPENAI_ASPECT_RATIOS)}, portrait, landscape, square. "
                "Gemini-only extreme ratios such as 1:4 and 8:1 exceed OpenAI's 3:1 limit."
            )
        width, height = (int(part) for part in normalized.split(":"))
        common = gcd(width, height)
        return width // common, height // common

    def _size_from_aspect_ratio(self, aspect_ratio: str, model: str) -> str:
        """Convert a supported aspect ratio without silently changing its shape."""
        ratio_width, ratio_height = self._parse_aspect_ratio(aspect_ratio)
        if max(ratio_width, ratio_height) / min(ratio_width, ratio_height) > 3:
            raise ValueError("OpenAI image aspect ratios must not exceed 3:1.")

        if not self._is_gpt_image_2(model):
            candidates = [size for size in OPENAI_LEGACY_SIZES if size != "auto"]
            target = ratio_width / ratio_height
            return min(
                candidates,
                key=lambda candidate: abs(
                    int(candidate.split("x")[0]) / int(candidate.split("x")[1]) - target
                ),
            )

        # Keep the shorter edge at least 1024px while preserving the exact
        # reduced ratio and OpenAI's multiple-of-16 requirement.
        short_component = min(ratio_width, ratio_height)
        scale = (1024 + short_component * 16 - 1) // (short_component * 16)
        width = ratio_width * 16 * scale
        height = ratio_height * 16 * scale
        return self._validate_size(f"{width}x{height}", model)

    @staticmethod
    def _aspect_ratio_from_size(size: str) -> str | None:
        """Return the exact reduced ratio for a concrete WIDTHxHEIGHT size."""
        parts = size.lower().split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return None
        width, height = (int(part) for part in parts)
        if width <= 0 or height <= 0:
            return None
        common = gcd(width, height)
        return f"{width // common}:{height // common}"

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image_data(image_response: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a list of {'b64_json'|'url', 'revised_prompt'} entries."""
        entries: list[dict[str, Any]] = []
        for item in image_response.get("data") or []:
            entry: dict[str, Any] = {}
            if "b64_json" in item and item["b64_json"]:
                entry["b64_json"] = item["b64_json"]
            elif "url" in item and item["url"]:
                entry["url"] = item["url"]
            if item.get("revised_prompt"):
                entry["revised_prompt"] = item["revised_prompt"]
            if entry:
                entries.append(entry)
        return entries

    @staticmethod
    def _extract_usage(image_response: dict[str, Any]) -> dict[str, int] | None:
        """Return the usage block from an images response, if present."""
        usage = image_response.get("usage")
        if not usage or not isinstance(usage, dict):
            return None
        # Normalize to plain ints so the result is JSON-serializable.
        normalized: dict[str, int] = {}
        for key, value in usage.items():
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                continue
        return normalized or None

    # ------------------------------------------------------------------
    # Public generate / edit entry points
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str | None = None,
        aspect_ratio: str | None = None,
        conversation_id: str | None = None,
        reference_images: list[str] | None = None,
        enable_enhancement: bool = False,
        persist_conversation: bool = False,
        api_key: str | None = None,
        assistant_model: str = "gpt-5.1",
        input_image_file_id: str | None = None,
        output_path: str | None = None,
        # New gpt-image-2 params (kwargs so base.ImageProvider.generate_image sig stays stable)
        openai_model: str | None = None,
        quality: str | None = None,
        openai_output_format: str | None = None,
        openai_output_compression: int | None = None,
        background: str | None = None,
        moderation: str | None = None,
        style: str | None = None,
        n: int | None = None,
        allow_unknown_model: bool = False,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image using OpenAI gpt-image-2."""
        start_time = time.time()
        image_model = openai_model or DEFAULT_OPENAI_IMAGE_MODEL

        try:
            image_model = self._resolve_model(openai_model, allow_unknown=allow_unknown_model)
            api_key = self._get_api_key(api_key)

            # Validate and normalize
            validated = await self.validate_params(
                prompt,
                size,
                aspect_ratio,
                quality=quality,
                openai_output_format=openai_output_format,
                openai_output_compression=openai_output_compression,
                background=background,
                moderation=moderation,
                style=style,
                n=n,
                model=image_model,
                allow_unknown_model=allow_unknown_model,
            )
            size = str(validated["size"])
            quality = validated.get("quality", quality)
            output_format = validated.get("output_format", openai_output_format)
            output_compression = validated.get("output_compression", openai_output_compression)
            background = validated.get("background", background)
            moderation = validated.get("moderation", moderation)
            style = validated.get("style", style)
            n = validated.get("n", n)

            explicit_extension = self._explicit_output_extension(output_path)
            if explicit_extension is not None:
                if explicit_extension not in OPENAI_OUTPUT_FORMATS:
                    raise ValueError("Explicit output file must end in .png, .jpeg/.jpg, or .webp.")
                if output_format is None:
                    output_format = explicit_extension
                elif explicit_extension != output_format:
                    raise ValueError(
                        f"output_path extension '.{explicit_extension}' does not match "
                        f"openai_output_format '{output_format}'."
                    )

            if input_image_file_id is not None:
                raise ValueError(
                    "input_image_file_id is unsupported. Continue with the conversation_id "
                    "returned by the prior generation, or use edit_image with a local file."
                )

            should_persist = persist_conversation or conversation_id is not None
            previous_image_b64: str | None = None
            if conversation_id is not None:
                previous_image_b64 = await self._get_last_image_from_conversation(conversation_id)
                if previous_image_b64 is None:
                    raise ValueError(
                        f"Conversation '{conversation_id}' has no prior image to refine."
                    )
            else:
                conversation_id = self._generate_conversation_id()
            planned_output_paths = self._plan_output_paths(output_path, n or 1)

            if reference_images:
                raise ValueError(
                    "OpenAI image generation cannot apply reference_images. "
                    "Use edit_image for OpenAI image-to-image work or choose Gemini."
                )

            # Pick the code path
            refined_prompt: str | None = None
            if previous_image_b64 is not None:
                image_response = await self._call_images_edit_bytes(
                    api_key=api_key,
                    model=image_model,
                    prompt=prompt,
                    image_b64=previous_image_b64,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                    background=background,
                    n=n,
                )
            elif enable_enhancement:
                result = await self._call_chat_completions_refinement(
                    prompt=prompt,
                    api_key=api_key,
                    conversation_id=conversation_id,
                    assistant_model=assistant_model,
                    image_model=image_model,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                    background=background,
                    moderation=moderation,
                    style=style,
                    n=n,
                )
                image_response = result.get("image_response", {})
                refined_prompt = result.get("refined_prompt")
            else:
                image_response = await self._call_images_generate_direct(
                    api_key=api_key,
                    model=image_model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                    background=background,
                    moderation=moderation,
                    style=style,
                    n=n,
                )

            # Parse images
            entries = self._extract_image_data(image_response)
            usage = self._extract_usage(image_response)

            image_path: Path | None = None
            additional_paths: list[Path] = []
            response_revised: str | None = None

            valid_entries = [e for e in entries if "b64_json" in e]
            if not valid_entries:
                raise ValueError("OpenAI returned no usable image data.")

            image_extension = str(image_response.get("output_format") or output_format or "png")
            # Save all images concurrently (each write is offloaded to a
            # thread); asyncio.gather preserves order.
            saved_paths = await asyncio.gather(
                *(
                    self._save_image(
                        entry["b64_json"],
                        prompt,
                        planned_output_paths[index],
                        extension=image_extension,
                    )
                    for index, entry in enumerate(valid_entries)
                )
            )
            image_path = saved_paths[0]
            additional_paths = list(saved_paths[1:])
            if valid_entries[0].get("revised_prompt"):
                response_revised = valid_entries[0]["revised_prompt"]
            # Persist the user instruction and first output for real edit
            # chaining on the next conversational turn.
            if should_persist:
                await self._store_conversation_turn(
                    conversation_id,
                    prompt,
                    {"type": "image_generated", "prompt": prompt},
                    valid_entries[0]["b64_json"],
                    {
                        "size": size,
                        "model": image_model,
                        "quality": quality,
                        "output_format": output_format,
                    },
                )

            generation_time = time.time() - start_time
            actual_size = str(image_response.get("size") or size)

            return ImageResult(
                success=True,
                provider=self.name,
                model=image_model,
                image_path=image_path,
                additional_paths=additional_paths or None,
                prompt=prompt,
                enhanced_prompt=refined_prompt or response_revised,
                size=actual_size,
                aspect_ratio=self._aspect_ratio_from_size(actual_size),
                quality=image_response.get("quality") or quality,
                output_format=image_response.get("output_format") or output_format,
                background=image_response.get("background") or background,
                conversation_id=conversation_id if should_persist else None,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                usage_tokens=usage,
            )

        except Exception as e:
            logger.error("OpenAI image generation failed error_type=%s", type(e).__name__)
            provider_error = e if isinstance(e, ProviderError) else None
            safe_error = (
                provider_error.user_message
                if provider_error
                else str(e)
                if isinstance(e, ValueError)
                else "OpenAI image generation failed."
            )
            return ImageResult(
                success=False,
                provider=self.name,
                model=image_model,
                prompt=prompt,
                error=safe_error,
                error_code=provider_error.code if provider_error else None,
                error_status=provider_error.status_code if provider_error else None,
                error_request_id=provider_error.request_id if provider_error else None,
                error_retryable=provider_error.retryable if provider_error else None,
            )

    @staticmethod
    def _validate_allowed_input_path(path: Path) -> None:
        """Require local edit inputs to stay inside explicitly allowed roots."""
        settings = get_settings()
        if settings.allowed_input_roots:
            roots = [Path(root).expanduser().resolve() for root in settings.allowed_input_roots]
        else:
            from ..config.paths import get_base_output_directory

            roots = [get_base_output_directory().resolve()]

        if not any(path == root or path.is_relative_to(root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise ValueError(
                f"Input path is outside IMAGEN_MCP_ALLOWED_INPUT_ROOTS. Allowed: {allowed}"
            )

    @staticmethod
    def _read_validated_image(
        path: Path, *, mask: bool = False
    ) -> tuple[bytes, str, tuple[int, int]]:
        """Read an actual bounded image, rejecting arbitrary-file uploads."""
        from PIL import Image

        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"Input image exceeds the 50 MB limit: {path}")

        image_bytes = path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                detected_format = (image.format or "").upper()
                dimensions = image.size
                image.verify()
        except Exception as e:
            raise ValueError(f"Input is not a valid image: {path}") from e

        if mask:
            if detected_format != "PNG" or path.suffix.lower() != ".png":
                raise ValueError("Edit mask must be a valid PNG image.")
            return image_bytes, "image/png", dimensions

        mime_by_format = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
        suffix_by_format = {
            "PNG": {".png"},
            "JPEG": {".jpg", ".jpeg"},
            "WEBP": {".webp"},
        }
        if detected_format not in mime_by_format:
            raise ValueError("Edit source must be a PNG, JPEG, or WebP image.")
        if path.suffix.lower() not in suffix_by_format[detected_format]:
            raise ValueError("Edit source extension does not match its decoded image format.")
        return image_bytes, mime_by_format[detected_format], dimensions

    async def edit_image(
        self,
        *,
        prompt: str,
        image_path: str,
        mask_path: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        openai_output_format: str | None = None,
        openai_output_compression: int | None = None,
        input_fidelity: str | None = None,
        n: int | None = None,
        openai_model: str | None = None,
        api_key: str | None = None,
        output_path: str | None = None,
        allow_unknown_model: bool = False,
    ) -> ImageResult:
        """Edit an image via /images/edits with a GPT Image model.

        This is the right entry point for image-to-image workflows on
        OpenAI (reference-image-style consistency). gpt-image-2 already uses
        high input fidelity, so the request omits ``input_fidelity``. Older
        models retain the configurable legacy parameter.
        """
        start_time = time.time()
        image_model = openai_model or DEFAULT_OPENAI_IMAGE_MODEL

        try:
            image_model = self._resolve_model(openai_model, allow_unknown=allow_unknown_model)
            api_key = self._get_api_key(api_key)

            # gpt-image-2 supports the same constrained arbitrary sizes for
            # generation and editing.  Preserve the historical edits enum for
            # callers that pin an older model.
            if self._is_gpt_image_2(image_model):
                size = self._validate_size(
                    size or get_settings().default_openai_size,
                    image_model,
                )
            elif size:
                size = size.strip().replace("X", "x")
                if size not in OPENAI_EDIT_SIZES:
                    raise ValueError(
                        f"Invalid size '{size}' for /images/edits with {image_model}. "
                        f"Supported: {', '.join(OPENAI_EDIT_SIZES)}"
                    )
            else:
                size = "auto"

            if quality is not None and quality not in OPENAI_QUALITY_OPTIONS:
                raise ValueError(
                    f"Invalid quality '{quality}'. Supported: {', '.join(OPENAI_QUALITY_OPTIONS)}"
                )
            if background is not None:
                background = self._validate_background(background, image_model)
            if (
                openai_output_format is not None
                and openai_output_format not in OPENAI_OUTPUT_FORMATS
            ):
                raise ValueError(
                    f"Invalid output_format '{openai_output_format}'. "
                    f"Supported: {', '.join(OPENAI_OUTPUT_FORMATS)}"
                )
            explicit_extension = self._explicit_output_extension(output_path)
            if explicit_extension is not None:
                if explicit_extension not in OPENAI_OUTPUT_FORMATS:
                    raise ValueError("Explicit output file must end in .png, .jpeg/.jpg, or .webp.")
                if openai_output_format is None:
                    openai_output_format = explicit_extension
                elif explicit_extension != openai_output_format:
                    raise ValueError(
                        f"output_path extension '.{explicit_extension}' does not match "
                        f"openai_output_format '{openai_output_format}'."
                    )
            if input_fidelity is not None and input_fidelity not in OPENAI_INPUT_FIDELITY_OPTIONS:
                raise ValueError(
                    f"Invalid input_fidelity '{input_fidelity}'. "
                    f"Supported: {', '.join(OPENAI_INPUT_FIDELITY_OPTIONS)}"
                )
            request_input_fidelity: str | None
            if self._is_gpt_image_2(image_model):
                request_input_fidelity = None
                if input_fidelity is not None:
                    logger.info(
                        "Omitting input_fidelity=%s for gpt-image-2; "
                        "inputs are always high fidelity.",
                        input_fidelity,
                    )
            else:
                request_input_fidelity = input_fidelity or DEFAULT_OPENAI_INPUT_FIDELITY
            if n is not None:
                n = int(n)
                if not (1 <= n <= OPENAI_MAX_N):
                    raise ValueError(f"n must be between 1 and {OPENAI_MAX_N}.")

            # Resolve + read files (async to avoid blocking the event loop)
            img_path = Path(image_path).expanduser().resolve()
            if not img_path.is_file():
                raise ValueError(f"Source image not found: {img_path}")
            self._validate_allowed_input_path(img_path)
            if output_path:
                requested_output = Path(output_path).expanduser().resolve()
                if requested_output == img_path:
                    raise ValueError("Edit output_path must not overwrite the source image.")
            planned_output_paths = self._plan_output_paths(output_path, n or 1)

            image_bytes, image_mime, image_dimensions = await asyncio.to_thread(
                self._read_validated_image, img_path
            )

            mask_bytes: bytes | None = None
            mask_mime: str | None = None
            if mask_path:
                m_path = Path(mask_path).expanduser().resolve()
                if not m_path.is_file():
                    raise ValueError(f"Mask not found: {m_path}")
                self._validate_allowed_input_path(m_path)
                mask_bytes, mask_mime, mask_dimensions = await asyncio.to_thread(
                    self._read_validated_image, m_path, mask=True
                )
                if mask_dimensions != image_dimensions:
                    raise ValueError("Edit mask dimensions must match the source image.")

            # Build multipart form
            files: dict[str, Any] = {
                "image": (img_path.name, image_bytes, image_mime),
            }
            if mask_bytes is not None:
                files["mask"] = ("mask.png", mask_bytes, mask_mime)

            form_data: dict[str, Any] = {
                "model": image_model,
                "prompt": prompt,
                "size": size,
                # gpt-image-2 returns b64_json in data[] by default;
                # the legacy "response_format" param is not supported.
            }
            if request_input_fidelity is not None:
                form_data["input_fidelity"] = request_input_fidelity
            if quality is not None:
                form_data["quality"] = quality
            if background is not None:
                form_data["background"] = background
            if openai_output_format is not None:
                form_data["output_format"] = openai_output_format
            if openai_output_compression is not None:
                form_data["output_compression"] = int(openai_output_compression)
            if n is not None and n > 1:
                form_data["n"] = str(n)

            logger.info(
                "Calling /images/edits model=%s size=%s fidelity=%s n=%s",
                image_model,
                size,
                request_input_fidelity or "automatic-high",
                n or 1,
            )
            image_response = await self._make_api_request(
                endpoint="/images/edits",
                api_key=api_key,
                files=files,
                data=form_data,
            )

            entries = self._extract_image_data(image_response)
            usage = self._extract_usage(image_response)

            saved_path: Path | None = None
            additional_paths: list[Path] = []
            response_revised: str | None = None
            valid_entries = [e for e in entries if "b64_json" in e]
            if not valid_entries:
                raise ValueError("OpenAI returned no usable edited image data.")

            image_extension = str(
                image_response.get("output_format") or openai_output_format or "png"
            )
            saved_paths = await asyncio.gather(
                *(
                    self._save_image(
                        entry["b64_json"],
                        prompt,
                        planned_output_paths[index],
                        extension=image_extension,
                    )
                    for index, entry in enumerate(valid_entries)
                )
            )
            saved_path = saved_paths[0]
            additional_paths = list(saved_paths[1:])
            if valid_entries[0].get("revised_prompt"):
                response_revised = valid_entries[0]["revised_prompt"]

            generation_time = time.time() - start_time

            return ImageResult(
                success=True,
                provider=self.name,
                model=image_model,
                image_path=saved_path,
                additional_paths=additional_paths or None,
                prompt=prompt,
                enhanced_prompt=response_revised,
                size=image_response.get("size") or size,
                quality=image_response.get("quality") or quality,
                output_format=image_response.get("output_format") or openai_output_format,
                background=image_response.get("background") or background,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                usage_tokens=usage,
            )

        except Exception as e:
            logger.error("OpenAI image edit failed error_type=%s", type(e).__name__)
            provider_error = e if isinstance(e, ProviderError) else None
            safe_error = (
                provider_error.user_message
                if provider_error
                else str(e)
                if isinstance(e, ValueError)
                else "OpenAI image editing failed."
            )
            return ImageResult(
                success=False,
                provider=self.name,
                model=image_model,
                prompt=prompt,
                error=safe_error,
                error_code=provider_error.code if provider_error else None,
                error_status=provider_error.status_code if provider_error else None,
                error_request_id=provider_error.request_id if provider_error else None,
                error_retryable=provider_error.retryable if provider_error else None,
            )


# Keep legacy default export path used by tests / downstream code
__all__ = [
    "OpenAIProvider",
    "DEFAULT_OPENAI_BACKGROUND",
    "DEFAULT_OPENAI_MODERATION",
    "DEFAULT_OPENAI_OUTPUT_FORMAT",
    "DEFAULT_OPENAI_QUALITY",
]

# Module-level marker: base64 is imported for potential in-memory workflows
# that the edit path might grow (stream decoded bytes into files directly
# without saving base64 to disk first). Currently unused; keep available.
_ = base64
