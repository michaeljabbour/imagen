"""Google Gemini 3 image provider implementation.

Supports the active GA Gemini 3.1 Flash Image, Gemini 3 Pro Image, and Gemini
3.1 Flash Lite Image models with model-aware resolution, reference-image, and
Search constraints.
"""

import asyncio
import base64
import io
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, cast
from uuid import uuid4

import httpx

from ..config.constants import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    GEMINI_ASPECT_RATIOS,
    GEMINI_FLASH_EXTENDED_ASPECT_RATIOS,
    GEMINI_MAX_PROMPT_LENGTH,
    GEMINI_MAX_REFERENCE_IMAGES,
    GEMINI_MODEL_ALIASES,
    GEMINI_MODELS,
    GEMINI_RETIRED_MODEL_MIGRATIONS,
    GEMINI_SIZES,
)
from ..config.settings import get_settings
from ..exceptions import AuthenticationError, GenerationError, ProviderError, RateLimitError
from .base import ImageProvider, ImageResult, ProviderCapabilities

logger = logging.getLogger(__name__)

# Dedicated thread pool for the (synchronous) google-genai SDK calls. Isolating
# the long-running generate_content calls from the short asyncio.to_thread
# filesystem/DB writes keeps a burst of image generations from starving the
# default executor (and vice-versa).
_gemini_executor: ThreadPoolExecutor | None = None
_MAX_REFERENCE_DECODED_BYTES = 20 * 1024 * 1024
_MAX_REFERENCE_TOTAL_BYTES = 50 * 1024 * 1024
_MAX_REFERENCE_DIMENSION = 8192
_MAX_REFERENCE_PIXELS = 40_000_000


def _get_gemini_executor() -> ThreadPoolExecutor:
    global _gemini_executor
    if _gemini_executor is None:
        _gemini_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini-sdk")
    return _gemini_executor


# Lazy import for google-genai (may not be installed)
genai: Any = None
types: Any = None
Image: Any = None


def _import_dependencies() -> None:
    """Lazily import Gemini dependencies."""
    global genai, types, Image
    if genai is None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
            from PIL import Image as _Image

            genai = _genai
            types = _types
            Image = _Image
            # Explicit default — guards against decompression bombs from
            # untrusted reference images or conversation-history payloads.
            Image.MAX_IMAGE_PIXELS = 89_478_485
        except ImportError as e:
            raise ImportError(
                "Gemini provider requires google-genai and pillow packages. "
                "Install with: pip install google-genai pillow"
            ) from e


class GeminiProvider(ImageProvider):
    """Google Gemini 3 image provider.

    Best for:
    - Photorealistic portraits and headshots
    - Product photography
    - High resolution (4K) output
    - Character consistency with reference images
    - Real-time data visualization (weather, stocks, events)
    - Multi-turn iterative refinement

    Features:
    - Up to 14 reference images (category limits vary by model)
    - Google Search grounding for real-time data
    - Optional minimal/high thinking on Gemini 3.1 Flash and Flash Lite
    - 10 baseline aspect ratios; Flash supports 4 additional extreme ratios
    - 1K, 2K, and 4K resolution support; Flash also supports 0.5K
    """

    def __init__(self, api_key: str | None = None):
        """Initialize Gemini provider."""
        self._api_key = api_key
        self._client = None
        self._active_api_key: str | None = None  # Track which key the client was created with

    def _ensure_initialized(self, api_key: str | None = None) -> None:
        """Ensure dependencies are imported and client is initialized."""
        _import_dependencies()

        resolved_key = api_key or self._api_key
        if not resolved_key:
            settings = get_settings()
            resolved_key = settings.get_gemini_api_key()

        # Reinitialize if key changed (e.g. per-request override)
        if self._client is not None and resolved_key != self._active_api_key:
            self._client = None

        if self._client is None:
            self._client = genai.Client(api_key=resolved_key)
            self._active_api_key = resolved_key

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini — Nano Banana 2 (default) / Pro / Lite"

    # Cached capabilities — constant across all instances, no need to rebuild per access.
    # Describes the Nano Banana family (Imagen 4 support was removed in
    # v0.3.0 — see the module docstring for rationale).
    _capabilities = ProviderCapabilities(
        name="gemini",
        display_name="Gemini Nano Banana 2 / Pro / Lite",
        supported_sizes=GEMINI_SIZES,
        supported_aspect_ratios=GEMINI_FLASH_EXTENDED_ASPECT_RATIOS,
        max_resolution="4K",
        supports_text_rendering=True,
        text_rendering_quality="good",  # "excellent" when routed to Nano Banana Pro (Thinking)
        supports_reference_images=True,
        max_reference_images=GEMINI_MAX_REFERENCE_IMAGES,
        supports_real_time_data=True,
        supports_thinking_mode=True,
        supports_multi_turn=True,
        typical_latency_seconds=None,
        cost_tier="standard",
        best_for=[
            "Photorealistic portraits and headshots",
            "Product photography",
            "High resolution (4K) output",
            "Character consistency with reference images (up to 14)",
            "Real-time data visualization (weather, stocks)",
            "Multi-turn iterative refinement",
            "Complex compositions with multiple subjects",
            "Cost-optimized 1K generation with Nano Banana Lite",
        ],
        not_recommended_for=[
            "Precise text rendering (OpenAI gpt-image-2 is better)",
            "Technical diagrams with detailed labels",
        ],
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate and normalize parameters for Gemini."""
        if len(prompt) > GEMINI_MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt too long. Maximum {GEMINI_MAX_PROMPT_LENGTH} characters for Gemini."
            )

        model_id = str(kwargs.get("model_id") or DEFAULT_GEMINI_IMAGE_MODEL)
        model_meta = GEMINI_MODELS.get(model_id)
        if model_meta is None:
            raise ValueError(f"Unsupported Gemini image model '{model_id}'.")
        supported_sizes = cast("list[str]", model_meta["supported_sizes"])

        # Validate/normalize size (must be uppercase K)
        if size:
            size = size.upper()
            # Convert OpenAI-style sizes to Gemini equivalents
            openai_to_gemini = {
                "1024X1024": "1K",
                "1024X1536": "2K",
                "1536X1024": "2K",
            }
            if size in openai_to_gemini:
                logger.info(f"Converting OpenAI size '{size}' to Gemini: {openai_to_gemini[size]}")
                size = openai_to_gemini[size]
            if size not in supported_sizes:
                raise ValueError(
                    f"Invalid size '{size}' for {model_id}. Supported sizes: "
                    f"{', '.join(supported_sizes)}"
                )
        else:
            configured_default = get_settings().default_gemini_size.upper()
            size = (
                configured_default
                if configured_default in supported_sizes
                else str(model_meta["default_size"])
            )

        supported_aspect_ratios = (
            GEMINI_FLASH_EXTENDED_ASPECT_RATIOS
            if model_id in {"gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"}
            else GEMINI_ASPECT_RATIOS
        )

        # Validate aspect ratio
        if aspect_ratio:
            if aspect_ratio not in supported_aspect_ratios:
                raise ValueError(
                    f"Invalid aspect ratio '{aspect_ratio}' for Gemini. "
                    f"Supported ratios: {', '.join(supported_aspect_ratios)}"
                )
        else:
            aspect_ratio = get_settings().default_gemini_aspect_ratio
            if aspect_ratio not in supported_aspect_ratios:
                raise ValueError(
                    f"Invalid configured Gemini aspect ratio '{aspect_ratio}'. "
                    f"Supported ratios: {', '.join(supported_aspect_ratios)}"
                )

        # Validate reference images count
        # Use `or []` because dict.get() returns None (not default) when key exists with None value
        reference_images = kwargs.get("reference_images") or []
        max_references = cast("int", model_meta["max_reference_images"])
        if len(reference_images) > max_references:
            raise ValueError(f"Too many reference images for {model_id}. Maximum {max_references}.")

        if kwargs.get("enable_google_search") and not model_meta.get(
            "supports_google_search", False
        ):
            raise ValueError(f"Google Search grounding is not supported by {model_id}.")

        thinking_level = kwargs.get("thinking_level")
        if thinking_level is not None:
            thinking_level = str(thinking_level).lower()
            supported_levels = cast("list[str]", model_meta.get("supported_thinking_levels", []))
            if thinking_level not in supported_levels:
                raise ValueError(
                    f"Thinking level '{thinking_level}' is not supported by {model_id}. "
                    f"Supported: {', '.join(supported_levels) or 'provider-managed only'}."
                )

        return {
            "prompt": prompt,
            "size": size,
            "aspect_ratio": aspect_ratio,
            "thinking_level": thinking_level,
        }

    def _resolve_model_id(self, model: str | None) -> str:
        """Resolve a user-provided model name/alias to a canonical Gemini
        model identifier.

        Accepts either:
        - a canonical model id from ``GEMINI_MODELS`` keys (e.g.
          ``"gemini-3.1-flash-image"``),
        - a friendly alias from ``GEMINI_MODEL_ALIASES`` (e.g. ``"nano-banana-2"``),
        - ``None`` (returns the default).

        Unknown names are rejected so a typo never silently bills a different
        model. Retired preview identifiers fail with an actionable GA migration.
        """
        if not model:
            return DEFAULT_GEMINI_IMAGE_MODEL

        if model in GEMINI_RETIRED_MODEL_MIGRATIONS:
            replacement = GEMINI_RETIRED_MODEL_MIGRATIONS[model]
            raise ValueError(
                f"Gemini image model '{model}' is retired. Pin the GA model "
                f"'{replacement}' explicitly."
            )

        # Try alias first
        if model in GEMINI_MODEL_ALIASES:
            return GEMINI_MODEL_ALIASES[model]

        if model in GEMINI_MODELS:
            return model

        raise ValueError(
            f"Unsupported Gemini image model '{model}'. Supported models: "
            f"{', '.join(sorted(GEMINI_MODELS))}."
        )

    @staticmethod
    def _is_retryable_api_error(error: Exception) -> bool:
        """Return whether a Gemini SDK failure is safe to retry."""
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            # A timed-out image render may still finish provider-side; retrying
            # risks duplicate work and multiplies the configured long timeout.
            return False

        # Keep google-genai lazy: this import occurs only while classifying an
        # SDK exception, after provider dependencies have already been loaded.
        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover - generation cannot run without SDK
            pass
        else:
            if isinstance(error, genai_errors.ClientError):
                return int(getattr(error, "code", 0)) == 429
            if isinstance(error, genai_errors.ServerError):
                return True
            if isinstance(error, genai_errors.APIError):
                return int(getattr(error, "code", 0)) >= 500

        if isinstance(error, httpx.TransportError):
            return True
        return isinstance(error, ConnectionError)

    @staticmethod
    def _structured_api_error(error: Exception) -> ProviderError | None:
        """Map Gemini SDK/transport failures to a safe stable error contract."""
        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover
            genai_errors = None  # type: ignore[assignment]

        if genai_errors is not None and isinstance(error, genai_errors.APIError):
            status = int(getattr(error, "code", 0)) or None
            if status in (401, 403):
                return AuthenticationError(
                    "Gemini authentication failed.",
                    provider="gemini",
                    status_code=status,
                    code=f"http_{status}",
                    user_message="Gemini authentication failed. Check the configured API key.",
                )
            if status == 429:
                return RateLimitError(
                    "Gemini rate limit exceeded.", provider="gemini", status_code=status
                )
            if status is not None and status >= 500:
                return ProviderError(
                    "Gemini service error.",
                    provider="gemini",
                    status_code=status,
                    code=f"http_{status}",
                    retryable=True,
                    user_message="Gemini is temporarily unavailable. Please try again.",
                )
            return GenerationError(
                "Gemini rejected the image request.",
                provider="gemini",
                status_code=status,
                code=f"http_{status or 400}",
                user_message="Gemini rejected the image request. Check its prompt and parameters.",
            )
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return ProviderError(
                "Gemini request timed out.",
                provider="gemini",
                code="timeout",
                user_message="Gemini image generation timed out; it was not retried.",
            )
        if isinstance(error, (httpx.TransportError, ConnectionError)):
            return ProviderError(
                "Gemini transport failure.",
                provider="gemini",
                code="transport_error",
                retryable=True,
                user_message="Could not reach Gemini. Please try again.",
            )
        return None

    def _decode_images(
        self, last_image_b64: str | None, reference_images: list[str] | None
    ) -> tuple[list[Any], list[Any]]:
        """Decode history + reference images into PIL objects (runs in a thread).

        base64-decoding and PIL-opening up to 14 multi-MB images is CPU/IO heavy
        and must not run on the event loop. Per-image failures are logged and
        skipped. Returns ``(image_objects, images_to_close)`` — the caller
        appends the objects to ``contents`` and closes them in its ``finally``.
        """
        objects: list[Any] = []
        to_close: list[Any] = []
        total_decoded_bytes = 0

        def _decode(encoded: str, label: str) -> Any:
            nonlocal total_decoded_bytes
            max_encoded = ((_MAX_REFERENCE_DECODED_BYTES + 2) // 3) * 4
            if len(encoded) > max_encoded:
                raise ValueError(f"{label} exceeds the 20 MB per-image limit.")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except Exception as e:
                raise ValueError(f"{label} is not valid base64 image data.") from e
            if len(image_bytes) > _MAX_REFERENCE_DECODED_BYTES:
                raise ValueError(f"{label} exceeds the 20 MB per-image limit.")
            total_decoded_bytes += len(image_bytes)
            if total_decoded_bytes > _MAX_REFERENCE_TOTAL_BYTES:
                raise ValueError("Reference images exceed the 50 MB aggregate limit.")

            image: Any = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    image = Image.open(io.BytesIO(image_bytes))
                    if (image.format or "").upper() not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError(f"{label} must be PNG, JPEG, or WebP.")
                    width, height = image.size
                    if max(width, height) > _MAX_REFERENCE_DIMENSION:
                        raise ValueError(f"{label} exceeds the 8192px dimension limit.")
                    if width * height > _MAX_REFERENCE_PIXELS:
                        raise ValueError(f"{label} exceeds the 40 megapixel limit.")
                    image.load()
                return image
            except Exception:
                if image is not None:
                    image.close()
                raise

        if last_image_b64:
            try:
                img = _decode(last_image_b64, "Conversation history image")
                objects.append(img)
                to_close.append(img)
            except Exception:
                for opened_image in to_close:
                    opened_image.close()
                raise

        if reference_images:
            for index, ref_b64 in enumerate(reference_images[:GEMINI_MAX_REFERENCE_IMAGES]):
                try:
                    img = _decode(ref_b64, f"Reference image {index + 1}")
                    objects.append(img)
                    to_close.append(img)
                except Exception:
                    for opened_image in to_close:
                        try:
                            opened_image.close()
                        except Exception:
                            pass
                    raise

        return objects, to_close

    @staticmethod
    def _normalize_generated_images(images_b64: list[str]) -> list[str]:
        """Validate Gemini image payloads and normalize every artifact to PNG.

        Gemini responses have returned JPEG bytes even when the surrounding
        response implied PNG.  The public MCP contract uses ``.png`` for
        Gemini artifacts, so trusting response labels would create files whose
        suffix disagrees with their contents.  Inspect the actual bytes and
        transcode supported non-PNG payloads before they reach persistence or
        conversation history.  This CPU-heavy work runs via ``to_thread``.
        """
        normalized: list[str] = []
        for index, encoded in enumerate(images_b64, start=1):
            try:
                raw = base64.b64decode(encoded, validate=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(raw)) as source:
                        source_format = (source.format or "").upper()
                        if source_format not in {"PNG", "JPEG", "WEBP"}:
                            raise ValueError("unsupported image encoding")
                        width, height = source.size
                        if max(width, height) > _MAX_REFERENCE_DIMENSION:
                            raise ValueError("image dimension exceeds 8192px")
                        if width * height > _MAX_REFERENCE_PIXELS:
                            raise ValueError("image exceeds 40 megapixels")
                        source.load()

                        if source_format == "PNG":
                            normalized_bytes = raw
                        else:
                            output = io.BytesIO()
                            if source.mode == "CMYK":
                                with source.convert("RGB") as converted:
                                    converted.save(output, format="PNG")
                            else:
                                source.save(output, format="PNG")
                            normalized_bytes = output.getvalue()
            except Exception as error:
                raise ValueError(
                    f"Gemini returned invalid image data for artifact {index}."
                ) from error
            normalized.append(base64.b64encode(normalized_bytes).decode())
        return normalized

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
        enable_google_search: bool = False,
        api_key: str | None = None,
        model: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image using Gemini.

        All supported GA models use ``generateContent``. Feature and size
        constraints are read from the model registry before the SDK call.
        """
        model_id = model or DEFAULT_GEMINI_IMAGE_MODEL
        start_time = time.time()

        try:
            model_id = self._resolve_model_id(model)
            explicit_extension = self._explicit_output_extension(output_path)
            if explicit_extension not in (None, "png"):
                raise ValueError("Gemini output files must use a .png extension.")
            planned_output_paths = self._plan_output_paths(output_path, 1)

            # Ensure initialized
            self._ensure_initialized(api_key)
            assert self._client is not None, "Gemini client not initialized"

            # Validate parameters (Nano Banana shape)
            validated = await self.validate_params(
                prompt,
                size,
                aspect_ratio,
                model_id=model_id,
                reference_images=reference_images,
                enable_google_search=enable_google_search,
                **kwargs,
            )
            size = validated["size"]
            aspect_ratio = validated["aspect_ratio"]
            thinking_level = validated.get("thinking_level")

            should_persist = persist_conversation or conversation_id is not None
            requested_conversation_id = conversation_id
            # Generate a working ID; expose/store it only for conversational mode.
            conversation_id = conversation_id or f"gemini_{uuid4().hex[:12]}"

            # Build contents list; track PIL images for cleanup
            contents: list[Any] = []
            pil_images_to_close: list[Any] = []

            try:
                # Decode the previous (history) image + any reference images off
                # the event loop — base64-decoding and PIL-opening up to 14
                # multi-MB images would otherwise block all other requests.
                last_image_b64 = (
                    await self._get_last_image_from_conversation(conversation_id)
                    if requested_conversation_id is not None
                    else None
                )
                if requested_conversation_id is not None and last_image_b64 is None:
                    raise ValueError(
                        f"Conversation '{requested_conversation_id}' has no prior image to refine."
                    )
                max_references = int(GEMINI_MODELS[model_id]["max_reference_images"])
                if last_image_b64 and len(reference_images or []) >= max_references:
                    raise ValueError(
                        f"Conversation history plus reference_images exceeds the {model_id} "
                        f"limit of {max_references} total images."
                    )
                image_objects, pil_images_to_close = await asyncio.to_thread(
                    self._decode_images, last_image_b64, reference_images
                )
                contents.extend(image_objects)
                if last_image_b64 and image_objects:
                    logger.info(f"Added prev image from conv {conversation_id} as context")

                # Add prompt
                contents.append(prompt)

                # Build config
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=size,
                )

                config_args: dict[str, Any] = {
                    "response_modalities": ["TEXT", "IMAGE"],
                    "image_config": image_config,
                }

                # Add Google Search grounding if enabled
                if enable_google_search:
                    config_args["tools"] = [{"google_search": {}}]
                if thinking_level:
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_level=str(thinking_level).upper()
                    )

                config = types.GenerateContentConfig(**config_args)

                logger.info(
                    f"Generating image with Gemini model={model_id}, size={size}, "
                    f"aspect_ratio={aspect_ratio}"
                )

                # Acquire rate limit before making request
                await self._acquire_rate_limit()

                # Generate content (SDK is synchronous, run in executor)
                # Wrap with timeout and retry to handle transient failures
                settings = get_settings()

                client = self._client  # Bind for closure (already asserted non-None)

                async def _do_generate() -> Any:
                    loop = asyncio.get_running_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            _get_gemini_executor(),
                            partial(
                                client.models.generate_content,
                                model=model_id,
                                contents=contents,
                                config=config,
                            ),
                        ),
                        timeout=settings.request_timeout,
                    )

                response = await self._retry_with_backoff(
                    _do_generate,
                    is_retryable=self._is_retryable_api_error,
                )
            finally:
                # Release PIL image memory immediately after API call
                for img in pil_images_to_close:
                    try:
                        img.close()
                    except Exception:
                        pass

            # Extract content from response
            extraction = self._extract_content(response)

            if not extraction["images"]:
                raise ValueError("No image data found in Gemini API response")
            if enable_google_search:
                grounding = extraction.get("grounding_metadata") or {}
                if not grounding.get("search_suggestions_html"):
                    raise ValueError(
                        "Gemini Search grounding returned no Search Suggestions; "
                        "the result cannot be displayed compliantly."
                    )

            # Save all images concurrently. Nano Banana usually returns one
            # image per call, but if the model returns several (batch), the
            # extras go to additional_paths. asyncio.gather preserves order.
            images_b64 = await asyncio.to_thread(
                self._normalize_generated_images, extraction["images"]
            )
            image_b64 = images_b64[0]
            if len(images_b64) > len(planned_output_paths):
                planned_output_paths = self._plan_output_paths(output_path, len(images_b64))
            saved_paths = await asyncio.gather(
                *(
                    self._save_image(b64, prompt, planned_output_paths[index], extension="png")
                    for index, b64 in enumerate(images_b64)
                )
            )
            image_path = saved_paths[0]
            additional_paths: list[Any] = list(saved_paths[1:])

            if should_persist:
                await self._store_conversation_turn(
                    conversation_id,
                    prompt,
                    {"type": "image_generated", "prompt": prompt},
                    image_b64,
                    {"size": size, "aspect_ratio": aspect_ratio, "model": model_id},
                )

            generation_time = time.time() - start_time

            # Don't carry image_base64 on the result when we already
            # persisted the file — avoids keeping ~4-12 MB in memory
            # for the lifetime of the result object.
            return ImageResult(
                success=True,
                provider=self.name,
                model=model_id,
                image_path=image_path,
                additional_paths=additional_paths or None,
                prompt=prompt,
                size=size,
                aspect_ratio=aspect_ratio,
                output_format="png",
                conversation_id=conversation_id if should_persist else None,
                timestamp=datetime.now(),
                generation_time_seconds=generation_time,
                thoughts=extraction.get("thoughts"),
                grounding_metadata=extraction.get("grounding_metadata"),
            )

        except Exception as e:
            logger.error("Gemini image generation failed error_type=%s", type(e).__name__)
            provider_error = self._structured_api_error(e)
            safe_error = (
                provider_error.user_message
                if provider_error
                else str(e)
                if isinstance(e, ValueError)
                else "Gemini image generation failed."
            )
            return ImageResult(
                success=False,
                provider=self.name,
                model=model or DEFAULT_GEMINI_IMAGE_MODEL,
                prompt=prompt,
                error=safe_error,
                error_code=provider_error.code if provider_error else None,
                error_status=provider_error.status_code if provider_error else None,
                error_request_id=provider_error.request_id if provider_error else None,
                error_retryable=provider_error.retryable if provider_error else None,
            )

    def _extract_content(self, response: Any) -> dict[str, Any]:
        """Extract images, text, and thoughts from Gemini response.

        Raw image bytes are encoded for transport here. Final-image validation
        and PNG normalization happen off the event loop before persistence.
        """
        images: list[str] = []
        text_parts: list[str] = []
        thoughts: list[dict[str, Any]] = []

        try:
            for idx, part in enumerate(response.parts):
                is_thought = getattr(part, "thought", False)

                # Extract image data
                if hasattr(part, "inline_data") and part.inline_data:
                    try:
                        inline_data = part.inline_data
                        image_bytes = inline_data.data

                        if is_thought:
                            # Never expose hidden reasoning images over MCP.
                            # Retain only bounded operational telemetry.
                            thoughts.append(
                                {
                                    "type": "image",
                                    "index": len(thoughts),
                                    "byte_length": len(image_bytes),
                                }
                            )
                        else:
                            # Encode final image bytes directly — avoids a PIL
                            # decode/re-encode round trip.
                            image_b64 = base64.b64encode(image_bytes).decode()
                            images.append(image_b64)
                    except Exception as e:
                        logger.error(
                            "Could not extract image from part %d error_type=%s",
                            idx,
                            type(e).__name__,
                        )

                # Extract text
                if hasattr(part, "text") and part.text:
                    if is_thought:
                        thoughts.append(
                            {
                                "type": "text",
                                "index": len(thoughts),
                                "character_count": len(part.text),
                            }
                        )
                    else:
                        text_parts.append(part.text)

        except Exception as e:
            logger.error("Error extracting content from response error_type=%s", type(e).__name__)

        result: dict[str, Any] = {
            "images": images,
            "text": text_parts,
            "thoughts": thoughts if thoughts else None,
        }

        # Include grounding metadata if available
        # In google-genai 2.x, grounding metadata lives on the first candidate,
        # not on the top-level response. Read it defensively via getattr so
        # mocked responses without a `candidates` attribute simply yield no
        # grounding metadata (and so Search-grounding citations aren't silently
        # dropped on real responses).
        candidates = getattr(response, "candidates", None)
        candidate = candidates[0] if candidates else None
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding is not None:
            search_entry = getattr(grounding, "search_entry_point", None)
            rendered_content = getattr(search_entry, "rendered_content", None)
            sources: list[dict[str, str]] = []
            for chunk in getattr(grounding, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                if uri:
                    sources.append(
                        {
                            "uri": str(uri),
                            "title": str(getattr(web, "title", None) or uri),
                        }
                    )
            citations: list[dict[str, Any]] = []
            for support in getattr(grounding, "grounding_supports", None) or []:
                segment = getattr(support, "segment", None)
                citations.append(
                    {
                        "text": str(getattr(segment, "text", "")),
                        "source_indices": list(
                            getattr(support, "grounding_chunk_indices", None) or []
                        ),
                    }
                )
            result["grounding_metadata"] = {
                "search_suggestions_html": str(rendered_content) if rendered_content else None,
                "sources": sources,
                "citations": citations,
            }

        return result

    async def close(self) -> None:
        """Clean up resources."""
        # genai SDK handles cleanup automatically
        self._client = None
