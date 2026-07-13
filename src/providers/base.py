"""
Abstract base class for image generation providers.

This module defines the interface that all providers must implement,
ensuring consistent behavior across OpenAI, Gemini, and future providers.
"""

import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config.constants import MAX_RETRIES
from ..config.paths import resolve_output_path

logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    """Describes what a provider can do."""

    name: str
    display_name: str

    # Size/resolution support
    supported_sizes: list[str]
    supported_aspect_ratios: list[str]
    max_resolution: str

    # Feature support
    supports_text_rendering: bool = True
    text_rendering_quality: str = "good"  # excellent, good, fair

    supports_reference_images: bool = False
    max_reference_images: int = 0

    supports_real_time_data: bool = False
    supports_thinking_mode: bool = False
    supports_multi_turn: bool = True

    # Performance characteristics
    typical_latency_seconds: float | None = None
    cost_tier: str = "standard"  # low, standard, premium

    # Best use cases
    best_for: list[str] = field(default_factory=list)
    not_recommended_for: list[str] = field(default_factory=list)


@dataclass
class ImageResult:
    """Result from image generation."""

    success: bool
    provider: str
    model: str

    # Image data (one of these will be set)
    image_path: Path | None = None
    image_base64: str | None = None
    image_url: str | None = None

    # Multi-image output (when n > 1). When present, image_path is the first.
    additional_paths: list[Path] | None = None

    # Metadata
    prompt: str = ""
    # Populated from OpenAI revised_prompt or dialogue refinement
    enhanced_prompt: str | None = None
    size: str | None = None
    aspect_ratio: str | None = None
    quality: str | None = None
    output_format: str | None = None  # Image encoding: png / jpeg / webp
    background: str | None = None  # transparent / opaque / auto

    # Conversation tracking
    conversation_id: str | None = None
    file_id: str | None = None  # Provider-specific file reference

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    generation_time_seconds: float | None = None

    # Cost / usage tracking (OpenAI gpt-image-2 returns usage in the response)
    usage_tokens: dict[str, int] | None = None  # {input_tokens, output_tokens, total_tokens, ...}

    # Additional data
    thoughts: list[dict[str, Any]] | None = None  # Gemini thinking mode
    grounding_metadata: dict[str, Any] | None = None  # Gemini search grounding
    verification_result: dict[str, Any] | None = None

    # Error info
    error: str | None = None
    error_code: str | None = None
    error_status: int | None = None
    error_request_id: str | None = None
    error_retryable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "provider": self.provider,
            "model": self.model,
            "image_path": str(self.image_path) if self.image_path else None,
            "additional_paths": (
                [str(p) for p in self.additional_paths] if self.additional_paths else None
            ),
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "quality": self.quality,
            "output_format": self.output_format,
            "background": self.background,
            "conversation_id": self.conversation_id,
            "file_id": self.file_id,
            "timestamp": self.timestamp.isoformat(),
            "generation_time_seconds": self.generation_time_seconds,
            "usage_tokens": self.usage_tokens,
            "thoughts": self.thoughts,
            "grounding_metadata": self.grounding_metadata,
            "verification_result": self.verification_result,
            "error": self.error,
            "error_code": self.error_code,
            "error_status": self.error_status,
            "error_request_id": self.error_request_id,
            "error_retryable": self.error_retryable,
        }


class ImageProvider(ABC):
    """
    Abstract base class for image generation providers.

    All providers must implement these methods to ensure consistent
    behavior across the MCP server.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'gemini')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and feature support."""
        pass

    @abstractmethod
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
        output_path: str | None = None,
        **kwargs: Any,
    ) -> ImageResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of desired image
            size: Image size (provider-specific format)
            aspect_ratio: Aspect ratio (provider-specific format)
            conversation_id: ID for multi-turn conversation
            reference_images: List of base64-encoded reference images
            enable_enhancement: Whether to enhance the prompt
            persist_conversation: Persist prompt/image history for later refinement
            output_path: Optional path to save the image
            **kwargs: Provider-specific parameters

        Returns:
            ImageResult with generated image data
        """
        pass

    @abstractmethod
    async def validate_params(
        self,
        prompt: str,
        size: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Validate and normalize parameters for this provider.

        Args:
            prompt: Text prompt to validate
            size: Size to validate
            aspect_ratio: Aspect ratio to validate
            **kwargs: Additional parameters

        Returns:
            Dictionary of validated/normalized parameters

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    async def close(self) -> None:  # noqa: B027
        """Clean up provider resources."""
        pass  # Default implementation does nothing

    def get_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get list of recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summaries
        """
        from ..services.conversation_store import get_conversation_store

        store = get_conversation_store()
        return store.list_conversations(provider=self.name, limit=limit)

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"{self.name}_{uuid4().hex[:12]}"

    @staticmethod
    def _explicit_output_extension(output_path: str | None) -> str | None:
        """Return an explicit file destination's normalized extension."""
        if not output_path or not output_path.strip():
            return None
        raw = output_path.strip()
        path = Path(raw).expanduser()
        is_directory = (
            raw.endswith(("/", "\\")) or (path.exists() and path.is_dir()) or not path.suffix
        )
        if is_directory:
            return None
        extension = path.suffix.lower().lstrip(".")
        return "jpeg" if extension == "jpg" else extension

    @staticmethod
    def _plan_output_paths(output_path: str | None, count: int) -> list[str | None]:
        """Plan collision-free destinations for a multi-image response.

        Directory destinations can reuse the same value because every saved
        image gets a UUID filename. An explicit file destination is indexed as
        ``name.ext``, ``name_2.ext``, ... and checked before any provider call.
        """
        if count < 1:
            raise ValueError("Image count must be at least 1.")
        if not output_path or not output_path.strip():
            return [output_path] * count

        raw = output_path.strip()
        path = Path(raw).expanduser()
        is_directory = (
            raw.endswith(("/", "\\")) or (path.exists() and path.is_dir()) or not path.suffix
        )
        if is_directory:
            return [output_path] * count

        planned = [path]
        planned.extend(
            path.with_name(f"{path.stem}_{index}{path.suffix}") for index in range(2, count + 1)
        )
        existing = next((candidate for candidate in planned if candidate.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Refusing to overwrite existing output file: {existing}")
        return [str(candidate) for candidate in planned]

    def _resolve_and_save_sync(self, b64_data: str, output_path: str | None, filename: str) -> Path:
        """Resolve output path, decode, and write — called via ``asyncio.to_thread()``.

        All filesystem work (``mkdir``, ``write_bytes``) happens here so the
        event loop is never blocked.
        """
        save_path = resolve_output_path(output_path, default_filename=filename, provider=self.name)
        image_bytes = base64.b64decode(b64_data)
        # Exclusive creation prevents both accidental overwrites and races
        # between concurrent saves targeting the same explicit path.
        descriptor = os.open(
            save_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as image_file:
                image_file.write(image_bytes)
        except BaseException:
            with suppress(FileNotFoundError):
                save_path.unlink()
            raise
        return save_path

    async def _save_image(
        self,
        b64_data: str,
        prompt: str,
        output_path: str | None = None,
        *,
        extension: str = "png",
        result: ImageResult | None = None,
    ) -> Path:
        """
        Save base64 image to path or Downloads folder.

        Path resolution (``mkdir``), decoding, and file I/O are all offloaded
        to a thread so the event loop is never blocked (images can be
        multi-megabyte).

        Args:
            b64_data: Base64-encoded image data
            prompt: Original prompt (used for filename)
            output_path: Optional custom output path
            extension: Image encoding extension (png / jpeg / webp)
            result: If provided, ``image_path`` is set and ``image_base64``
                is released after the save to free memory.

        Returns:
            Path where image was saved
        """
        extension = extension.lower().lstrip(".")
        if extension not in {"png", "jpeg", "jpg", "webp"}:
            raise ValueError(f"Unsupported image filename extension: {extension}")

        # Generate default filename (cheap, stays on event loop)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid4().hex[:8]
        prompt_snippet = "".join(c for c in prompt[:30] if c.isalnum() or c == " ").strip()
        prompt_snippet = prompt_snippet.replace(" ", "_")[:20]
        filename = f"{self.name}_{timestamp}_{prompt_snippet}_{short_id}.{extension}"

        save_path = await asyncio.to_thread(
            self._resolve_and_save_sync, b64_data, output_path, filename
        )

        logger.info(f"Image saved to: {save_path}")

        # Release the multi-MB base64 string from memory now that
        # the image is persisted to disk.
        if result is not None:
            result.image_path = save_path
            result.image_base64 = None

        return save_path

    async def _retry_with_backoff(
        self,
        func: Any,
        *args: Any,
        max_retries: int = MAX_RETRIES,
        base_delay: float = 1.0,
        non_retryable: tuple[type[BaseException], ...] = (),
        is_retryable: Callable[[Exception], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with exponential backoff retry.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (doubles each attempt)
            non_retryable: Exception types that should fail immediately instead
                of being retried (e.g. timeouts on long-running renders, where
                retrying just multiplies the wait).
            is_retryable: Optional provider-specific predicate. When supplied,
                exceptions for which it returns false fail immediately.
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except non_retryable:
                # Fail fast — retrying won't help these.
                raise
            except Exception as e:
                if is_retryable is not None and not is_retryable(e):
                    raise
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Attempt %d/%d failed error_type=%s; retrying in %ss",
                        attempt + 1,
                        max_retries,
                        type(e).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise last_error if last_error else RuntimeError("Retry failed with no error")

    async def _acquire_rate_limit(self) -> None:
        """Acquire rate limit permission before making an API call."""
        from ..services.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()
        await limiter.acquire(self.name)

    async def _store_conversation_turn(
        self,
        conversation_id: str,
        user_content: Any,
        assistant_content: Any,
        image_base64: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a complete turn atomically and propagate storage failures."""
        from ..services.conversation_store import get_conversation_store

        store = get_conversation_store()
        await asyncio.to_thread(
            store.add_turn,
            conversation_id,
            self.name,
            user_content,
            assistant_content,
            image_base64,
            metadata,
        )

    async def _get_last_image_from_conversation(self, conversation_id: str) -> str | None:
        """
        Get the last generated image from a conversation (off the event loop).

        Args:
            conversation_id: Conversation ID

        Returns:
            Base64-encoded image or None
        """
        from ..services.conversation_store import get_conversation_store

        store = get_conversation_store()
        return await asyncio.to_thread(store.get_last_image, conversation_id)
