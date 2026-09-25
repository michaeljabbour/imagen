"""
Provider registry for imagen.

Manages provider instances and provides factory methods
for creating and accessing providers.
"""

import logging
from functools import lru_cache
from importlib.util import find_spec
from typing import Any

from ..config.settings import get_settings
from .base import ImageProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .selector import ProviderRecommendation, ProviderSelector

logger = logging.getLogger(__name__)


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


class ProviderRegistry:
    """
    Registry for image generation providers.

    Manages provider instances and provides methods for:
    - Getting specific providers by name
    - Auto-selecting providers based on prompt analysis
    - Listing available providers
    """

    def __init__(self) -> None:
        """Initialize the provider registry."""
        self._providers: dict[str, ImageProvider] = {}
        self._selector = ProviderSelector()
        self._settings = get_settings()

    def get_provider(self, name: str, *, api_key: str | None = None) -> ImageProvider:
        """
        Get a provider by name.

        Args:
            name: Provider name ("openai" or "gemini")
            api_key: Optional request-scoped API key override

        Returns:
            ImageProvider instance

        Raises:
            ValueError: If provider not found or not configured
        """
        name = name.lower()

        if name not in self.list_all_providers():
            raise ValueError(f"Unknown provider: {name}. Available: openai, gemini")

        if not self.is_provider_available(name, api_key=api_key):
            env_var = "OPENAI_API_KEY" if name == "openai" else "GEMINI_API_KEY"
            provider_label = "OpenAI" if name == "openai" else "Gemini"
            raise ValueError(
                f"{provider_label} provider not available. Set {env_var} environment "
                "variable or provide an API key override."
            )

        # Return cached instance if available
        if name in self._providers:
            return self._providers[name]

        # Create new instance
        if name == "openai":
            provider: ImageProvider = OpenAIProvider(api_key=api_key)
        elif name == "gemini":
            provider = GeminiProvider(api_key=api_key)

        # Only cache providers backed by environment configuration. Request-scoped
        # API keys should not become implicit credentials for later requests.
        if api_key is None:
            self._providers[name] = provider
        return provider

    def get_provider_for_prompt(
        self,
        prompt: str,
        *,
        size: str | None = None,
        reference_images: list[str] | None = None,
        enable_google_search: bool = False,
        explicit_provider: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
    ) -> tuple[ImageProvider, ProviderRecommendation]:
        """
        Get the best provider for a given prompt.

        Uses the ProviderSelector to analyze the prompt and choose
        the most appropriate provider.

        Args:
            prompt: Image generation prompt
            size: Requested size
            reference_images: Reference images (requires Gemini)
            enable_google_search: Enable Google Search grounding
            explicit_provider: User-specified provider override
            openai_api_key: Optional request-scoped OpenAI API key
            gemini_api_key: Optional request-scoped Gemini API key

        Returns:
            Tuple of (ImageProvider, ProviderRecommendation)
        """
        available = self.list_providers(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
        )
        # Get recommendation
        recommendation = self._selector.suggest_provider(
            prompt,
            size=size,
            reference_images=reference_images,
            enable_google_search=enable_google_search,
            explicit_provider=explicit_provider,
            available_providers=available,
        )

        # Get the provider
        provider_api_key = openai_api_key if recommendation.provider == "openai" else gemini_api_key
        provider = self.get_provider(recommendation.provider, api_key=provider_api_key)

        return provider, recommendation

    def list_providers(
        self,
        *,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
    ) -> list[str]:
        """List all available provider names (with API keys and dependencies)."""
        return [
            p
            for p in self.list_all_providers()
            if self.is_provider_available(
                p,
                api_key=openai_api_key if p == "openai" else gemini_api_key,
            )
        ]

    def list_all_providers(self) -> list[str]:
        """List all supported provider names (including unavailable)."""
        return ["openai", "gemini"]

    def is_provider_available(self, name: str, *, api_key: str | None = None) -> bool:
        """Check if a provider is available (has API key and dependencies)."""
        name = name.lower()
        if name == "openai":
            if not (api_key or self._settings.has_openai_key()):
                return False
            return _module_available("httpx")
        elif name == "gemini":
            if not (api_key or self._settings.has_gemini_key()):
                return False
            return _module_available("google.genai") and _module_available("PIL")
        return False

    def get_provider_info(self, name: str) -> dict:
        """Get information about a provider."""
        name = name.lower()

        # Prefer the cached instance to avoid creating throwaway providers
        if name in self._providers:
            caps = self._providers[name].capabilities
        elif name == "openai":
            caps = OpenAIProvider().capabilities
        elif name == "gemini":
            caps = GeminiProvider().capabilities
        else:
            raise ValueError(f"Unknown provider: {name}")

        return {
            "name": caps.name,
            "display_name": caps.display_name,
            "available": self.is_provider_available(name),
            "supported_sizes": caps.supported_sizes,
            "supported_aspect_ratios": caps.supported_aspect_ratios,
            "max_resolution": caps.max_resolution,
            "text_rendering_quality": caps.text_rendering_quality,
            "supports_reference_images": caps.supports_reference_images,
            "supports_real_time_data": caps.supports_real_time_data,
            "supports_thinking_mode": caps.supports_thinking_mode,
            "best_for": caps.best_for,
            "not_recommended_for": caps.not_recommended_for,
        }

    def get_comparison(self) -> str:
        """Get a formatted comparison of providers."""
        return self._selector.get_provider_comparison(available_providers=self.list_providers())

    async def close_all(self) -> None:
        """Close all provider instances."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

    def list_conversations(
        self, limit: int = 10, provider_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List recent conversations directly from the persistent store.

        Reads SQLite directly — no provider instantiation, API key, or network
        client construction is required just to list conversations. The store's
        own query already sorts by recency and computes message counts.

        Args:
            limit: Max conversations to return
            provider_filter: Optional provider name to filter by

        Returns:
            List of conversation summaries
        """
        from ..services.conversation_store import get_conversation_store

        store = get_conversation_store()
        return store.list_conversations(provider=provider_filter, limit=limit)


@lru_cache(maxsize=1)
def get_provider_registry() -> ProviderRegistry:
    """Get the singleton provider registry."""
    return ProviderRegistry()
