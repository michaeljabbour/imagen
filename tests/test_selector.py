"""Tests for provider selection logic."""

import os

import pytest

# Set dummy API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.providers.selector import ProviderSelector


@pytest.fixture
def selector():
    """Create a provider selector instance."""
    return ProviderSelector()


class TestProviderSelection:
    """Tests for auto provider selection."""

    def test_text_rendering_selects_openai(self, selector):
        """Text-heavy prompts should select OpenAI."""
        prompts = [
            "Create a menu card for an Italian restaurant with prices",
            "Design an infographic about climate change",
            "Make a poster with headline text",
            "Create a comic strip with dialogue bubbles",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "openai", f"Expected OpenAI for: {prompt}"

    def test_photorealistic_selects_gemini(self, selector):
        """Photorealistic prompts should select Gemini."""
        prompts = [
            "Professional headshot with studio lighting",
            "Product photography of a perfume bottle",
            "Photorealistic portrait of a woman",
            "4K landscape photo of mountains",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "gemini", f"Expected Gemini for: {prompt}"

    def test_available_providers_limits_selection(self, selector):
        """Selection should respect the provided availability list."""
        rec = selector.suggest_provider(
            "Professional headshot with studio lighting",
            available_providers=["openai"],
        )
        assert rec.provider == "openai"

    def test_4k_shorthand_can_route_to_openai(self, selector):
        """4K is no longer a hard Gemini requirement for gpt-image-2."""
        rec = selector.suggest_provider(
            "A poster with headline text",
            size="4K",
            available_providers=["openai", "gemini"],
        )
        assert rec.provider == "openai"

    def test_reference_images_require_gemini(self, selector):
        """Reference images should force Gemini."""
        rec = selector.suggest_provider("A simple cat", reference_images=["base64encodedimage"])
        assert rec.provider == "gemini"
        assert rec.confidence == 1.0

    def test_google_search_requires_gemini(self, selector):
        """Google Search grounding should force Gemini."""
        rec = selector.suggest_provider("Show me the weather", enable_google_search=True)
        assert rec.provider == "gemini"
        assert rec.confidence == 1.0

    def test_explicit_provider_override(self, selector):
        """Explicit provider should override auto-selection."""
        # Force OpenAI even for portrait
        rec = selector.suggest_provider("Professional headshot", explicit_provider="openai")
        assert rec.provider == "openai"
        assert rec.confidence == 1.0

    def test_realtime_keywords_require_gemini(self, selector):
        """Real-time data keywords should force Gemini."""
        prompts = [
            "Show the current weather in NYC",
            "What's today's stock price for AAPL",
            "Real-time traffic map",
        ]
        for prompt in prompts:
            rec = selector.suggest_provider(prompt)
            assert rec.provider == "gemini", f"Expected Gemini for: {prompt}"
            assert rec.confidence == 1.0


class TestProviderRecommendation:
    """Tests for recommendation metadata."""

    def test_recommendation_includes_reasoning(self, selector):
        """Recommendations should include reasoning."""
        rec = selector.suggest_provider("Create a menu with prices")
        assert rec.reasoning is not None
        assert len(rec.reasoning) > 0

    def test_recommendation_includes_confidence(self, selector):
        """Recommendations should include confidence score."""
        rec = selector.suggest_provider("A beautiful sunset")
        assert 0.0 <= rec.confidence <= 1.0

    def test_recommendation_detects_image_type(self, selector):
        """Recommendations should detect image type when applicable."""
        rec = selector.suggest_provider("Professional portrait photo")
        assert rec.detected_image_type == "portrait"

    def test_recommendation_includes_alternative(self, selector):
        """Recommendations should include alternative provider."""
        rec = selector.suggest_provider("A simple landscape")
        assert rec.alternative in ["openai", "gemini", None]


class TestFallbackBehavior:
    """Tests for fallback notices when preferred provider is unavailable."""

    def test_soft_fallback_sets_notice_for_photorealistic(self, selector):
        """Photorealistic prompt with only OpenAI should produce a fallback notice."""
        rec = selector.suggest_provider(
            "Professional headshot with studio lighting",
            available_providers=["openai"],
        )
        assert rec.provider == "openai"
        assert rec.is_fallback is True
        assert rec.fallback_notice is not None
        assert "GEMINI_API_KEY" in rec.fallback_notice
        assert rec.preferred_provider == "gemini"

    def test_soft_fallback_sets_notice_for_text_rendering(self, selector):
        """Text-rendering prompt with only Gemini should produce a fallback notice."""
        rec = selector.suggest_provider(
            "Create a restaurant menu with prices and text labels",
            available_providers=["gemini"],
        )
        assert rec.provider == "gemini"
        assert rec.is_fallback is True
        assert rec.fallback_notice is not None
        assert "OPENAI_API_KEY" in rec.fallback_notice
        assert rec.preferred_provider == "openai"

    def test_hard_requirement_reference_images_fails_closed(self, selector):
        """Reference inputs must never be silently dropped on another provider."""
        with pytest.raises(ValueError, match="Hard feature requirements never fall back"):
            selector.suggest_provider(
                "A cat in the style of my reference",
                reference_images=["base64data"],
                available_providers=["openai"],
            )

    def test_4k_is_not_reported_as_missing_on_openai(self, selector):
        """OpenAI maps the shorthand to a valid near-4K custom size."""
        rec = selector.suggest_provider(
            "A poster with headline text",
            size="4K",
            available_providers=["openai"],
        )
        assert rec.provider == "openai"
        assert not rec.missing_features

    def test_hard_requirement_google_search_fails_closed(self, selector):
        """Grounding requests must never run ungrounded on another provider."""
        with pytest.raises(ValueError, match="Hard feature requirements never fall back"):
            selector.suggest_provider(
                "Current weather in NYC",
                enable_google_search=True,
                available_providers=["openai"],
            )

    def test_explicit_provider_pin_fails_closed(self, selector):
        """An explicit vendor choice must never silently cross providers."""
        with pytest.raises(ValueError, match="explicit provider pins never fall back"):
            selector.suggest_provider(
                "A sunset",
                explicit_provider="gemini",
                available_providers=["openai"],
            )

    def test_no_fallback_when_preferred_is_available(self, selector):
        """When both providers are available, no fallback notice should appear."""
        rec = selector.suggest_provider(
            "Professional headshot with studio lighting",
            available_providers=["openai", "gemini"],
        )
        assert rec.provider == "gemini"
        assert rec.is_fallback is False
        assert rec.fallback_notice is None

    def test_no_fallback_for_ambiguous_prompts(self, selector):
        """Ambiguous prompts (similar scores) shouldn't generate fallback notices."""
        rec = selector.suggest_provider(
            "A beautiful sunset",
            available_providers=["openai"],
        )
        assert rec.provider == "openai"
        # For ambiguous prompts, the score diff is small, so no notice
        if rec.is_fallback:
            assert rec.fallback_notice is None  # score_diff <= 0.1 suppresses notice

    def test_fallback_reduces_confidence(self, selector):
        """Fallback should reduce the confidence score."""
        rec_full = selector.suggest_provider(
            "Professional headshot with studio lighting",
            available_providers=["openai", "gemini"],
        )
        rec_fallback = selector.suggest_provider(
            "Professional headshot with studio lighting",
            available_providers=["openai"],
        )
        assert rec_fallback.confidence < rec_full.confidence
