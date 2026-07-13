"""
Provider selection logic for imagen-mcp.

Analyzes prompts and requirements to recommend the best provider
for each image generation task.
"""

import logging
import re
from dataclasses import dataclass

from ..config.constants import (
    GEMINI_PREFERRED_KEYWORDS,
    GEMINI_REQUIRED_KEYWORDS,
    OPENAI_PREFERRED_KEYWORDS,
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Pre-compiled image-type patterns (module-level to avoid re-compilation)
_IMAGE_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("portrait", re.compile(r"\b(portrait|headshot|face|person|selfie)\b")),
    ("product", re.compile(r"\b(product|item|merchandise|goods|e-commerce)\b")),
    ("landscape", re.compile(r"\b(landscape|scenery|vista|panorama)\b")),
    ("diagram", re.compile(r"\b(diagram|flowchart|infographic|chart|graph)\b")),
    ("logo", re.compile(r"\b(logo|icon|symbol|emblem)\b")),
    ("comic", re.compile(r"\b(comic|cartoon|panel|manga|anime)\b")),
    ("menu", re.compile(r"\b(menu|card|list|catalog)\b")),
    ("poster", re.compile(r"\b(poster|flyer|banner|advertisement)\b")),
    ("photo", re.compile(r"\b(photo|photograph|picture|image of)\b")),
    ("art", re.compile(r"\b(painting|artwork|illustration|drawing)\b")),
]


@dataclass
class ProviderRecommendation:
    """Recommendation for which provider to use."""

    provider: str  # "openai" or "gemini"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    alternative: str | None = None
    alternative_reasoning: str | None = None

    # Feature requirements that drove the decision
    requires_text_rendering: bool = False
    requires_reference_images: bool = False
    requires_real_time_data: bool = False
    requires_high_resolution: bool = False
    detected_image_type: str | None = None

    # Fallback information (set when preferred provider was unavailable)
    is_fallback: bool = False
    fallback_notice: str | None = None
    preferred_provider: str | None = None
    missing_features: list[str] | None = None


class ProviderSelector:
    """
    Intelligent provider selection based on prompt analysis.

    Analyzes prompts to determine which provider (OpenAI or Gemini)
    is best suited for the task based on:
    - Text rendering requirements
    - Image type (portrait, product, diagram, etc.)
    - Resolution requirements
    - Real-time data needs
    - Reference image requirements
    """

    def __init__(self) -> None:
        """Initialize the provider selector."""
        self.settings = get_settings()

    def suggest_provider(
        self,
        prompt: str,
        *,
        size: str | None = None,
        reference_images: list[str] | None = None,
        enable_google_search: bool = False,
        explicit_provider: str | None = None,
        available_providers: list[str] | None = None,
    ) -> ProviderRecommendation:
        """
        Analyze prompt and requirements to suggest the best provider.

        Args:
            prompt: The image generation prompt
            size: Requested size (may indicate resolution needs)
            reference_images: List of reference images (requires Gemini)
            enable_google_search: Whether real-time data is needed
            explicit_provider: User-specified provider override

        Returns:
            ProviderRecommendation with provider choice and reasoning
        """
        prompt_lower = prompt.lower()

        # Check available providers
        available = (
            available_providers
            if available_providers is not None
            else self.settings.available_providers()
        )
        if not available:
            raise ValueError("No providers available. Set OPENAI_API_KEY or GEMINI_API_KEY.")

        # Handle explicit provider request
        if explicit_provider:
            if explicit_provider in available:
                return ProviderRecommendation(
                    provider=explicit_provider,
                    confidence=1.0,
                    reasoning=f"User explicitly requested {explicit_provider} provider.",
                )
            raise ValueError(
                f"Requested provider '{explicit_provider}' is not available. "
                f"Set {explicit_provider.upper()}_API_KEY; explicit provider pins never fall back."
            )

        # Check for hard requirements that force a specific provider
        forced_provider, force_reason = self._check_hard_requirements(
            prompt_lower, reference_images, enable_google_search, size
        )
        if forced_provider:
            if forced_provider in available:
                return ProviderRecommendation(
                    provider=forced_provider,
                    confidence=1.0,
                    reasoning=force_reason,
                    requires_reference_images=bool(reference_images),
                    requires_real_time_data=enable_google_search,
                    requires_high_resolution=self._needs_high_resolution(size),
                )
            raise ValueError(
                f"This request requires {forced_provider} ({force_reason}), but that provider "
                "is not available. Hard feature requirements never fall back."
            )

        # Score each provider based on prompt analysis
        openai_score, openai_reasons = self._score_for_openai(prompt_lower)
        gemini_score, gemini_reasons = self._score_for_gemini(prompt_lower)

        # Detect image type
        image_type = self._detect_image_type(prompt_lower)

        # Adjust scores based on detected type
        if image_type in ["portrait", "headshot", "product", "photo", "selfie"]:
            gemini_score += 0.2
            gemini_reasons.append(f"Detected {image_type} image type (Gemini excels)")
        elif image_type in ["diagram", "infographic", "comic", "menu", "poster"]:
            openai_score += 0.2
            openai_reasons.append(f"Detected {image_type} image type (OpenAI excels)")

        # Choose provider
        if openai_score > gemini_score:
            primary, alt = "openai", "gemini"
            primary_reasons, alt_reasons = openai_reasons, gemini_reasons
        else:
            primary, alt = "gemini", "openai"
            primary_reasons, alt_reasons = gemini_reasons, openai_reasons

        # Ensure primary is available — track if we had to swap
        swapped = False
        original_primary = primary
        if primary not in available:
            primary, alt = alt, primary
            primary_reasons, alt_reasons = alt_reasons, primary_reasons
            swapped = True

        # Calculate confidence based on score difference
        score_diff = abs(openai_score - gemini_score)
        confidence = min(0.95, 0.5 + score_diff)
        if swapped:
            confidence = max(0.3, confidence - 0.2)

        # Build fallback info if we had to swap
        fallback_notice = None
        if swapped and score_diff > 0.1:
            fallback_notice = (
                f"**{original_primary.title()}** would be better for this prompt"
                f" ({'; '.join(alt_reasons) if alt_reasons else 'higher score'}), "
                f"but it's not configured. Using **{primary}** instead. "
                f"Set `{original_primary.upper()}_API_KEY` for better results."
            )

        return ProviderRecommendation(
            provider=primary,
            confidence=confidence,
            reasoning="; ".join(primary_reasons) if primary_reasons else "Default selection",
            alternative=alt if alt in available else None,
            alternative_reasoning="; ".join(alt_reasons) if alt_reasons else None,
            requires_text_rendering=self._needs_text_rendering(prompt_lower),
            detected_image_type=image_type,
            is_fallback=swapped,
            fallback_notice=fallback_notice,
            preferred_provider=original_primary if swapped else None,
        )

    def _check_hard_requirements(
        self,
        prompt_lower: str,
        reference_images: list[str] | None,
        enable_google_search: bool,
        size: str | None,
    ) -> tuple[str | None, str]:
        """Check for requirements that force a specific provider."""
        # Reference images require Gemini (for /images/generations-style
        # multi-reference consistency). OpenAI's edit_image tool supports
        # single-image editing but not the ref-set pattern Gemini exposes.
        if reference_images and len(reference_images) > 0:
            return "gemini", "Reference images require Gemini provider"

        # Google Search grounding requires Gemini
        if enable_google_search:
            return "gemini", "Google Search grounding requires Gemini provider"

        # Check for real-time data keywords
        for keyword in GEMINI_REQUIRED_KEYWORDS:
            if keyword in prompt_lower:
                return "gemini", f"Real-time data keyword '{keyword}' requires Gemini"

        return None, ""

    def _describe_missing_features(
        self,
        preferred: str,
        fallback: str,
        reference_images: list[str] | None,
        enable_google_search: bool,
        size: str | None,
    ) -> list[str]:
        """Describe features that will be missing due to provider fallback."""
        missing = []
        if preferred == "gemini" and fallback == "openai":
            if reference_images:
                missing.append(
                    "multi-reference image sets "
                    "(OpenAI supports single-image editing via edit_image only)"
                )
            if enable_google_search:
                missing.append("Google Search grounding (OpenAI does not support this)")
        elif preferred == "openai" and fallback == "gemini":
            missing.append("OpenAI's stronger text-rendering behavior")
        return missing

    def _score_for_openai(self, prompt_lower: str) -> tuple[float, list[str]]:
        """Score how well OpenAI fits this prompt."""
        score = 0.5  # Base score
        reasons = []

        # Check for text rendering keywords
        text_keywords = [
            "text",
            "label",
            "title",
            "headline",
            "caption",
            "menu",
            "sign",
            "banner",
            "poster",
            "certificate",
        ]
        text_count = sum(1 for k in text_keywords if k in prompt_lower)
        if text_count > 0:
            score += 0.15 * min(text_count, 3)
            reasons.append(f"Text rendering needed ({text_count} keywords)")

        # Check for OpenAI-preferred keywords
        openai_matches = sum(1 for k in OPENAI_PREFERRED_KEYWORDS if k in prompt_lower)
        if openai_matches > 0:
            score += 0.1 * min(openai_matches, 3)
            reasons.append(f"OpenAI-preferred content ({openai_matches} matches)")

        # Check for diagram/infographic indicators
        if any(k in prompt_lower for k in ["diagram", "infographic", "flowchart", "chart"]):
            score += 0.2
            reasons.append("Diagram/infographic content")

        # Check for comic/sequential art
        if any(k in prompt_lower for k in ["comic", "panel", "speech bubble", "dialogue"]):
            score += 0.2
            reasons.append("Comic/sequential art content")

        return score, reasons

    def _score_for_gemini(self, prompt_lower: str) -> tuple[float, list[str]]:
        """Score how well Gemini fits this prompt."""
        score = 0.5  # Base score
        reasons = []

        # Check for portrait/person keywords
        person_keywords = ["portrait", "headshot", "person", "face", "selfie", "photo of"]
        person_count = sum(1 for k in person_keywords if k in prompt_lower)
        if person_count > 0:
            score += 0.15 * min(person_count, 3)
            reasons.append(f"Portrait/person content ({person_count} keywords)")

        # Check for Gemini-preferred keywords
        gemini_matches = sum(1 for k in GEMINI_PREFERRED_KEYWORDS if k in prompt_lower)
        if gemini_matches > 0:
            score += 0.1 * min(gemini_matches, 3)
            reasons.append(f"Gemini-preferred content ({gemini_matches} matches)")

        # Check for photorealistic indicators
        if any(k in prompt_lower for k in ["photorealistic", "realistic", "photograph", "studio"]):
            score += 0.2
            reasons.append("Photorealistic content")

        # Check for product photography
        if any(k in prompt_lower for k in ["product", "e-commerce", "catalog", "commercial"]):
            score += 0.15
            reasons.append("Product photography content")

        # Check for high resolution mentions
        if any(k in prompt_lower for k in ["4k", "high resolution", "high quality", "detailed"]):
            score += 0.1
            reasons.append("High resolution requested")

        return score, reasons

    def _detect_image_type(self, prompt_lower: str) -> str | None:
        """Detect the type of image being requested."""
        for image_type, pattern in _IMAGE_TYPE_PATTERNS:
            if pattern.search(prompt_lower):
                return image_type

        return None

    def _needs_text_rendering(self, prompt_lower: str) -> bool:
        """Check if prompt requires text rendering."""
        text_indicators = [
            "text",
            "label",
            "caption",
            "title",
            "headline",
            "sign",
            "banner",
            "poster",
            "menu",
            "certificate",
            "quote",
            "saying",
            "words",
            "write",
            "spell",
        ]
        return any(indicator in prompt_lower for indicator in text_indicators)

    def _needs_high_resolution(self, size: str | None) -> bool:
        """Check if high resolution is needed."""
        if not size:
            return False
        return size.upper() in ["4K", "2K", "1536x1024", "1024x1536"]

    def get_provider_comparison(self, *, available_providers: list[str] | None = None) -> str:
        """Get a formatted comparison of available providers."""
        available = (
            available_providers
            if available_providers is not None
            else self.settings.available_providers()
        )

        lines = [
            "## Provider Comparison",
            "",
            "| Feature | OpenAI gpt-image-2 | Gemini 3.1 Flash Image (default) |",
            "|---------|---------------------|------------------------|",
            f"| Available | {'✅' if 'openai' in available else '❌'} | "
            f"{'✅' if 'gemini' in available else '❌'} |",
            "| Text Rendering | ⭐⭐⭐ Excellent | ⭐⭐ Good |",
            "| Photorealism | ⭐⭐⭐ Near-photographic | ⭐⭐⭐ Excellent |",
            "| Latency | Varies by size and quality | Varies by model and size |",
            "| Max Resolution | 3840px edge / 8.29MP (>2K experimental) | 4K |",
            "| Reference Images | Single-image via edit_image | ✅ Multi-ref (up to 14) |",
            "| Real-time Data | ❌ | ✅ (Google Search) |",
            "| Sequential Editing | ✅ (preserve-pixel edits) | ⚠️ Limited |",
            "| Token Usage Tracking | ✅ | ❌ |",
            "",
            "### When to Use Each:",
            "",
            "**OpenAI gpt-image-2 (ChatGPT Images 2.0):**",
            "- Text-heavy images (menus, infographics, posters)",
            "- UI mockups and screenshot-style renders",
            "- Comics with dialogue and speech bubbles",
            "- Technical diagrams with precise labels",
            "- Multi-step edits that preserve unchanged pixels",
            "",
            "**Gemini 3.1 Flash Image (default):**",
            "- Photorealistic portraits and headshots",
            "- Product photography",
            "- Native 4K resolution output",
            "- Character consistency across multiple reference images",
            "- Real-time grounded visualization (weather, stocks)",
        ]

        return "\n".join(lines)
