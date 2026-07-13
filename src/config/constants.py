"""
Constants for imagen-mcp providers.

This module defines provider-specific constants including supported sizes,
aspect ratios, quality tiers, and model identifiers.
"""

# ============================
# OpenAI gpt-image-2 Constants
# ============================
#
# Hand-maintained against the OpenAI REST API (/v1/images/generations and
# /v1/images/edits). This server calls the REST endpoints directly via httpx —
# there is no `openai` Python SDK dependency — so these constants mirror the
# documented request options and let validate_params() reject anything the API
# would reject.

OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# GPT Image model IDs accepted by the provider's Image API surface.
OPENAI_IMAGE_MODELS = {
    "gpt-image-2": "gpt-image-2",  # ChatGPT Images 2.0 (GA April 2026) - default
    "gpt-image-1": "gpt-image-1",  # Legacy dedicated image model (April 2025)
    "gpt-image-1.5": "gpt-image-1.5",  # Interim model (Dec 2025)
}

# Conversation orchestration models are deliberately separate: these may
# refine prompts but must never be sent to /images/generations or /images/edits.
OPENAI_ASSISTANT_MODELS = {
    "gpt-5.1": "gpt-5.1",
    "gpt-4o": "gpt-4o",
}

# Default OpenAI model for image generation (ChatGPT Images 2.0)
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"

# Representative sizes supported by gpt-image-2.  Unlike earlier GPT Image
# models, this is not an exhaustive enum: gpt-image-2 accepts any ``WxH`` size
# that satisfies the constraints below.
OPENAI_SIZES = [
    "auto",  # Let the model choose
    "1024x1024",  # Square (baseline)
    "1536x1024",  # Landscape 3:2
    "1024x1536",  # Portrait 2:3
    "1792x1024",  # Widescreen
    "1024x1792",  # Tall
    "2560x1440",  # Recommended upper reliability boundary
    "3840x2160",  # 4K / UHD, exactly at the total-pixel ceiling
]

# Provider-neutral aspect ratios that can be converted to an exact valid
# gpt-image-2 custom size. Extreme Gemini-only ratios exceed OpenAI's 3:1
# long-to-short edge limit and are intentionally excluded.
OPENAI_ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

# gpt-image-2 custom-size constraints. Edge and pixel bounds are inclusive.
OPENAI_GPT_IMAGE_2_MAX_EDGE = 3840
OPENAI_GPT_IMAGE_2_MIN_PIXELS = 655_360
OPENAI_GPT_IMAGE_2_MAX_PIXELS = 8_294_400
OPENAI_GPT_IMAGE_2_MAX_ASPECT_RATIO = 3.0

# Earlier GPT Image models retain their enumerated size behavior.  Keep the
# historical generation list for callers that explicitly pin a legacy model.
OPENAI_LEGACY_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
]

# Historical sizes accepted by the edits path for pre-gpt-image-2 models.
OPENAI_EDIT_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
]

# Quality tiers supported by GPT Image models. ``standard``/``hd`` are
# DALL-E-only and are intentionally rejected by this GPT-only provider.
OPENAI_QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
DEFAULT_OPENAI_QUALITY = "auto"

# Output image encoding formats
OPENAI_OUTPUT_FORMATS = ["png", "jpeg", "webp"]
DEFAULT_OPENAI_OUTPUT_FORMAT = "png"

# Background treatment.  gpt-image-2 does not support transparent output;
# older GPT Image models retain the legacy transparent option.
OPENAI_BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
OPENAI_GPT_IMAGE_2_BACKGROUND_OPTIONS = ["auto", "opaque"]
DEFAULT_OPENAI_BACKGROUND = "auto"

# Content moderation strictness
OPENAI_MODERATION_OPTIONS = ["auto", "low"]
DEFAULT_OPENAI_MODERATION = "auto"

# Input fidelity for /images/edits on older GPT Image models.  gpt-image-2
# always processes inputs at high fidelity and rejects/ignores this parameter,
# so the provider omits it for that model.
OPENAI_INPUT_FIDELITY_OPTIONS = ["high", "low"]
DEFAULT_OPENAI_INPUT_FIDELITY = "high"

# Max number of images per request (SDK caps; OpenAI may return fewer)
OPENAI_MAX_N = 10

# ============================
# Google Gemini Constants
# ============================
#
# Only active GA Gemini 3 image models are registered. Preview identifiers
# shut down on 2026-06-25 and are rejected rather than silently changing an
# explicit model pin.

# Endpoint identifier — Nano Banana uses chat-style generation
GEMINI_ENDPOINT_GENERATECONTENT = "generateContent"

GEMINI_MODELS: dict[str, dict[str, object]] = {
    "gemini-3.1-flash-image": {
        "marketing_name": "Nano Banana 2",
        "description": ("Nano Banana 2 (Gemini 3.1 Flash Image) — GA image generation model."),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "quality": "good",
        "max_resolution": "4K",
        "supported_sizes": ["0.5K", "1K", "2K", "4K"],
        "default_size": "1K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "max_reference_images": 14,
        "max_object_images": 10,
        "max_human_images": 4,
        "supports_google_search": True,
        "supports_thinking_mode": True,
        "supported_thinking_levels": ["minimal", "high"],
    },
    "gemini-3-pro-image": {
        "marketing_name": "Nano Banana Pro",
        "description": ("Nano Banana Pro (Gemini 3 Pro Image) — GA high-fidelity image model."),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "quality": "best",
        "max_resolution": "4K",
        "supported_sizes": ["1K", "2K", "4K"],
        "default_size": "1K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "max_reference_images": 14,
        "max_object_images": 6,
        "max_human_images": 5,
        "max_style_images": 3,
        "supports_google_search": True,
        "supports_thinking_mode": True,
    },
    "gemini-3.1-flash-lite-image": {
        "marketing_name": "Nano Banana 2 Lite",
        "description": ("Gemini 3.1 Flash Lite Image — GA cost-optimized 1K image model."),
        "endpoint": GEMINI_ENDPOINT_GENERATECONTENT,
        "quality": "efficient",
        "max_resolution": "1K",
        "supported_sizes": ["1K"],
        "default_size": "1K",
        "supports_conversational_edit": True,
        "supports_reference_images": True,
        "max_reference_images": 14,
        "max_object_images": 14,
        "max_human_images": 0,
        "supports_google_search": False,
        "supports_thinking_mode": True,
        "supported_thinking_levels": ["minimal", "high"],
        "supports_synthid": True,
        "supports_c2pa": True,
    },
}

# Friendly aliases — users can pass these human-readable names and we map
# them to canonical API identifiers.
GEMINI_MODEL_ALIASES: dict[str, str] = {
    "nano-banana-2": "gemini-3.1-flash-image",
    "nano-banana-pro": "gemini-3-pro-image",
    "nano-banana-lite": "gemini-3.1-flash-lite-image",
}

GEMINI_RETIRED_MODEL_MIGRATIONS: dict[str, str] = {
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image",
}

# Fast lookup — the active (non-deprecated) Nano Banana models
GEMINI_NANO_BANANA_MODELS: set[str] = {
    mid
    for mid, meta in GEMINI_MODELS.items()
    if meta.get("endpoint") == GEMINI_ENDPOINT_GENERATECONTENT and not meta.get("deprecated", False)
}

# Default Gemini model — Nano Banana 2, which Google uses across their
# own surfaces (Gemini app, Search, Flow).
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"

# Baseline ratios supported across the Gemini image family.
GEMINI_ASPECT_RATIOS = [
    "1:1",  # Square
    "2:3",  # Portrait (phone)
    "3:2",  # Landscape (photo)
    "3:4",  # Portrait (social)
    "4:3",  # Landscape (classic)
    "4:5",  # Portrait (Instagram)
    "5:4",  # Landscape (social)
    "9:16",  # Portrait (Stories/Reels)
    "16:9",  # Landscape (video)
    "21:9",  # Ultra-wide
]

# Gemini 3.1 Flash Image adds extreme portrait/landscape ratios.
GEMINI_FLASH_EXTENDED_ASPECT_RATIOS = [
    *GEMINI_ASPECT_RATIOS,
    "1:4",
    "4:1",
    "1:8",
    "8:1",
]

# Gemini resolution options (use uppercase K)
GEMINI_SIZES = [
    "0.5K",  # Gemini 3.1 Flash Image only
    "1K",  # Default for active Gemini 3 image models
    "2K",
    "4K",  # Maximum resolution
]

# Maximum across the supported family. Model-specific limits live in
# ``GEMINI_MODELS`` and are enforced after model resolution.
GEMINI_MAX_REFERENCE_IMAGES = 14
GEMINI_MAX_OBJECT_IMAGES = 10
GEMINI_MAX_HUMAN_IMAGES = 5

# ============================
# Shared Constants
# ============================

OPENAI_MAX_PROMPT_LENGTH = 32_000
GEMINI_MAX_PROMPT_LENGTH = 8_192
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120  # seconds

# ============================
# Provider Selection Keywords
# ============================

# Keywords that suggest OpenAI is better (text-heavy tasks)
OPENAI_PREFERRED_KEYWORDS = [
    "text",
    "label",
    "menu",
    "infographic",
    "diagram",
    "comic",
    "dialogue",
    "speech bubble",
    "caption",
    "title",
    "headline",
    "poster",
    "flyer",
    "certificate",
    "badge",
    "logo with text",
    "banner",
    "sign",
    "watermark",
    # 2.0-era strengths: UI mockups, brand-accurate rendering, world knowledge
    "ui mockup",
    "screenshot",
    "app interface",
    "webpage",
    "dashboard",
]

# Keywords that suggest Gemini is better
GEMINI_PREFERRED_KEYWORDS = [
    "portrait",
    "headshot",
    "photo",
    "photorealistic",
    "realistic",
    "product shot",
    "product photography",
    "studio lighting",
    "4k",
    "high resolution",
    "character consistency",
    "reference image",
    "weather",
    "stock",
    "current",
    "today",
    "real-time",
    "selfie",
    "person",
    "face",
    "beauty",
    "fashion",
    "magazine",
]

# Keywords that require Gemini (real-time data)
GEMINI_REQUIRED_KEYWORDS = [
    "current weather",
    "today's",
    "real-time",
    "stock price",
    "latest news",
    "live",
]
