"""
Pydantic input models for imagen-mcp tools.

These models define the parameters accepted by MCP tools
with rich descriptions for Claude to understand how to use them.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class Provider(str, Enum):
    """Available image generation providers."""

    AUTO = "auto"  # Auto-select based on prompt analysis
    OPENAI = "openai"  # OpenAI gpt-image-2 (ChatGPT Images 2.0)
    GEMINI = "gemini"  # Google Gemini 3 image family (default: 3.1 Flash Image)


class OutputFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class ImageGenerationInput(BaseModel):
    """
    Input model for unified image generation.

    This model supports both OpenAI (gpt-image-2) and Gemini providers with
    intelligent auto-selection based on prompt analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description=(
            "Text description of the desired image. Be specific about subject, "
            "composition, style, lighting, and mood. "
            "For text-heavy images (menus, infographics, UI mockups), OpenAI is "
            "auto-selected. For portraits, products, and 4K output, Gemini is "
            "auto-selected."
        ),
        min_length=1,
        max_length=32000,
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description=(
            "Image generation provider to use:\n"
            "- 'auto' (default): Automatically selects best provider based on prompt\n"
            "- 'openai': OpenAI gpt-image-2 - best for text, UI mockups, diagrams\n"
            "- 'gemini': Gemini 3.1 Flash Image - portraits, products, and up to 4K"
        ),
    )

    # Size/resolution (provider-specific)
    size: str | None = Field(
        default=None,
        description=(
            "Image size. Format depends on provider:\n"
            "- OpenAI gpt-image-2: 'auto' or WIDTHxHEIGHT. Both edges must be "
            "multiples of 16 and at most 3840px; ratio <= 3:1; total pixels "
            "655,360-8,294,400. Common values: '1024x1024', '1536x1024', "
            "'1024x1536', '2560x1440'.\n"
            "- Older OpenAI models use their enumerated legacy sizes.\n"
            "- Gemini: '1K' (default), '2K', '4K'; Gemini 3.1 Flash Image "
            "also supports '0.5K'.\n"
            "Auto-detected from prompt if not specified."
        ),
    )

    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio (Gemini only). Options: "
            "'1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', "
            "'21:9'; Gemini 3.1 Flash and Flash Lite also support "
            "'1:4', '4:1', '1:8', '8:1'. "
            "For OpenAI gpt-image-2, baseline ratios are converted to exact valid "
            "custom sizes; older GPT Image models use their nearest enumerated size."
        ),
    )

    # --- Gemini-specific features ---
    reference_images: list[str] | None = Field(
        default=None,
        description=(
            "Base64-encoded reference images for style/character consistency "
            "(Gemini only). Up to 14 total; category limits are model-specific "
            "(Flash: 10 objects/4 characters; Pro: 6 objects/5 characters/3 styles; "
            "Lite: 14 object references). "
            "If provided, Gemini provider is automatically selected."
        ),
    )

    enable_google_search: bool | None = Field(
        default=None,
        description=(
            "Enable Google Search grounding for real-time data (Gemini only). "
            "Use for current weather, stock prices, live events, etc. "
            "If enabled, Gemini provider is automatically selected. When omitted, "
            "uses the server's ENABLE_GOOGLE_SEARCH setting."
        ),
    )

    gemini_model: str | None = Field(
        default=None,
        description=(
            "Specific Google image model (Gemini only). Accepts canonical "
            "API IDs or friendly aliases:\n"
            "- 'gemini-3.1-flash-image' / alias 'nano-banana-2' (default; "
            "supports 0.5K/1K/2K/4K)\n"
            "- 'gemini-3-pro-image' / alias 'nano-banana-pro' "
            "(high fidelity, 1K/2K/4K, Thinking mode)\n"
            "- 'gemini-3.1-flash-lite-image' / alias 'nano-banana-lite' "
            "(cost-optimized 1K only; no Google Search)\n"
            "All support conversational editing and model-specific reference "
            "inputs; Flash and Pro support Google Search grounding."
        ),
    )

    thinking_level: str | None = Field(
        default=None,
        description=("Optional Gemini 3.1 Flash/Flash Lite thinking level: 'minimal' or 'high'."),
    )

    # --- OpenAI gpt-image-2 specific features ---
    openai_model: str | None = Field(
        default=None,
        description=(
            "Specific OpenAI image model to use (OpenAI only). Options:\n"
            "- 'gpt-image-2' (default): current production model with flexible sizing\n"
            "- 'gpt-image-1.5': Interim Dec 2025 model\n"
            "- 'gpt-image-1': Legacy Apr 2025 model"
        ),
    )

    quality: str | None = Field(
        default=None,
        description=(
            "Image quality tier (OpenAI only):\n"
            "- 'auto' (default): let the model choose\n"
            "- 'low' / 'medium' / 'high': explicit tier\n"
            "Higher quality costs more output tokens but produces sharper images."
        ),
    )

    openai_output_format: str | None = Field(
        default=None,
        description=(
            "Image file encoding (OpenAI only). Options: 'png' (default, lossless), "
            "'jpeg' (smaller, lossy), 'webp' (best compression)."
        ),
    )

    openai_output_compression: int | None = Field(
        default=None,
        description=(
            "Compression level 0-100 for jpeg/webp (OpenAI only). "
            "Higher = better quality, larger file. Ignored for png."
        ),
        ge=0,
        le=100,
    )

    background: str | None = Field(
        default=None,
        description=(
            "Background treatment (OpenAI only):\n"
            "- 'auto' (default): model decides\n"
            "- 'opaque': solid background\n"
            "gpt-image-2 does not support transparent output. 'transparent' is "
            "accepted only for older models that support it."
        ),
    )

    moderation: str | None = Field(
        default=None,
        description=(
            "Content moderation strictness (OpenAI only):\n"
            "- 'auto' (default): standard filtering\n"
            "- 'low': more permissive (may still refuse unsafe content)"
        ),
    )

    n: int | None = Field(
        default=None,
        description=(
            "Number of images to generate in a single request (OpenAI only, 1-10). "
            "Default is 1. Gemini always returns 1 per call."
        ),
        ge=1,
        le=10,
    )

    # --- Common options ---
    enhance_prompt: bool | None = Field(
        default=None,
        description=(
            "Whether to enhance the prompt before generating. For OpenAI this "
            "adds a Chat Completions prompt-refinement call (using gpt-5.1 by default) "
            "before image generation for richer context — at the cost "
            "of extra latency. When omitted, uses the server's "
            "ENABLE_PROMPT_ENHANCEMENT setting (false by default). Enabling it adds a "
            "separate assistant-model API call, latency, and cost. Has no effect on Gemini."
        ),
    )

    output_path: str | None = Field(
        default=None,
        description=(
            "Optional path to save the generated image. "
            "If a directory, saves with generated filename. "
            "If a file path, saves to that exact path. "
            "Supports `~` and environment variables; defaults to `OUTPUT_DIR/{provider}` or "
            "`~/Downloads/images/{provider}`."
        ),
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response (markdown or json).",
    )

    include_preview: bool | None = Field(
        default=False,
        description=(
            "Opt in to a bounded MCP ImageContent thumbnail (max 512px/200KB). "
            "Leave false for clients that serialize non-text content as JSON."
        ),
    )

    # --- API keys (optional overrides — hidden from repr/serialization) ---
    openai_api_key: SkipJsonSchema[str | None] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override (uses OPENAI_API_KEY env var if not provided).",
    )

    gemini_api_key: SkipJsonSchema[str | None] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Gemini API key override (uses GEMINI_API_KEY env var if not provided).",
    )


class ConversationalImageInput(BaseModel):
    """
    Input model for conversational image generation with multi-turn refinement.

    Supports iterative refinement where each prompt builds on previous results.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description=(
            "Text description for new image or refinement instruction. "
            "For refinements, use natural language like 'make it darker' or 'add more detail'."
        ),
        min_length=1,
        max_length=32000,
    )

    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation ID from previous generation to continue refining. "
            "Omit to start a new conversation. Auto-generated if not provided."
        ),
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description=(
            "Provider to use. Note: Cannot switch providers mid-conversation. "
            "The provider from the first message in a conversation is used throughout."
        ),
    )

    # Dialogue system options
    dialogue_mode: str | None = Field(
        default="guided",
        description=(
            "Dialogue depth for pre-generation refinement:\n"
            "- 'quick': 1-2 questions, fast path\n"
            "- 'guided': 3-5 questions, balanced (default)\n"
            "- 'explorer': Deep exploration with 6+ questions\n"
            "- 'skip': Direct generation, no dialogue"
        ),
    )

    skip_dialogue: bool | None = Field(
        default=False,
        description="Set to true to skip dialogue and generate immediately.",
    )

    # Size/resolution
    size: str | None = Field(
        default=None,
        description="Image size (provider-specific format). Auto-detected if not specified.",
    )

    aspect_ratio: str | None = Field(
        default=None,
        description="Aspect ratio; converted to an exact valid size for OpenAI gpt-image-2.",
    )

    # Reference images (Gemini only)
    reference_images: list[str] | None = Field(
        default=None,
        description="Base64-encoded reference images (Gemini only, up to 14 model-dependent).",
    )

    enable_google_search: bool | None = Field(
        default=None,
        description=(
            "Enable Google Search grounding (Gemini only). When omitted, uses "
            "the server's ENABLE_GOOGLE_SEARCH setting."
        ),
    )

    enhance_prompt: bool | None = Field(
        default=None,
        description=(
            "Enable OpenAI-only assistant prompt enhancement. When omitted, uses the "
            "server's ENABLE_PROMPT_ENHANCEMENT setting (false by default)."
        ),
    )

    # Gemini-specific
    gemini_model: str | None = Field(
        default=None,
        description=(
            "Specific Gemini model (Gemini only):\n"
            "- 'gemini-3.1-flash-image': Nano Banana 2 (default)\n"
            "- 'gemini-3-pro-image': Nano Banana Pro\n"
            "- 'gemini-3.1-flash-lite-image': Nano Banana Lite (1K only, no Search)"
        ),
    )

    thinking_level: str | None = Field(
        default=None,
        description="Gemini 3.1 Flash/Flash Lite thinking level: 'minimal' or 'high'.",
    )

    # OpenAI-specific
    openai_model: str | None = Field(
        default=None,
        description=("Specific OpenAI image model (OpenAI only). Default 'gpt-image-2'."),
    )

    assistant_model: str | None = Field(
        default=None,
        description="GPT model for understanding refinement instructions (OpenAI only).",
    )

    quality: str | None = Field(
        default=None,
        description="Image quality tier: 'auto' / 'low' / 'medium' / 'high' (OpenAI only).",
    )

    background: str | None = Field(
        default=None,
        description=(
            "Background: 'auto' / 'opaque' for gpt-image-2. 'transparent' is "
            "legacy-model-only (OpenAI only)."
        ),
    )

    # Output options
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional path to save the generated image. "
            "If a directory, saves with generated filename. "
            "If a file path, saves to that exact path. "
            "Supports `~` and environment variables; defaults to `OUTPUT_DIR/{provider}` or "
            "`~/Downloads/images/{provider}`."
        ),
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    include_preview: bool | None = Field(
        default=False,
        description="Opt in to a bounded MCP ImageContent thumbnail.",
    )

    # API keys (hidden from repr/serialization)
    openai_api_key: SkipJsonSchema[str | None] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override.",
    )

    gemini_api_key: SkipJsonSchema[str | None] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="Gemini API key override.",
    )


class EditImageInput(BaseModel):
    """
    Input model for image editing with gpt-image-2.

    Uses the /images/edits endpoint. gpt-image-2 always processes inputs at
    high fidelity, so the legacy input_fidelity parameter is omitted for that
    model. Supports inpainting via optional mask.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description=(
            "Instruction describing the edit. "
            "Examples: 'add red roses to the ad frames', "
            "'change the sky to sunset', 'remove the person in the background'."
        ),
        min_length=1,
        max_length=32000,
    )

    image_path: str = Field(
        ...,
        description=(
            "Absolute or ~-expanded path to the source image on disk (png / jpeg / webp)."
        ),
    )

    mask_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a PNG mask. Transparent pixels indicate regions "
            "the model should edit; opaque pixels remain unchanged."
        ),
    )

    size: str | None = Field(
        default=None,
        description=(
            "Output size. gpt-image-2 accepts 'auto' or constrained WIDTHxHEIGHT "
            "(multiples of 16, each edge at most 3840px, ratio <= 3:1, total pixels "
            "655,360-8,294,400). Older models retain enumerated legacy sizes. "
            "Defaults to 'auto'."
        ),
    )

    quality: str | None = Field(
        default=None,
        description="Quality tier: 'auto' / 'low' / 'medium' / 'high'.",
    )

    background: str | None = Field(
        default=None,
        description=(
            "Background: 'auto' / 'opaque' for gpt-image-2. 'transparent' is "
            "available only on older models that support it."
        ),
    )

    openai_output_format: str | None = Field(
        default=None,
        description="Output encoding: 'png' / 'jpeg' / 'webp'.",
    )

    openai_output_compression: int | None = Field(
        default=None,
        description="Compression 0-100 for jpeg/webp.",
        ge=0,
        le=100,
    )

    input_fidelity: str | None = Field(
        default=None,
        description=(
            "Legacy-model input fidelity: 'high' (default) or 'low'. gpt-image-2 "
            "is always high fidelity, so this value is accepted for compatibility "
            "but omitted from gpt-image-2 API requests."
        ),
    )

    n: int | None = Field(
        default=None,
        description="Number of edited variants to generate (1-10).",
        ge=1,
        le=10,
    )

    openai_model: str | None = Field(
        default=None,
        description="OpenAI image model to use. Default 'gpt-image-2'.",
    )

    output_path: str | None = Field(
        default=None,
        description="Optional path to save the edited image.",
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    include_preview: bool | None = Field(
        default=False,
        description="Opt in to a bounded MCP ImageContent thumbnail.",
    )

    openai_api_key: SkipJsonSchema[str | None] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="OpenAI API key override.",
    )


class CostEstimateInput(BaseModel):
    """Input model for the estimate_cost tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(
        ...,
        description="Prompt to estimate cost for (used for auto provider selection).",
        min_length=1,
        max_length=32000,
    )

    provider: Provider | None = Field(
        default=Provider.AUTO,
        description="Provider to estimate for: 'auto', 'openai', or 'gemini'.",
    )

    model: str | None = Field(
        default=None,
        description=(
            "Optional provider model ID for model-specific published pricing, e.g. "
            "'gpt-image-2', 'gemini-3.1-flash-image', or 'gemini-3-pro-image'."
        ),
    )

    quality: str | None = Field(
        default=None,
        description="Quality tier (OpenAI): 'auto' / 'low' / 'medium' / 'high'.",
    )

    size: str | None = Field(
        default=None,
        description="Image size (provider-specific, e.g. '1024x1024' or '2K').",
    )

    n: int = Field(
        default=1,
        description="Number of images to estimate for (1-10).",
        ge=1,
        le=10,
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )


class ImageRefinementElicitation(BaseModel):
    """Schema for MCP Elicitation during conversational refinement.

    All fields are optional so the client can render a lightweight form;
    whatever the user fills in is appended to the prompt.
    """

    style: str | None = Field(
        default=None,
        description="Visual style, e.g. photorealistic, illustration, oil-painting.",
    )
    mood: str | None = Field(
        default=None,
        description="Desired mood, e.g. warm, dramatic, serene.",
    )
    additional_details: str | None = Field(
        default=None,
        description="Any other details to incorporate into the image.",
    )


class BatchItem(BaseModel):
    """A single image request within a batch."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(..., description="Image prompt.", min_length=1, max_length=32000)
    provider: Provider | None = Field(
        default=None,
        description="Override provider for this item ('auto'/'openai'/'gemini'). "
        "Falls back to the batch default_provider when omitted.",
    )
    size: str | None = Field(default=None, description="Image size (provider-specific).")
    aspect_ratio: str | None = Field(
        default=None,
        description="Aspect ratio; converted to an exact valid size for OpenAI gpt-image-2.",
    )
    n: int | None = Field(
        default=None, description="Images for this item (OpenAI, 1-10).", ge=1, le=10
    )
    quality: str | None = Field(default=None, description="Quality tier (OpenAI).")
    gemini_model: str | None = Field(default=None, description="Specific Gemini model.")
    thinking_level: str | None = Field(
        default=None,
        description="Gemini 3.1 Flash/Flash Lite thinking level: minimal/high.",
    )
    openai_model: str | None = Field(default=None, description="Specific OpenAI model.")
    reference_images: list[str] | None = Field(
        default=None, description="Base64 reference images (Gemini)."
    )
    enable_google_search: bool | None = Field(
        default=None,
        description=("Google Search grounding (Gemini). When omitted, uses the server default."),
    )
    enhance_prompt: bool | None = Field(
        default=None,
        description="OpenAI-only prompt enhancement; omitted uses the false server default.",
    )
    output_path: str | None = Field(default=None, description="Optional save path for this item.")


class BatchGenerationInput(BaseModel):
    """Input model for concurrent multi-prompt image generation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    items: list[BatchItem] = Field(
        ...,
        description="Image requests to generate concurrently.",
        min_length=1,
        max_length=50,
    )

    max_concurrency: int = Field(
        default=4,
        description="Maximum number of items generated at once (1-16).",
        ge=1,
        le=16,
    )

    default_provider: Provider | None = Field(
        default=Provider.AUTO,
        description="Provider used for items that don't specify their own.",
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )

    include_preview: bool | None = Field(
        default=False,
        description="Opt in to one bounded MCP ImageContent thumbnail for the batch.",
    )

    openai_api_key: SkipJsonSchema[str | None] = Field(
        default=None, repr=False, exclude=True, description="OpenAI API key override."
    )
    gemini_api_key: SkipJsonSchema[str | None] = Field(
        default=None, repr=False, exclude=True, description="Gemini API key override."
    )


class ListConversationsInput(BaseModel):
    """Input model for listing saved conversations."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    limit: int | None = Field(
        default=10,
        description="Maximum number of conversations to return.",
        ge=1,
        le=100,
    )

    provider: str | None = Field(
        default=None,
        description="Filter by provider ('openai' or 'gemini').",
    )

    output_format: OutputFormat | None = Field(
        default=OutputFormat.MARKDOWN,
        description="Output format for the tool response.",
    )
