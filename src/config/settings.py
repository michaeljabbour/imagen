"""
Settings management for imagen-mcp.

Handles API keys and configuration from environment variables.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # API Keys
    openai_api_key: str | None = None
    gemini_api_key: str | None = None  # Also checks GOOGLE_API_KEY

    # Defaults
    default_provider: str = "auto"  # auto, openai, gemini
    default_openai_size: str = "1024x1024"
    default_gemini_size: str = "1K"
    default_gemini_aspect_ratio: str = "1:1"

    # Feature flags
    enable_prompt_enhancement: bool = False
    enable_google_search: bool = False  # Gemini-only feature

    # Timeouts (seconds). Generous read ceiling for slow high-quality image
    # renders; the OpenAI client pairs this with a short 10s connect timeout.
    request_timeout: int = 600

    # Client-side rate limiting (per provider). Tuned for parallelism: a small
    # min-interval and a real burst allowance so concurrent requests aren't
    # serialized. Raise the intervals if you hit provider 429s.
    openai_rpm: int = 10
    openai_min_interval_seconds: float = 0.5
    openai_burst_limit: int = 5
    gemini_rpm: int = 15
    gemini_min_interval_seconds: float = 0.5
    gemini_burst_limit: int = 5

    # Output
    output_dir: str | None = None
    allowed_input_roots: tuple[str, ...] = ()
    conversation_retention_days: int = 30

    # Logging
    log_dir: str | None = None
    log_level: str = "INFO"
    log_max_bytes: int = 5_242_880  # 5 MiB
    log_backup_count: int = 3
    log_prompts: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY for Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        log_dir = os.getenv("IMAGEN_MCP_LOG_DIR") or os.getenv("LOG_DIR")
        log_level = os.getenv("IMAGEN_MCP_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
        log_max_bytes = (
            os.getenv("IMAGEN_MCP_LOG_MAX_BYTES") or os.getenv("LOG_MAX_BYTES") or "5242880"
        )
        log_backup_count = (
            os.getenv("IMAGEN_MCP_LOG_BACKUP_COUNT") or os.getenv("LOG_BACKUP_COUNT") or "3"
        )
        log_prompts = os.getenv("IMAGEN_MCP_LOG_PROMPTS") or os.getenv("LOG_PROMPTS") or "false"

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=gemini_key,
            default_provider=os.getenv("DEFAULT_PROVIDER", "auto"),
            default_openai_size=os.getenv("DEFAULT_OPENAI_SIZE", "1024x1024"),
            default_gemini_size=os.getenv("DEFAULT_GEMINI_SIZE", "1K"),
            default_gemini_aspect_ratio=os.getenv("DEFAULT_GEMINI_ASPECT_RATIO", "1:1"),
            enable_prompt_enhancement=(
                os.getenv("ENABLE_PROMPT_ENHANCEMENT", "false").lower() == "true"
            ),
            enable_google_search=os.getenv("ENABLE_GOOGLE_SEARCH", "false").lower() == "true",
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "600")),
            openai_rpm=int(os.getenv("OPENAI_RPM", "10")),
            openai_min_interval_seconds=float(os.getenv("OPENAI_MIN_INTERVAL_SECONDS", "0.5")),
            openai_burst_limit=int(os.getenv("OPENAI_BURST_LIMIT", "5")),
            gemini_rpm=int(os.getenv("GEMINI_RPM", "15")),
            gemini_min_interval_seconds=float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "0.5")),
            gemini_burst_limit=int(os.getenv("GEMINI_BURST_LIMIT", "5")),
            output_dir=os.getenv("OUTPUT_DIR"),
            allowed_input_roots=tuple(
                root
                for root in os.getenv("IMAGEN_MCP_ALLOWED_INPUT_ROOTS", "").split(os.pathsep)
                if root.strip()
            ),
            conversation_retention_days=int(
                os.getenv("IMAGEN_MCP_CONVERSATION_RETENTION_DAYS", "30")
            ),
            log_dir=log_dir,
            log_level=log_level,
            log_max_bytes=int(log_max_bytes),
            log_backup_count=int(log_backup_count),
            log_prompts=log_prompts.lower() == "true",
        )

    def get_openai_api_key(self, provided_key: str | None = None) -> str:
        """Get OpenAI API key from provided value or settings."""
        api_key = provided_key or self.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        return api_key

    def get_gemini_api_key(self, provided_key: str | None = None) -> str:
        """Get Gemini API key from provided value or settings."""
        api_key = provided_key or self.gemini_api_key
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        return api_key

    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key)

    def has_gemini_key(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(self.gemini_api_key)

    def available_providers(self) -> list[str]:
        """Return list of providers with valid API keys."""
        providers = []
        if self.has_openai_key():
            providers.append("openai")
        if self.has_gemini_key():
            providers.append("gemini")
        return providers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
