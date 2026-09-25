"""Smart-tool library capabilities for imagen.

This is the actual capability surface (per the smart tools spec, "the
library is the tool"). Every function here is a plain async function
returning a structured, JSON-serializable dict -- no generation logic is
reimplemented; each delegates to the existing ``src.providers`` classes the
MCP server already uses, after resolving *which model* to call via
:mod:`src.smart_tool.model_resolution`.

Deterministic capabilities (no model provider required): ``manifest``,
``list_providers``, ``estimate_cost``.
Model-backed capabilities (require a configured provider credential):
``generate_image``, ``edit_image``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.config.pricing import estimate_generation_cost
from src.providers.openai_provider import OpenAIProvider
from src.providers.registry import get_provider_registry
from src.providers.selector import ProviderSelector

from . import model_resolution

# The repo-root SMART_TOOL.md is canonical (the spec's distribution root);
# an identical copy ships inside the package so an installed tool (uv tool
# install / pip install, where the repo root does not exist) can still render
# its manifest. tests/test_smart_tool_manifest_packaging.py guards drift.
_PACKAGED_MD_PATH = Path(__file__).resolve().parent / "SMART_TOOL.md"
_REPO_ROOT_MD_PATH = Path(__file__).resolve().parent.parent.parent / "SMART_TOOL.md"
_SMART_TOOL_MD_PATH = _PACKAGED_MD_PATH if _PACKAGED_MD_PATH.is_file() else _REPO_ROOT_MD_PATH
# Intentionally NOT renamed: this is the macOS Keychain service prefix under
# which users already have API keys stored (dev.imagen-mcp.OPENAI_API_KEY /
# dev.imagen-mcp.GEMINI_API_KEY). Changing it would orphan existing stored
# keys, so it stays pinned to the pre-rename name even though the project
# and distribution are now called "imagen".
_KEYCHAIN_SERVICE_PREFIX = "dev.imagen-mcp"


class SmartToolError(RuntimeError):
    """A capability failed in a way the caller can act on.

    The message always names what went wrong and, where applicable, what to
    configure -- per the spec's invocation/failure semantics.
    """


def _keychain_api_key(env_var: str) -> str | None:
    """Fall back to the same macOS Keychain lookup ``run.sh`` uses.

    Never prints or logs the retrieved value. Returns ``None`` on any
    failure (missing ``security`` binary, no matching entry, non-macOS).
    """
    account = os.environ.get("USER")
    if not account:
        return None
    service = f"{_KEYCHAIN_SERVICE_PREFIX}.{env_var}"
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _resolve_api_key(provider: str) -> str | None:
    """Environment first, then the macOS Keychain fallback."""
    env_var = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    value = os.environ.get(env_var) or (
        os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None
    )
    if value:
        return value
    return _keychain_api_key(env_var)


def _available_providers() -> list[str]:
    available = []
    for provider in ("openai", "gemini"):
        if _resolve_api_key(provider):
            available.append(provider)
    return available


async def _resolved_model(
    provider: str, requested_model: str | None, *, family: str = "flash"
) -> str:
    api_key = _resolve_api_key(provider)
    return await model_resolution.resolve_model(
        provider, requested_model, api_key=api_key, family=family
    )


# ---------------------------------------------------------------------------
# Deterministic capabilities
# ---------------------------------------------------------------------------


def manifest() -> dict[str, Any]:
    """Return the tool's own manifest (frontmatter fields + body). Deterministic."""
    text = _SMART_TOOL_MD_PATH.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise SmartToolError(f"SMART_TOOL.md at {_SMART_TOOL_MD_PATH} is missing frontmatter")
    frontmatter_text, body = match.group(1), match.group(2)
    frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(frontmatter, dict):
        raise SmartToolError(f"SMART_TOOL.md at {_SMART_TOOL_MD_PATH} has invalid frontmatter")
    return {"frontmatter": frontmatter, "body": body.strip()}


def list_providers() -> dict[str, Any]:
    """List providers, availability, and capabilities. Deterministic."""
    registry = get_provider_registry()
    providers = {}
    for name in registry.list_all_providers():
        info = registry.get_provider_info(name)
        info["available"] = name in _available_providers()
        providers[name] = info
    return {"providers": providers, "available": _available_providers()}


def estimate_cost(
    *,
    provider: str | None = None,
    prompt: str = "",
    model: str | None = None,
    quality: str | None = None,
    size: str | None = None,
    n: int = 1,
) -> dict[str, Any]:
    """Estimate generation cost for a provider/model/size/quality combination.

    Deterministic (published sample pricing, not a live billing call) --
    no provider credential is required, even when ``provider`` is omitted.
    When omitted and at least one provider is configured, the same
    prompt-based selector the MCP server uses picks one; otherwise this
    defaults to "openai" so the capability still works with zero
    credentials configured.
    """
    if provider:
        resolved_provider = provider
    else:
        available = _available_providers()
        resolved_provider = (
            ProviderSelector().suggest_provider(prompt, available_providers=available).provider
            if available
            else "openai"
        )
    estimate = estimate_generation_cost(
        resolved_provider, model=model, quality=quality, size=size, n=n
    )
    return {
        "provider": estimate.provider,
        "model": estimate.model,
        "quality": estimate.quality,
        "size": estimate.size,
        "n": estimate.n,
        "per_image_usd": estimate.per_image_usd,
        "total_usd": estimate.total_usd,
        "approximate": estimate.approximate,
        "note": estimate.note,
    }


# ---------------------------------------------------------------------------
# Model-backed capabilities
# ---------------------------------------------------------------------------


async def generate_image(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    size: str | None = None,
    aspect_ratio: str | None = None,
    output_path: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    reference_images: list[str] | None = None,
    enable_google_search: bool = False,
    n: int | None = None,
    family: str = "flash",
) -> dict[str, Any]:
    """Generate an image. Model-backed -- requires a configured provider.

    ``provider`` picks the backend explicitly ("openai" or "gemini"); when
    omitted, the same prompt-based selector the MCP server uses picks one
    from the configured providers. ``model`` follows the resolution order
    documented in :mod:`src.smart_tool.model_resolution`: explicit choice,
    then the user's config file, then live discovery of the newest model,
    then a cached or hardcoded fallback.
    """
    available = _available_providers()
    if not available:
        raise SmartToolError(
            "No image-generation provider is configured. Set OPENAI_API_KEY and/or "
            "GEMINI_API_KEY (environment variable, or the macOS Keychain fallback "
            "run.sh uses)."
        )

    if provider:
        provider = provider.lower()
        if provider not in available:
            raise SmartToolError(
                f"Requested provider '{provider}' is not available. Set {provider.upper()}_API_KEY."
            )
        resolved_provider = provider
    else:
        recommendation = ProviderSelector().suggest_provider(
            prompt,
            size=size,
            reference_images=reference_images,
            enable_google_search=enable_google_search,
            available_providers=available,
        )
        resolved_provider = recommendation.provider

    resolved_model = await _resolved_model(resolved_provider, model, family=family)
    registry = get_provider_registry()
    # Pass the resolved key explicitly: it may have come from the Keychain
    # fallback rather than the environment, which the registry's cached
    # Settings snapshot cannot see.
    provider_instance = registry.get_provider(
        resolved_provider, api_key=_resolve_api_key(resolved_provider)
    )

    kwargs: dict[str, Any] = {
        "size": size,
        "aspect_ratio": aspect_ratio,
        "output_path": output_path,
        "reference_images": reference_images,
        "allow_unknown_model": True,
    }
    if resolved_provider == "openai":
        kwargs["openai_model"] = resolved_model
        kwargs["quality"] = quality
        kwargs["background"] = background
        kwargs["n"] = n
    else:
        kwargs["model"] = resolved_model
        kwargs["enable_google_search"] = enable_google_search

    result = await provider_instance.generate_image(prompt, **kwargs)
    if not result.success:
        raise SmartToolError(result.error or "Image generation failed.")
    return result.to_dict()


async def edit_image(
    prompt: str,
    image_path: str,
    *,
    model: str | None = None,
    mask_path: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Edit an existing image via OpenAI's /images/edits. Model-backed.

    Editing (image-to-image with pixel preservation and inpainting) is an
    OpenAI gpt-image capability; there is no Gemini equivalent in this tool.
    """
    if not _resolve_api_key("openai"):
        raise SmartToolError(
            "OpenAI is not configured. Set OPENAI_API_KEY to use edit_image "
            "(environment variable, or the macOS Keychain fallback run.sh uses)."
        )
    resolved_model = await _resolved_model("openai", model)
    provider = OpenAIProvider(api_key=_resolve_api_key("openai"))
    try:
        result = await provider.edit_image(
            prompt=prompt,
            image_path=image_path,
            mask_path=mask_path,
            size=size,
            quality=quality,
            background=background,
            openai_model=resolved_model,
            output_path=output_path,
            allow_unknown_model=True,
        )
    finally:
        await provider.close()
    if not result.success:
        raise SmartToolError(result.error or "Image edit failed.")
    return result.to_dict()


__all__ = [
    "SmartToolError",
    "edit_image",
    "estimate_cost",
    "generate_image",
    "list_providers",
    "manifest",
]
