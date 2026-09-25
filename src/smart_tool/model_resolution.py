"""Model-agnostic resolution for the imagen-mcp smart tool.

Resolves *which* provider model id to call, in this order:

1. **Explicit caller choice** -- ``requested`` that is not ``None``/``"latest"``
   and is not a configured alias name is returned unchanged. The tool never
   rejects an unrecognized model string here; it is the provider's own API
   that validates it (see ``allow_unknown_model=True`` on the provider calls
   in :mod:`src.smart_tool.lib`).
2. **User-editable alias/config file** -- ``openai: latest`` / ``gemini: latest``
   pin a family to runtime discovery, or a literal model id pins that model
   directly. Named entries under ``aliases:`` map a short name (``fast``,
   ``quality``, ...) to a concrete model id.
3. **Runtime discovery** -- lists each provider's models API, filters to the
   image-capable family, and picks the newest. Results are cached (in-memory
   for the process, and on disk as a last-known-good fallback) so a discovery
   failure degrades to the most recent successful answer, and only then to
   the provider's own hardcoded default.

Config file: ``~/.config/imagen-mcp/models.yaml`` (override with the
``IMAGEN_MCP_MODELS_CONFIG`` env var), parsed with ``yaml.safe_load``:

    openai: latest
    gemini: latest
    aliases:
      fast: gemini-3.1-flash-lite-image
      quality: gpt-image-2

Discovery cache: ``~/.cache/imagen-mcp/model_discovery_cache.json`` (override
with ``IMAGEN_MCP_DISCOVERY_CACHE``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL_SECONDS = 3600
_DISCOVERY_TIMEOUT_SECONDS = 10.0

# Hardcoded last-resort fallbacks -- only used when there is no config, no
# usable cache, and discovery itself failed. These intentionally mirror
# ``src.config.constants`` defaults but are duplicated here (not imported)
# so the smart tool's fallback path never depends on the MCP server's
# static registries: a currently-shipped model is a safe floor even if the
# registry those constants live in is stale or absent.
_HARDCODED_FALLBACK = {
    "openai": "gpt-image-2",
    "gemini": "gemini-3.1-flash-image",
}


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    override = os.environ.get("IMAGEN_MCP_MODELS_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "imagen-mcp" / "models.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the user-editable model config, or ``{}`` if none exists/invalid."""
    path = Path(config_path).expanduser() if config_path else default_config_path()
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read model config %s: %s", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


# ---------------------------------------------------------------------------
# Disk cache (last-known-good, survives process restarts)
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    override = os.environ.get("IMAGEN_MCP_DISCOVERY_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "imagen-mcp" / "model_discovery_cache.json"


def _read_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_cache(data: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write discovery cache %s: %s", path, exc)


def _cache_get(cache_key: str, ttl_seconds: int) -> str | None:
    cache = _read_cache()
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    model = entry.get("model")
    ts = entry.get("ts")
    if not isinstance(model, str) or not isinstance(ts, (int, float)):
        return None
    if ttl_seconds >= 0 and time.time() - ts > ttl_seconds:
        # Stale for "fresh discovery" purposes, but still a valid
        # last-known-good fallback if a live discovery attempt fails.
        return None
    return model


def _cache_get_stale(cache_key: str) -> str | None:
    """Return a cached model regardless of age -- the fallback-of-last-resort."""
    cache = _read_cache()
    entry = cache.get(cache_key)
    if isinstance(entry, dict):
        model = entry.get("model")
        if isinstance(model, str):
            return model
    return None


def _cache_put(cache_key: str, model: str) -> None:
    cache = _read_cache()
    cache[cache_key] = {"model": model, "ts": time.time()}
    _write_cache(cache)


# ---------------------------------------------------------------------------
# OpenAI discovery
# ---------------------------------------------------------------------------

# Matches the primary rolling release line only: "gpt-image-N" / "gpt-image-N.M".
# Deliberately excludes dated snapshot pins ("gpt-image-2-2026-04-21") and
# codenamed previews ("gpt-image-2.5-flare") observed on the live API --
# those are opt-in via an explicit ``model=`` or a config-file pin, not
# something "latest" discovery should silently start returning.
_OPENAI_IMAGE_MODEL_PATTERN = re.compile(r"^gpt-image-(\d+)(?:\.(\d+))?$")


async def discover_openai_models(
    api_key: str, *, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """List OpenAI models. Raises on any transport/HTTP/parse failure."""
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT_SECONDS)
    try:
        response = await http_client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("OpenAI /v1/models returned an unexpected payload shape")
        return data
    finally:
        if owns_client:
            await http_client.aclose()


def newest_openai_image_model(models: list[dict[str, Any]]) -> str | None:
    """Pick the newest ``gpt-image-*`` id, ranked by version then ``created``."""
    best: tuple[tuple[int, int, int], str] | None = None
    for entry in models:
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        match = _OPENAI_IMAGE_MODEL_PATTERN.match(model_id)
        if not match:
            continue
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        created = int(entry.get("created") or 0)
        rank = (major, minor, created)
        if best is None or rank > best[0]:
            best = (rank, model_id)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Gemini discovery
# ---------------------------------------------------------------------------

# Ordered longest-prefix-first so "flash-lite" is not misclassified as "flash".
_GEMINI_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("flash-lite", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash-lite-image$")),
    ("flash", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash-image$")),
    ("pro", re.compile(r"^gemini-(\d+)(?:\.(\d+))?-pro-image$")),
]


async def discover_gemini_models(
    api_key: str, *, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """List Gemini models. Raises on any transport/HTTP/parse failure."""
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT_SECONDS)
    try:
        response = await http_client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("Gemini models.list returned an unexpected payload shape")
        return models
    finally:
        if owns_client:
            await http_client.aclose()


def newest_gemini_image_model(models: list[dict[str, Any]], family: str = "flash") -> str | None:
    """Pick the newest image-capable Gemini model id in ``family``.

    ``family`` is one of ``flash`` (default -- matches the current shipped
    default), ``flash-lite``, or ``pro``.
    """
    pattern = next((p for name, p in _GEMINI_FAMILY_PATTERNS if name == family), None)
    if pattern is None:
        raise ValueError(f"Unknown Gemini model family '{family}'")

    best: tuple[tuple[int, int], str] | None = None
    for entry in models:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        model_id = name.removeprefix("models/")
        match = pattern.match(model_id)
        if not match:
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if methods and "generateContent" not in methods:
            continue
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        rank = (major, minor)
        if best is None or rank > best[0]:
            best = (rank, model_id)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Public resolution entry point
# ---------------------------------------------------------------------------


async def resolve_model(
    provider: str,
    requested: str | None = None,
    *,
    config_path: Path | str | None = None,
    api_key: str | None = None,
    cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    family: str = "flash",
) -> str:
    """Resolve a concrete model id for ``provider`` ("openai" or "gemini").

    Never raises for an unrecognized ``requested`` value -- explicit caller
    choice always passes through unchanged (step (a) in the module
    docstring). Discovery/config failures degrade gracefully rather than
    raising: cache -> hardcoded fallback.
    """
    provider = provider.lower()
    if provider not in ("openai", "gemini"):
        raise ValueError(f"Unknown provider '{provider}'; expected 'openai' or 'gemini'")

    config = load_config(config_path)
    raw_aliases = config.get("aliases")
    aliases: dict[str, Any] = raw_aliases if isinstance(raw_aliases, dict) else {}

    # (a) Explicit caller choice, unless it names a configured alias.
    if requested and requested != "latest":
        if requested in aliases:
            return str(aliases[requested])
        return requested

    # (b) Config file entry for this provider.
    configured = config.get(provider)
    if isinstance(configured, str) and configured != "latest":
        return configured

    # (c) Runtime discovery, newest per family, with cache + fallback.
    cache_key = provider if provider == "openai" else f"gemini:{family}"
    if api_key:
        try:
            if provider == "openai":
                models = await discover_openai_models(api_key)
                discovered = newest_openai_image_model(models)
            else:
                models = await discover_gemini_models(api_key)
                discovered = newest_gemini_image_model(models, family=family)
            if discovered:
                _cache_put(cache_key, discovered)
                return discovered
            logger.warning("%s model discovery returned no matching models", provider)
        except Exception as exc:  # noqa: BLE001 - discovery must never raise outward
            logger.warning("%s model discovery failed: %s", provider, exc)

    cached = _cache_get(cache_key, cache_ttl_seconds) or _cache_get_stale(cache_key)
    if cached:
        return cached

    return _HARDCODED_FALLBACK[provider]
