"""
Path utilities for saving generated images.

Centralizes output directory behavior across providers:
- Expands `~` for user-supplied paths
- Supports `OUTPUT_DIR` as the default save directory when `output_path` is omitted
- Creates parent directories as needed
- Prevents path traversal attacks by confining output to the allowed base directory
"""

from __future__ import annotations

import os
from pathlib import Path

from .settings import get_settings


def _mkdir_private(path: Path) -> None:
    """Create a server-owned directory as owner-only without changing existing paths."""

    existed = path.exists()
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not existed:
        path.chmod(0o700)


def expand_path(path: str) -> Path:
    """Expand `~` in a path string.

    Only tilde expansion is performed.  Environment-variable expansion
    (``os.path.expandvars``) is deliberately omitted to avoid leaking
    server-side environment variables through user-supplied paths.
    """
    return Path(path).expanduser()


def get_base_output_directory() -> Path:
    """Get the base directory for saving images (provider subdirs are created under this)."""
    settings = get_settings()
    if settings.output_dir:
        output_dir = expand_path(settings.output_dir)
    else:
        output_dir = Path.home() / "Downloads" / "images"

    _mkdir_private(output_dir)
    return output_dir


def get_provider_output_directory(provider: str) -> Path:
    """Get the default provider-specific directory for saving images."""
    provider_dir = get_base_output_directory() / provider
    _mkdir_private(provider_dir)
    return provider_dir


def get_log_directory() -> Path:
    """Get the directory for server logs."""
    settings = get_settings()
    if settings.log_dir:
        log_dir = expand_path(settings.log_dir)
    else:
        log_dir = get_base_output_directory() / "logs"

    _mkdir_private(log_dir)
    return log_dir


def resolve_output_path(
    output_path: str | None, *, default_filename: str, provider: str | None = None
) -> Path:
    """
    Resolve an `output_path` (directory or file path) into a concrete file path.

    Rules:
    - If `output_path` is omitted, saves under `~/Downloads/images/{provider}` by default
      (or under `OUTPUT_DIR/{provider}` when `OUTPUT_DIR` is set)
    - If `output_path` ends with a path separator, treat it as a directory
    - If `output_path` exists and is a directory, treat it as a directory
    - If `output_path` has no suffix, treat it as a directory
    - Otherwise treat it as a file path
    """
    if not default_filename:
        raise ValueError("default_filename must not be empty")

    is_directory = False

    if output_path and output_path.strip():
        raw = output_path.strip()
        path_obj = expand_path(raw)

        is_directory = (
            raw.endswith(("/", "\\"))
            or (path_obj.exists() and path_obj.is_dir())
            or not path_obj.suffix
        )
        if is_directory:
            save_path = path_obj / default_filename
        else:
            save_path = path_obj
    else:
        if provider:
            base_dir = get_provider_output_directory(provider)
        else:
            base_dir = get_base_output_directory()
        save_path = base_dir / default_filename

    # --- Path traversal prevention ---
    # Ensure the resolved save path stays within the intended directory,
    # blocking attempts like "../../etc/passwd".
    resolved = save_path.resolve()

    if output_path and output_path.strip():
        # User provided an explicit output path — validate against
        # the user's intended directory, not the global default.
        user_base = expand_path(output_path.strip()).resolve()
        if is_directory:
            # User specified a directory — save_path must be within it
            if not str(resolved).startswith(str(user_base) + os.sep):
                raise ValueError(f"Output path must be within {user_base}")
        # File-path case: resolved == user_base by construction (no check needed)
    else:
        # No custom path — validate against the default output base
        allowed_base = get_base_output_directory().resolve()
        if not str(resolved).startswith(str(allowed_base) + os.sep) and resolved != allowed_base:
            raise ValueError(f"Output path must be within {allowed_base}")

    _mkdir_private(save_path.parent)
    return save_path
