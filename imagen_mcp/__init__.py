"""Canonical package namespace for imagen-mcp."""

from src import __version__

from ._compat import install_import_aliases

install_import_aliases()

__all__ = ["__version__"]
