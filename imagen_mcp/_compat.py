"""Import compatibility for the historical :mod:`src` package.

``imagen_mcp`` is the public package name, while the implementation remains in
``src`` during the 0.4 compatibility window.  A normal namespace-path shim
would execute the same source file twice under two module names, splitting
module globals, caches, and other singletons.  This finder instead makes every
``imagen_mcp.<name>`` import resolve to the already canonical ``src.<name>``
module object.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

_PUBLIC_PREFIX = "imagen_mcp."
_IMPLEMENTATION_PREFIX = "src."
_LOCAL_MODULES = frozenset({"_compat", "__main__"})


class _ImplementationAliasLoader(importlib.abc.Loader):
    """Return one implementation module under its public import alias."""

    def __init__(self, implementation_name: str) -> None:
        self.implementation_name = implementation_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        """Reuse the implementation module instead of executing it twice."""
        return importlib.import_module(self.implementation_name)

    def exec_module(self, module: ModuleType) -> None:
        """The implementation module was fully executed by ``create_module``."""


class _ImplementationAliasFinder(importlib.abc.MetaPathFinder):
    """Map public submodule imports to their historical implementation names."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if not fullname.startswith(_PUBLIC_PREFIX):
            return None

        relative_name = fullname.removeprefix(_PUBLIC_PREFIX)
        if relative_name.split(".", 1)[0] in _LOCAL_MODULES:
            return None

        implementation_name = f"{_IMPLEMENTATION_PREFIX}{relative_name}"
        try:
            implementation_spec = importlib.util.find_spec(implementation_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return None
        if implementation_spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _ImplementationAliasLoader(implementation_name),
            is_package=implementation_spec.submodule_search_locations is not None,
        )


def install_import_aliases() -> None:
    """Install the public-to-implementation finder exactly once."""
    if not any(isinstance(finder, _ImplementationAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ImplementationAliasFinder())
