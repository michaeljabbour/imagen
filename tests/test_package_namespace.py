"""Canonical ``imagen_mcp`` package compatibility tests."""

from __future__ import annotations

import ast
import importlib
import re
import runpy
import subprocess
import sys
from pathlib import Path


def test_python_m_imagen_mcp_delegates_to_server(monkeypatch):
    import imagen_mcp.server as server

    calls: list[bool] = []
    monkeypatch.setattr(server, "main", lambda: calls.append(True))

    runpy.run_module("imagen_mcp", run_name="__main__")

    assert calls == [True]


def test_canonical_and_legacy_imports_share_module_objects() -> None:
    """Compatibility imports must not split caches or module-level singletons."""
    module_names = (
        "config.settings",
        "providers.registry",
        "services.conversation_store",
        "server",
    )

    for module_name in module_names:
        canonical = importlib.import_module(f"imagen_mcp.{module_name}")
        legacy = importlib.import_module(f"src.{module_name}")
        assert canonical is legacy

    canonical_settings = importlib.import_module("imagen_mcp.config.settings")
    legacy_settings = importlib.import_module("src.config.settings")
    assert canonical_settings.get_settings is legacy_settings.get_settings
    assert canonical_settings.get_settings() is legacy_settings.get_settings()

    canonical_registry = importlib.import_module("imagen_mcp.providers.registry")
    legacy_registry = importlib.import_module("src.providers.registry")
    assert canonical_registry.get_provider_registry is legacy_registry.get_provider_registry


def test_namespace_identity_is_import_order_independent() -> None:
    """A fresh interpreter also aliases modules first imported through ``src``."""
    script = """
import importlib
legacy = importlib.import_module('src.config.settings')
canonical = importlib.import_module('imagen_mcp.config.settings')
assert canonical is legacy
assert canonical.get_settings() is legacy.get_settings()
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_runtime_has_no_external_orchestrator_dependency() -> None:
    """The standalone server must not depend on the adjacent orchestrator stack."""
    root = Path(__file__).resolve().parents[1]
    forbidden_name = "ampli" + "fier"
    project_text = (root / "pyproject.toml").read_text()
    dependency_block = re.search(r"dependencies\s*=\s*\[(.*?)\]", project_text, re.DOTALL)
    assert dependency_block is not None
    assert forbidden_name not in dependency_block.group(1).lower()

    imported_modules: set[str] = set()
    for source_root in (root / "src", root / "imagen_mcp"):
        for python_file in source_root.rglob("*.py"):
            tree = ast.parse(python_file.read_text(), filename=str(python_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)

    assert not any(
        module.split(".", 1)[0].startswith(forbidden_name) for module in imported_modules
    )
