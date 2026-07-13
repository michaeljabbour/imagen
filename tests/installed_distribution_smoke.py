"""Smoke an installed imagen-mcp artifact outside the source checkout."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
from datetime import timedelta
from importlib.metadata import distribution
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def assert_distribution_metadata() -> None:
    """Confirm release metadata survived artifact construction."""
    installed = distribution("imagen-mcp")
    assert installed.version == "0.4.0"
    assert installed.metadata["License-Expression"] == "MIT"
    assert any(str(file).endswith("licenses/LICENSE") for file in (installed.files or ()))


def assert_namespace_identity() -> None:
    """Confirm compatibility imports share modules in the installed artifact."""
    for relative_name in (
        "config.settings",
        "providers.registry",
        "services.conversation_store",
        "server",
    ):
        canonical = importlib.import_module(f"imagen_mcp.{relative_name}")
        legacy = importlib.import_module(f"src.{relative_name}")
        assert canonical is legacy

    settings = importlib.import_module("imagen_mcp.config.settings")
    legacy_settings = importlib.import_module("src.config.settings")
    assert settings.get_settings() is legacy_settings.get_settings()


async def assert_stdio_server() -> None:
    """Initialize the packaged module runner through a real MCP stdio client."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(
            {
                "OPENAI_API_KEY": "",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "IMAGEN_MCP_LOG_LEVEL": "CRITICAL",
                "IMAGEN_MCP_TRANSPORT": "stdio",
                "OUTPUT_DIR": temp_dir,
            }
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "imagen_mcp"],
            cwd=Path(temp_dir),
            env=env,
        )
        with open(os.devnull, "w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=15),
                ) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()

        assert initialized.serverInfo.name == "imagen_mcp"
        assert initialized.serverInfo.version == "0.4.0"
        assert "generate_image" in {tool.name for tool in tools.tools}


def main() -> None:
    """Run all clean-install assertions."""
    assert_distribution_metadata()
    assert_namespace_identity()
    asyncio.run(assert_stdio_server())


if __name__ == "__main__":
    main()
