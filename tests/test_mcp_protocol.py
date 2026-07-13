"""Protocol-level MCP regressions using a real stdio ClientSession."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_tool_failures_set_mcp_is_error(tmp_path):
    """Provider/configuration failures must not look like successful tool calls."""
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "IMAGEN_MCP_TRANSPORT": "stdio",
            "IMAGEN_MCP_LOG_LEVEL": "CRITICAL",
            "OUTPUT_DIR": str(tmp_path),
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "imagen_mcp"],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=10),
            ) as session:
                initialization = await session.initialize()
                assert initialization.serverInfo.name == "imagen_mcp"
                assert initialization.serverInfo.version == "0.4.0"

                generation = await session.call_tool(
                    "generate_image",
                    {"params": {"prompt": "A test image", "provider": "openai"}},
                )
                model_listing = await session.call_tool("list_gemini_models", {})

    assert generation.isError is True
    assert model_listing.isError is True
    generation_text = generation.content[0].text  # type: ignore[union-attr]
    envelope_start = generation_text.index('{"error"')
    envelope = json.loads(generation_text[envelope_start:])
    assert envelope["error"]["code"] == "configuration_error"
    assert "No providers available" in envelope["error"]["message"]
    assert envelope["error"]["request_id"]
    assert envelope["error"]["retryable"] is False
