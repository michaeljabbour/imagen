"""Smart-tool surface for imagen-mcp: model-agnostic image generation.

This package is an additional adapter over the existing ``src.providers``
library (per the Amplifier Smart Tools spec: "the library is the tool").
It does not reimplement generation -- it resolves *which* model to call via
:mod:`src.smart_tool.model_resolution`, then delegates to the same
``OpenAIProvider`` / ``GeminiProvider`` classes the MCP server already uses.
"""

from __future__ import annotations

__all__: list[str] = []
