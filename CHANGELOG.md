# Changelog

## 0.4.0 - 2026-07-13

- Validate Gemini response bytes and normalize JPEG/WebP payloads to genuine
  PNG artifacts before persistence, keeping file suffixes and metadata honest.
- Create MCP-owned output files with mode `0600` and newly created output
  directories with mode `0700` without changing pre-existing directory modes.

### Added

- Canonical `imagen_mcp` package, module runner, and console entry points while
  preserving identity-compatible `src` imports for the 0.4 transition.
- Eight-tool MCP contract with batch generation, cost estimation, provider and
  model discovery, real conversation continuation, tool annotations, progress,
  optional bounded previews, and stable structured error envelopes.
- Gemini 3.1 Flash, Gemini 3 Pro, and Gemini 3.1 Flash Lite GA support with
  model-specific sizes, aspect ratios, reference limits, thinking controls,
  Search-grounding attribution, and current pricing estimates.
- Clean wheel/sdist installation, PEP 639 license metadata, Python 3.10–3.14 CI,
  protocol-level stdio tests, type checking, and an 80% coverage gate.

### Changed

- Updated the default image models to OpenAI `gpt-image-2` and Google
  `gemini-3.1-flash-image`; retired preview IDs now fail with migration guidance
  instead of silently changing an explicit model pin.
- Provider-specific controls now route or fail closed instead of being silently
  ignored. Prompt enhancement is explicit, OpenAI-only, and off by default.
- OpenAI supports constrained custom resolutions, strict model/quality/format
  validation, bounded retries, exclusive output writes, multi-image paths, and
  edit-backed conversational continuation.
- Conversation persistence is opt-in, atomic, private to the current user, and
  subject to a configurable 30-day inactivity retention policy.

### Security and reliability

- Added input-root enforcement for edits; format, dimension, decoded-byte, and
  aggregate limits for reference images; secret-safe logging; sanitized provider
  errors; and bounded MCP output previews.
- Google Search grounding fails closed when required Search Suggestions cannot
  be returned through a compliant output path.
- MCP and provider dependency major versions are bounded, CI actions are pinned
  to immutable commits, and clean artifacts are tested outside the source tree.

### Breaking changes

- Runtime package metadata and the server contract are now version `0.4.0`.
- Explicit unavailable providers/models and conflicting provider-specific
  controls fail instead of falling back.
- `conversational_image` rejects unknown arguments and requires a valid stored
  predecessor when continuing a conversation.
- `edit_image` reads only from the configured output root unless
  `IMAGEN_MCP_ALLOWED_INPUT_ROOTS` grants additional local roots.
