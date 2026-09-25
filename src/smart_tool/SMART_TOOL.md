---
smart_tool_format: 1
name: imagen
version: 0.5.0
description: >
  Model-agnostic multi-provider image generation and editing (OpenAI and
  Google Gemini). Use when an agent needs to generate, edit, or estimate the
  cost of an image and should keep working as providers ship newer image
  models, without code or config changes.
use_cases:
  - Generate an image from a text prompt, letting the tool pick OpenAI or
    Gemini based on the prompt (or pin one explicitly)
  - Edit an existing image (inpainting, image-to-image) via OpenAI gpt-image
  - Estimate the cost of a generation before running it
  - List available providers, models, and their capabilities
platforms:
  - macos
  - linux
requires:
  - name: security
    purpose: >
      macOS Keychain fallback for OPENAI_API_KEY/GEMINI_API_KEY when neither
      is set in the environment. Without it, only environment-variable
      credentials work.
    optional: true
    install: README.md#credentials
---

# imagen (smart tool)

Generates and edits images through OpenAI's gpt-image family and Google's
Gemini image family, picking a model automatically so it keeps working when
either provider ships a new one -- no code edits, no version pins to update.

**The library is the tool.** `src.smart_tool.lib` holds every capability; the
CLI (`imagen`) is a thin wrapper that parses arguments, calls the
library, and prints JSON. Every deterministic capability
(`manifest`, `list-providers`, `estimate-cost`) runs with no provider
credential configured. The model-backed capabilities (`generate-image`,
`edit-image`) need `OPENAI_API_KEY` and/or `GEMINI_API_KEY` -- set as an
environment variable, or (macOS only) in the Keychain under
`dev.imagen-mcp.OPENAI_API_KEY` / `dev.imagen-mcp.GEMINI_API_KEY` for the
current `$USER`, the same fallback this repository's own `run.sh` uses.

## When to reach for it

- You need an image generated or edited and don't want to hardcode a model
  id that will go stale when the provider ships a newer one.
- You want a single interface across OpenAI and Gemini rather than two
  separate SDKs.

## When not to

- You need the full MCP tool surface (conversational multi-turn refinement,
  batch generation) -- run the `imagen-mcp` MCP server instead; this smart
  tool covers generate / edit / list / estimate only.
- You need guaranteed reproducible model pinning across runs without any
  possibility of picking up a newer model -- pass an explicit `model=` (or
  set the config file's provider entry to a literal model id, not `latest`).

## Model resolution (the model-agnostic part)

For each provider, in order:

1. **Explicit choice** -- the `model=` argument, when given, is used as-is.
   An unrecognized id is *not* rejected by this tool; the provider's own API
   validates it.
2. **Config file** -- `~/.config/imagen/models.yaml`
   (override with `IMAGEN_MCP_MODELS_CONFIG`; falls back to the pre-rename
   `~/.config/imagen-mcp/models.yaml` for reads if the new path doesn't
   exist):

   ```yaml
   openai: latest
   gemini: latest
   aliases:
     fast: gemini-3.1-flash-lite-image
     quality: gpt-image-2
   ```

   `openai`/`gemini` set to `latest` trigger discovery (step 3); set to a
   literal model id, that id is used directly. Named `aliases` map a short
   name you pass as `model=` to a concrete id.
3. **Live discovery** -- lists the provider's own models API
   (`GET /v1/models` for OpenAI, `models.list` for Gemini), filters to the
   image-capable family, and picks the newest by parsed version (and, for
   OpenAI, `created` timestamp). The result is cached
   (`~/.cache/imagen/model_discovery_cache.json`, also falling back to the
   pre-rename `~/.cache/imagen-mcp/` cache for reads) so a later failure
   falls back to the last successful discovery, then to a hardcoded default
   (`gpt-image-2` / `gemini-3.1-flash-image`).

Discovery requires the relevant API key; without one, resolution falls
straight to the cache or hardcoded default. Discovery only considers the
primary rolling release line (`gpt-image-N[.M]`, `gemini-N[.M]-<family>-image`)
-- dated snapshot pins and codenamed previews are excluded from "latest" and
must be requested by an explicit `model=`.

## Worked invocations

```bash
# Deterministic -- no credentials needed
imagen manifest
imagen list-providers
imagen estimate-cost --provider openai --size 1024x1024 --quality medium

# Model-backed -- needs OPENAI_API_KEY and/or GEMINI_API_KEY
imagen generate-image "a cozy coffee shop, morning light" --size 1024x1024
imagen generate-image "professional headshot" --provider gemini
imagen edit-image "add a rainbow" ./photo.png --output-path ./edited.png
```

Each capability's own `--help` documents its arguments, result shape, and
failures in full (`imagen generate-image --help`).

## Sharp edges

- `edit-image` is OpenAI-only; there is no Gemini equivalent in this tool.
- Cost estimates are published sample prices, not a live billing call, and
  return `note` (no numeric estimate) for combinations without a documented
  price -- this is not an error, it's the tool refusing to extrapolate.
- A smart-path capability with no provider configured fails immediately,
  naming exactly which environment variable to set.

## Where to read more

- `README.md` in this repository -- setup, credentials, the underlying MCP
  server this smart tool sits alongside.
- `src/providers/` -- the OpenAI and Gemini provider implementations this
  tool delegates to; nothing here reimplements generation.
