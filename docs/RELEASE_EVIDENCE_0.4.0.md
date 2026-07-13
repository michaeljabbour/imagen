# imagen-mcp 0.4.0 release evidence

Evidence date: 2026-07-13

This record intentionally omits API keys and full prompt bodies. Request IDs,
artifact hashes, dimensions, and bounded visual observations are retained so
the release can be audited without preserving sensitive inputs.

## Deterministic gates

- Ruff formatting and linting passed across `imagen_mcp`, `src`, and `tests`.
- Mypy passed across both public package namespaces.
- The lockfile passed `uv lock --check`.
- 321 tests passed on Python 3.12 with 84.35% branch-aware coverage, above the
  enforced 80% floor.
- Wheel and source distribution builds both clean-installed outside the source
  checkout, passed dependency checks, preserved package/import identity, and
  completed a real MCP stdio initialize plus tool-discovery smoke test.
- The runtime dependency graph contains no Amplifier package, bundle, import,
  or sibling-checkout dependency.

## Live provider canaries

Both canaries ran through the real MCP stdio boundary with server identity
`imagen_mcp` and version `0.4.0`. Each successful canary made exactly one paid
provider request after initialization.

### OpenAI

- Model/options: `gpt-image-2`, low quality, 1024x1024, JPEG.
- Provider request ID: `c20d05083e7c`.
- Result: HTTP 200; 51,745 bytes; SHA-256
  `20b337331be57e0a90c572eb0bbe11bec522d33885a875cffc574e7a9e286abe`.
- Usage: 29 input, 196 output, 225 total tokens; local estimate about $0.006.
- Visual QA: the decoded image showed the requested blue ceramic mug on an
  off-white background, with no text or people.

### Google Gemini

- An initial unsupported 0.5K/minimal combination failed safely with HTTP 400
  and was not retried.
- Corrected model/options: `gemini-3.1-flash-image`, 1K, 1:1, with Search,
  thinking, enhancement, persistence, and preview disabled.
- Provider request ID: `684fa1218f60`.
- Result: HTTP 200; 1024x1024 RGB; 617,225 bytes; SHA-256
  `82d8b501d779ab4329963e4eeaf4413c0c65e75315076ff3b73909796e491f2f`;
  local estimate about $0.067 because the response exposed no usage cost.
- Visual QA: the decoded image matched the bounded safe canary request.

The Gemini response contained valid JPEG bytes despite the requested output
path ending in `.png`. The same run also showed that newly written artifacts
inherited mode `0644`. Release blockers were fixed before tagging: Gemini now
validates actual response bytes and normalizes supported JPEG/WebP payloads to
genuine PNG before persistence, while MCP-owned files and newly created output
directories use modes `0600` and `0700`. Regression tests exercise a real JPEG
payload, reject corrupt response bytes before writing, verify PNG decoding, and
assert private modes. No second paid request was made solely to retest these
deterministic local transformations.

## Scope

The canaries establish credentialed provider compatibility, MCP transport,
successful decoding, and human visual inspection for one bounded request per
provider. They are not quality benchmarks, load tests, or guarantees of future
provider availability.
