# Repository Guidelines

For contributors building and maintaining the MCP image generation server.

## Project Structure & Module Organization
- `imagen_mcp` is the canonical package and `python -m imagen_mcp` is the MCP entry point. `src` is an identity-preserving compatibility namespace.
- `src/providers/` holds provider implementations (`openai_provider.py`, `gemini_provider.py`), `selector.py` for auto-selection, and `registry.py` for factory wiring.
- `src/config/` contains constants and settings; `src/models/input_models.py` defines Pydantic request models.
- Tests live in `tests/` mirroring modules (`test_selector.py`, `test_providers.py`, `test_server.py`); `run.sh` is the wrapper used by MCP clients; dependencies are declared in `pyproject.toml`.

## Build, Test, and Development Commands
- Install locked development dependencies: `uv sync --extra dev`.
- Run the server locally: `uv run python -m imagen_mcp` (or `./run.sh` when invoked by clients); export at least one provider API key first.
- Format and lint: `uv run ruff format imagen_mcp src tests && uv run ruff check imagen_mcp src tests`.
- Type check: `uv run mypy imagen_mcp src`.
- Tests: `uv run pytest` (verbosity and discovery configured in `pytest.ini`).
- Build and smoke distributions: `uv build`, then clean-install the wheel/sdist and run `tests/installed_distribution_smoke.py` outside the checkout.

## Coding Style & Naming Conventions
- Python 3.10+; Ruff line length 100; prefer explicit imports and typed signatures (mypy is strict: no implicit Optional, no untyped defs).
- Use snake_case for modules/functions, PascalCase for classes; keep provider IDs consistent with registry keys (`openai`, `gemini`).
- Keep side effects out of import time; guard script entry with `if __name__ == "__main__":` when needed.

## Testing Guidelines
- Add or extend tests in `tests/` near the related module using the `test_*.py`/`Test*`/`test_*` pattern.
- Mock external APIs—reuse dummy env vars as in `tests/test_selector.py`; avoid live requests in CI.
- Cover new branching in provider selection, config defaults, and tool metadata (reasoning, confidence, alternatives) when relevant.

## Commit & Pull Request Guidelines
- Follow Conventional Commits as in history (`feat:`, `fix:`, `docs:`, `chore:`); keep subjects under ~72 characters.
- PRs should summarize intent, list testing (`ruff`, `mypy`, `pytest`), and note impacted MCP tools or endpoints.
- Link issues when applicable and include before/after examples or log snippets for behavior changes.

## Security & Configuration Tips
- Never commit API keys; load via env (`OPENAI_API_KEY`, `GEMINI_API_KEY`, optional `GOOGLE_API_KEY` alias).
- Prefer `output_path` overrides for local testing to avoid cluttering `~/Downloads/images/`.
- Avoid logging sensitive prompts or keys; remove debug prints before merging.
