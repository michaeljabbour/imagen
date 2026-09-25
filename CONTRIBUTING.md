# Contributing to imagen

Thank you for your interest in contributing!

## Development Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/michaeljabbour/imagen.git
    cd imagen
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    uv sync --extra dev
    ```

## Code Quality

We use `ruff` for linting and formatting, and `mypy` for static type checking.

### Linting & Formatting

Run the following command to format code and fix linting issues:

```bash
uv run ruff format imagen_mcp src tests
uv run ruff check imagen_mcp src tests
```

### Type Checking

Run `mypy` to check for type errors:

```bash
uv run mypy imagen_mcp src
```

Ensure all type checks pass before submitting a pull request.

## Testing

Run the test suite using `pytest`:

```bash
uv run pytest tests/ -v
```

## Pull Request Process

1.  Create a new branch for your feature or fix.
2.  Make your changes.
3.  Run linters, type checks, and tests.
4.  Submit a PR with a clear description of your changes.
