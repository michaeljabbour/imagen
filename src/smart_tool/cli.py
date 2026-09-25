"""Thin CLI wrapper over :mod:`src.smart_tool.lib`.

Per the smart tools spec, the CLI hooks up argument parsing and I/O
conventions and calls the library -- it adds no capability of its own.
``-h`` is the terse per-command summary; ``--help`` on the tool (no
subcommand) renders the tool's own skill (manifest body) so an agent reading
it gets the same guidance as a person reading ``SMART_TOOL.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from . import lib

_CAPABILITIES: dict[str, tuple[str, bool]] = {
    "manifest": ("Print the tool's own manifest as JSON.", False),
    "list-providers": ("List providers, availability, and capabilities.", False),
    "estimate-cost": ("Estimate generation cost for a provider/model/size/quality.", False),
    "generate-image": ("Generate an image from a text prompt.", True),
    "edit-image": ("Edit an existing image via OpenAI's /images/edits.", True),
}


def _render_skill() -> str:
    data = lib.manifest()
    frontmatter = data["frontmatter"]
    lines = [
        f'<skill_content name="{frontmatter.get("name", "imagen-mcp")}">',
        f"Repository: {frontmatter.get('repository', 'https://github.com/michaeljabbour/imagen-mcp')}",
        "",
        data["body"],
        "",
        "## Capabilities",
        "",
        "Each has its own skill: `imagen-mcp-smart <capability> --help`.",
        "",
    ]
    for name, (summary, model_backed) in _CAPABILITIES.items():
        tag = "model-backed" if model_backed else "deterministic"
        lines.append(f"- `{name}` [{tag}] -- {summary}")
    lines.append("")
    lines.append("</skill_content>")
    return "\n".join(lines)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imagen-mcp-smart",
        description="Model-agnostic multi-provider image generation (smart tool).",
    )
    subparsers = parser.add_subparsers(dest="capability")

    manifest_p = subparsers.add_parser("manifest", help=_CAPABILITIES["manifest"][0])
    manifest_p.set_defaults(func=lambda args: lib.manifest())

    list_p = subparsers.add_parser("list-providers", help=_CAPABILITIES["list-providers"][0])
    list_p.set_defaults(func=lambda args: lib.list_providers())

    cost_p = subparsers.add_parser("estimate-cost", help=_CAPABILITIES["estimate-cost"][0])
    cost_p.add_argument("--provider", default=None)
    cost_p.add_argument("--prompt", default="")
    cost_p.add_argument("--model", default=None)
    cost_p.add_argument("--quality", default=None)
    cost_p.add_argument("--size", default=None)
    cost_p.add_argument("--n", type=int, default=1)
    cost_p.set_defaults(
        func=lambda args: lib.estimate_cost(
            provider=args.provider,
            prompt=args.prompt,
            model=args.model,
            quality=args.quality,
            size=args.size,
            n=args.n,
        )
    )

    gen_p = subparsers.add_parser("generate-image", help=_CAPABILITIES["generate-image"][0])
    gen_p.add_argument("prompt")
    gen_p.add_argument("--provider", default=None)
    gen_p.add_argument("--model", default=None)
    gen_p.add_argument("--size", default=None)
    gen_p.add_argument("--aspect-ratio", default=None)
    gen_p.add_argument("--output-path", default=None)
    gen_p.add_argument("--quality", default=None)
    gen_p.add_argument("--background", default=None)
    gen_p.add_argument("--enable-google-search", action="store_true")
    gen_p.add_argument("--n", type=int, default=None)
    gen_p.set_defaults(
        func=lambda args: asyncio.run(
            lib.generate_image(
                args.prompt,
                provider=args.provider,
                model=args.model,
                size=args.size,
                aspect_ratio=args.aspect_ratio,
                output_path=args.output_path,
                quality=args.quality,
                background=args.background,
                enable_google_search=args.enable_google_search,
                n=args.n,
            )
        )
    )

    edit_p = subparsers.add_parser("edit-image", help=_CAPABILITIES["edit-image"][0])
    edit_p.add_argument("prompt")
    edit_p.add_argument("image_path")
    edit_p.add_argument("--model", default=None)
    edit_p.add_argument("--mask-path", default=None)
    edit_p.add_argument("--size", default=None)
    edit_p.add_argument("--quality", default=None)
    edit_p.add_argument("--background", default=None)
    edit_p.add_argument("--output-path", default=None)
    edit_p.set_defaults(
        func=lambda args: asyncio.run(
            lib.edit_image(
                args.prompt,
                args.image_path,
                model=args.model,
                mask_path=args.mask_path,
                size=args.size,
                quality=args.quality,
                background=args.background,
                output_path=args.output_path,
            )
        )
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--help" in argv and (len(argv) == 1 or argv[0] == "--help"):
        print(_render_skill())
        return 0

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.capability is None:
        parser.print_help()
        return 1

    try:
        result = args.func(args)
    except lib.SmartToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
