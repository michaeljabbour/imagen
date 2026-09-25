"""Smoke tests for the imagen CLI (thin wrapper over lib)."""

from __future__ import annotations

import json

import pytest

from src.smart_tool import cli


def test_manifest_capability_runs_deterministically(capsys):
    exit_code = cli.main(["manifest"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frontmatter"]["name"] == "imagen"


def test_list_providers_capability(capsys):
    exit_code = cli.main(["list-providers"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "providers" in payload


def test_top_level_help_renders_skill(capsys):
    exit_code = cli.main(["--help"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert '<skill_content name="imagen">' in out
    assert "## Capabilities" in out


def test_no_capability_prints_usage_and_fails(capsys):
    exit_code = cli.main([])
    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_smart_tool_error_exits_nonzero_with_message(capsys, monkeypatch):
    def _boom(*args, **kwargs):
        raise cli.lib.SmartToolError("no provider configured, set OPENAI_API_KEY")

    monkeypatch.setattr(cli.lib, "estimate_cost", _boom)
    exit_code = cli.main(["estimate-cost"])
    assert exit_code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("capability", ["manifest", "list-providers", "estimate-cost"])
def test_deterministic_capabilities_never_need_a_provider_key(capability, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(cli.lib, "_keychain_api_key", lambda env_var: None)
    exit_code = cli.main([capability])
    assert exit_code == 0
