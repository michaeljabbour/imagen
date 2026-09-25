"""The installed smart tool must be able to read its own manifest.

An installed distribution has no repo root, so SMART_TOOL.md ships inside the
package too. The two copies must stay byte-identical.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_packaged_manifest_matches_repo_root_manifest():
    root = (REPO / "SMART_TOOL.md").read_bytes()
    packaged = (REPO / "src" / "smart_tool" / "SMART_TOOL.md").read_bytes()
    assert packaged == root, "copy SMART_TOOL.md into src/smart_tool/ after editing it"


def test_manifest_is_declared_as_package_data():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"smart_tool/SMART_TOOL.md"' in text


def test_lib_prefers_the_packaged_manifest():
    from src.smart_tool import lib

    assert lib._SMART_TOOL_MD_PATH == lib._PACKAGED_MD_PATH
    assert lib.manifest()
