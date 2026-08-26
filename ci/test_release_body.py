"""
Unit tests for ``release_body.py`` (Markdown release-notes extraction).

This ``ci/`` script has no existing pytest CI wiring in this repo; run
directly with::

    pytest ci/test_release_body.py

Covers the two guarantees the publish workflow relies on: the emitted body is
exactly the tag's section with the heading stripped, and ``check_newest``
accepts the newest real release (ignoring any leading ``Unreleased`` block)
while rejecting anything else.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release_body",
    Path(__file__).resolve().parent / "release_body.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


_RELEASES = (
    "# Release notes\n"
    "\n"
    "## Unreleased\n"
    "\n"
    "### New Features\n"
    "\n"
    "* Work in progress ([#99](https://example/pull/99))\n"
    "\n"
    "## 0.5.0\n"
    "\n"
    "### Breaking Changes\n"
    "\n"
    "* Requires cstar-ocean 0.11.0 ([#137](https://example/pull/137))\n"
    "\n"
    "### Bug Fixes\n"
    "\n"
    "* Fixed a thing ([#121](https://example/pull/121))\n"
    "\n"
    "## 0.4.0\n"
    "\n"
    "### New Features\n"
    "\n"
    "* Older feature ([#113](https://example/pull/113))\n"
)


@pytest.fixture
def releases_file(tmp_path, monkeypatch):
    path = tmp_path / "releases.md"
    path.write_text(_RELEASES)
    monkeypatch.setattr(_MODULE, "RELEASES_MD", path)
    return path


def test_section_body_strips_heading_and_stops_at_next_section(releases_file):
    body = _MODULE.section_body("0.5.0")
    assert body == (
        "### Breaking Changes\n"
        "\n"
        "* Requires cstar-ocean 0.11.0 ([#137](https://example/pull/137))\n"
        "\n"
        "### Bug Fixes\n"
        "\n"
        "* Fixed a thing ([#121](https://example/pull/121))\n"
    )
    # Neither the Unreleased block above nor the 0.4.0 block below leaks in.
    assert "Unreleased" not in body
    assert "0.4.0" not in body and "Older feature" not in body


def test_section_body_accepts_v_prefix(releases_file):
    assert _MODULE.section_body("v0.5.0") == _MODULE.section_body("0.5.0")


def test_section_body_missing_tag_exits(releases_file):
    with pytest.raises(SystemExit):
        _MODULE.section_body("9.9.9")


def test_check_newest_ignores_unreleased(releases_file):
    # 0.5.0 is the newest *real* section even though Unreleased sits above it.
    _MODULE.check_newest("0.5.0")


def test_check_newest_rejects_stale_tag(releases_file):
    with pytest.raises(SystemExit):
        _MODULE.check_newest("0.4.0")


def test_normalize_tag():
    assert _MODULE.normalize_tag("v0.5.0") == "0.5.0"
    assert _MODULE.normalize_tag("0.5.0") == "0.5.0"


def test_heading_inside_fenced_code_block_is_not_a_section(tmp_path, monkeypatch):
    path = tmp_path / "releases.md"
    path.write_text(
        "# Release notes\n"
        "\n"
        "## 0.5.0\n"
        "\n"
        "### New Features\n"
        "\n"
        "* Example config:\n"
        "\n"
        "  ```yaml\n"
        "  ## not a heading, just a comment in a code block\n"
        "  key: value\n"
        "  ```\n"
        "\n"
        "* Another real note\n"
        "\n"
        "## 0.4.0\n"
        "\n"
        "### New Features\n"
        "\n"
        "* Older\n"
    )
    monkeypatch.setattr(_MODULE, "RELEASES_MD", path)
    body = _MODULE.section_body("0.5.0")
    # The whole 0.5.0 section survives; the fenced '## ' did not split it.
    assert "Another real note" in body
    assert "key: value" in body
    # And it still stops at the real 0.4.0 heading.
    assert "Older" not in body


def test_empty_section_body_exits(tmp_path, monkeypatch):
    path = tmp_path / "releases.md"
    path.write_text("# Release notes\n\n## 0.5.0\n\n## 0.4.0\n\nolder\n")
    monkeypatch.setattr(_MODULE, "RELEASES_MD", path)
    with pytest.raises(SystemExit):
        _MODULE.section_body("0.5.0")
