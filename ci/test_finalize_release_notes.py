"""
Unit tests for ``drop_empty_sections`` in ``finalize_release_notes.py``.

This ``ci/`` script has no existing pytest CI wiring in this repo; run
directly with::

    pytest ci/test_finalize_release_notes.py

Covers the guarantees that make the drop safe: only the block being
finalized is touched, hand-written content is never removed, and the
Markdown stays well-formed no matter which categories disappear.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "finalize_release_notes",
    Path(__file__).resolve().parent / "finalize_release_notes.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
drop_empty_sections = _MODULE.drop_empty_sections
find_unreleased_heading = _MODULE.find_unreleased_heading
_is_placeholder = _MODULE._is_placeholder


def _run(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    idx = find_unreleased_heading(lines)
    assert idx is not None
    new_lines, dropped = drop_empty_sections(lines, idx)
    return "".join(new_lines), dropped


def test_empty_and_placeholder_categories_dropped():
    text = (
        "# Release notes\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Breaking Changes\n"
        "\n"
        "* A real note\n"
        "\n"
        "### New Features\n"
        "\n"
        "* N/A\n"
        "\n"
        "### Bug Fixes\n"
        "\n"
    )
    out, dropped = _run(text)
    assert dropped == ["New Features", "Bug Fixes"]
    assert "New Features" not in out
    assert "Bug Fixes" not in out
    assert "* A real note\n" in out


def test_already_released_blocks_untouched():
    text = (
        "# Release notes\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Improvements\n"
        "\n"
        "### Miscellaneous\n"
        "\n"
        "* A real note\n"
        "\n"
        "## 0.2.0\n"
        "\n"
        "### Improvements\n"
        "\n"
        "### Miscellaneous\n"
    )
    out, dropped = _run(text)
    assert dropped == ["Improvements"]
    # The published 0.2.0 block keeps both of its empty categories
    tail = out.split("## 0.2.0", 1)[1]
    assert "### Improvements" in tail
    assert "### Miscellaneous" in tail


def test_blank_line_kept_before_next_release_heading():
    # Dropping the block's *last* category removes the blank line that
    # separated it from the next "## " heading; it must be restored.
    text = (
        "# Release notes\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Breaking Changes\n"
        "\n"
        "* A real note\n"
        "\n"
        "### Miscellaneous\n"
        "\n"
        "## 0.2.0\n"
    )
    out, _ = _run(text)
    assert "* A real note\n\n## 0.2.0\n" == out.split("### Breaking Changes\n\n", 1)[1]


def test_hand_written_prose_is_not_dropped():
    text = (
        "# Release notes\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Improvements\n"
        "\n"
        "See the migration guide for details.\n"
    )
    _, dropped = _run(text)
    assert dropped == []


def test_all_categories_empty_leaves_valid_markdown():
    text = (
        "# Release notes\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Breaking Changes\n"
        "\n"
        "### Bug Fixes\n"
        "\n"
        "## 0.2.0\n"
        "\n"
        "### Improvements\n"
        "\n"
        "* old\n"
    )
    out, dropped = _run(text)
    assert dropped == ["Breaking Changes", "Bug Fixes"]
    assert (
        out
        == "# Release notes\n\n## Unreleased\n\n## 0.2.0\n\n### Improvements\n\n* old\n"
    )


def test_nothing_dropped_when_every_category_has_notes():
    text = "# Release notes\n\n## Unreleased\n\n### Bug Fixes\n\n* A real note\n"
    out, dropped = _run(text)
    assert dropped == []
    assert out == text


def test_placeholder_recognition():
    for value in ("N/A", "n/a", "None", "none.", "Nothing", "No changes"):
        assert _is_placeholder(value), value
    for value in (
        "Nonetheless, we fixed it",
        "No longer crashes",
        "N/A handling added",
    ):
        assert not _is_placeholder(value), value
