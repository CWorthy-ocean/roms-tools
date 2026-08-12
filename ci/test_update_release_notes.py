"""
Unit tests for ``parse_pr_body`` in ``update_release_notes.py``.

This ``ci/`` script has no existing pytest CI wiring in this repo; run
directly with::

    pytest ci/test_update_release_notes.py

Focused on the un-bulleted-prose fallback: authors regularly delete the PR
template's ``- `` markers and write plain sentences under a category, and
those notes used to be dropped silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "update_release_notes", Path(__file__).resolve().parent / "update_release_notes.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_pr_body = _MODULE.parse_pr_body


def test_bulleted_sections_unchanged():
    body = (
        "# Summary\nDoes a thing.\n\n## Bug Fixes\n- Fixed one thing\n- Fixed another\n"
    )
    assert parse_pr_body(body) == {
        "Bug Fixes": [("Fixed one thing", []), ("Fixed another", [])]
    }


def test_unbulleted_prose_becomes_a_note():
    body = "## Bug Fixes\nPassive tracers were not written to the output files.\n"
    assert parse_pr_body(body) == {
        "Bug Fixes": [("Passive tracers were not written to the output files.", [])]
    }


def test_wrapped_prose_joins_into_one_note_per_paragraph():
    body = "## Improvements\nFirst paragraph line one\nline two.\n\nSecond paragraph.\n"
    assert parse_pr_body(body) == {
        "Improvements": [
            ("First paragraph line one line two.", []),
            ("Second paragraph.", []),
        ]
    }


def test_prose_ignored_when_the_section_also_has_bullets():
    # An introductory paragraph followed by bullets must still yield only the
    # bullets, exactly as before the fallback existed.
    body = (
        "## New Features\nThis PR adds the following:\n\n- Feature one\n- Feature two\n"
    )
    assert parse_pr_body(body) == {
        "New Features": [("Feature one", []), ("Feature two", [])]
    }


def test_placeholder_prose_and_bullets_are_dropped():
    body = "## Breaking Changes\nNone\n\n## New Features\n- N/A\n"
    assert parse_pr_body(body) == {}


def test_summary_and_checklist_prose_never_scraped():
    body = (
        "# Summary\n"
        "A long prose summary that must not become a release note.\n"
        "\n"
        "## Code Review Checklist\n"
        "Everything looks fine to me.\n"
    )
    assert parse_pr_body(body) == {}


def test_prose_under_an_invented_heading_is_ignored():
    # PR authors add narrative headings ("Testing", "Setup", …); only the
    # template's own categories are eligible for the prose fallback.
    body = "## Testing\nRan the full suite locally and it passed.\n"
    assert parse_pr_body(body) == {}


def test_fenced_code_and_markup_lines_excluded():
    body = (
        "## Miscellaneous\n"
        "Now supports the new flag.\n"
        "\n"
        "```bash\n"
        "run --with-flag\n"
        "```\n"
        "\n"
        '<img width="600" src="https://example.invalid/x.png" />\n'
        "\n"
        "| col | col |\n"
        "|-----|-----|\n"
    )
    assert parse_pr_body(body) == {
        "Miscellaneous": [("Now supports the new flag.", [])]
    }


def test_html_comments_do_not_become_notes():
    body = (
        "## Bug Fixes\n"
        "<!-- List any behavioral changes resulting from pre-existing code -->\n"
        "Fixed the thing.\n"
    )
    assert parse_pr_body(body) == {"Bug Fixes": [("Fixed the thing.", [])]}


def test_sub_bullets_still_attach_to_their_parent():
    body = "## Improvements\n- Parent note\n  - child one\n  - child two\n"
    assert parse_pr_body(body) == {
        "Improvements": [("Parent note", ["child one", "child two"])]
    }
