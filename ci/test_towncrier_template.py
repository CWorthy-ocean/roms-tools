"""
Tests for the towncrier setup that turns ``changes/`` into ``docs/releases.md``.

Run with ``pytest ci/test_towncrier_template.py``. Needs ``towncrier`` (in the
``dev`` extra) and is skipped without it; like the other ``ci/`` tests, CI does
not run it.

Each test builds a release from sample fragments with the repository's own
``[tool.towncrier]`` config and template, in a scratch git repository, exactly as
``finalize_release_notes.yaml`` does, then checks the result against the style
of the releases already in the file and reads it back with ``release_body.py``,
as ``publish_release.yaml`` does.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("towncrier")

REPO = Path(__file__).resolve().parent.parent
START = "<!-- towncrier release notes start -->\n"
URL = "https://github.com/CWorthy-ocean/roms-tools/pull"
OLD_RELEASE = f"## 5.3.0\n\n### Bug Fixes\n\n* An old fix. ([#1]({URL}/1))\n"

_SPEC = importlib.util.spec_from_file_location(
    "release_body", REPO / "ci" / "release_body.py"
)
assert _SPEC is not None and _SPEC.loader is not None
release_body = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_body)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    """A git repository holding the real towncrier config and template."""
    shutil.copy(REPO / "pyproject.toml", tmp_path)
    (tmp_path / "ci").mkdir()
    shutil.copy(REPO / "ci" / "towncrier_template.md", tmp_path / "ci")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "releases.md").write_text(
        f"# Release notes\n\n{START}\n{OLD_RELEASE}"
    )
    (tmp_path / "changes").mkdir()
    shutil.copy(REPO / "changes" / "README.md", tmp_path / "changes")
    _run(tmp_path, "git", "init", "-q")
    _run(tmp_path, "git", "add", "-A")
    return tmp_path


def _add_fragments(repo: Path, fragments: dict[str, str]) -> None:
    for name, text in fragments.items():
        (repo / "changes" / name).write_text(text)
    _run(repo, "git", "add", "-A")


def _build(repo: Path, version: str) -> subprocess.CompletedProcess[str]:
    return _run(
        repo, sys.executable, "-m", "towncrier", "build", "--yes", "--version", version
    )


def test_real_releases_file_has_the_start_marker():
    assert START in (REPO / "docs" / "releases.md").read_text()


def test_build_matches_release_notes_style(scratch, monkeypatch):
    _add_fragments(
        scratch,
        {
            "12.breaking.md": "Lead text.\n* First detail.\n* Second detail.\n",
            "14.deprecation.md": "Deprecated: `a`, removed in 6.0.0; use `b`.\n",
            "12.feature.md": "A feature.\n",
            "13.bugfix.md": "A later fix.\n",
            "11.bugfix.md": "An earlier fix.\n",
            "15.misc.1.md": "One note.\n",
            "15.misc.2.md": "Another note.\n",
        },
    )
    result = _build(scratch, "5.4.0")
    assert result.returncode == 0, result.stderr

    body = (
        "### Breaking Changes\n\n"
        f"* Lead text. ([#12]({URL}/12))\n"
        "  * First detail.\n"
        "  * Second detail.\n\n"
        "### Deprecations\n\n"
        f"* Deprecated: `a`, removed in 6.0.0; use `b`. ([#14]({URL}/14))\n\n"
        "### New Features\n\n"
        f"* A feature. ([#12]({URL}/12))\n\n"
        "### Bug Fixes\n\n"
        f"* An earlier fix. ([#11]({URL}/11))\n"
        f"* A later fix. ([#13]({URL}/13))\n\n"
        "### Miscellaneous\n\n"
    )
    text = (scratch / "docs" / "releases.md").read_text()
    head, misc_and_rest = text.split("### Miscellaneous\n\n")
    assert head + "### Miscellaneous\n\n" == (
        f"# Release notes\n\n{START}\n## 5.4.0\n\n{body}"
    )
    # Two notes from one PR and type come out in directory-listing order.
    misc, rest = misc_and_rest.split("\n\n", 1)
    assert sorted(misc.splitlines()) == [
        f"* Another note. ([#15]({URL}/15))",
        f"* One note. ([#15]({URL}/15))",
    ]
    assert rest == OLD_RELEASE
    assert [p.name for p in (scratch / "changes").iterdir()] == ["README.md"]

    monkeypatch.setattr(release_body, "RELEASES_MD", scratch / "docs" / "releases.md")
    release_body.check_newest("5.4.0")
    assert release_body.section_body("5.4.0") == body + misc + "\n"


def test_fragment_written_as_a_bullet(scratch):
    _add_fragments(scratch, {"16.feature.md": "- A feature.\n  * A detail.\n"})
    result = _build(scratch, "5.4.0")
    assert result.returncode == 0, result.stderr
    text = (scratch / "docs" / "releases.md").read_text()
    assert (
        f"### New Features\n\n* A feature. ([#16]({URL}/16))\n    * A detail.\n\n"
        f"{OLD_RELEASE}"
    ) in text


def test_link_stays_on_the_lead(scratch):
    _add_fragments(
        scratch,
        {
            "17.feature.md": "A feature.\n\n* A detail.\n",
            "18.bugfix.md": "A fix:\n```python\nx = 1\n```\n",
        },
    )
    result = _build(scratch, "5.4.0")
    assert result.returncode == 0, result.stderr
    text = (scratch / "docs" / "releases.md").read_text()
    # A blank line before the sub-bullets is dropped.
    assert f"* A feature. ([#17]({URL}/17))\n  * A detail.\n" in text
    # The closing fence stays a fence.
    assert f"  x = 1\n  ```\n  ([#18]({URL}/18))\n" in text


def test_build_without_fragments_is_not_empty(scratch, monkeypatch):
    result = _build(scratch, "5.4.0")
    assert result.returncode == 0, result.stderr
    text = (scratch / "docs" / "releases.md").read_text()
    assert text == (
        f"# Release notes\n\n{START}\n## 5.4.0\n\nNo significant changes.\n\n"
        f"{OLD_RELEASE}"
    )
    # release_body.py refuses an empty section, which would stop the release.
    monkeypatch.setattr(release_body, "RELEASES_MD", scratch / "docs" / "releases.md")
    assert release_body.section_body("5.4.0") == "No significant changes.\n"


@pytest.mark.parametrize("name", ["PR.misc.md", "12.features.md"])
def test_fragment_names_are_checked(scratch, name):
    _add_fragments(scratch, {name: "A note.\n"})
    result = _build(scratch, "5.4.0")
    assert result.returncode != 0
    assert (scratch / "docs" / "releases.md").read_text().endswith(OLD_RELEASE)
