"""
Tests for ``number_fragments.py`` (naming fragments after their PR at release).

Run with ``pytest ci/test_number_fragments.py``; like the other ``ci/`` tests,
CI does not run it. The end-to-end test also needs ``towncrier`` (in the
``dev`` extra) and is skipped without it.

Each test commits fragments to a scratch git repository with the real
``[tool.towncrier]`` config, the way squash merges put them on ``main``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
URL = "https://github.com/CWorthy-ocean/roms-tools/pull"

_SPEC = importlib.util.spec_from_file_location(
    "number_fragments", REPO / "ci" / "number_fragments.py"
)
assert _SPEC is not None and _SPEC.loader is not None
nf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nf)

# Keep the developer's git config (hooks, signing, identity) out of the tests.
_GIT = [
    "git",
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@example.invalid",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "commit.gpgsign=false",
]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [*_GIT, *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _commit(repo: Path, files: dict[str, str], subject: str) -> None:
    for name, text in files.items():
        (repo / "changes" / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", subject)


def _no_lookup(sha: str) -> int | None:
    return None


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with the real towncrier config, template and README."""
    shutil.copy(REPO / "pyproject.toml", tmp_path)
    (tmp_path / "ci").mkdir()
    shutil.copy(REPO / "ci" / "towncrier_template.md", tmp_path / "ci")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "releases.md").write_text(
        "# Release notes\n\n<!-- towncrier release notes start -->\n\n## 5.3.0\n"
    )
    (tmp_path / "changes").mkdir()
    shutil.copy(REPO / "changes" / "README.md", tmp_path / "changes")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "Start")
    return tmp_path


def _names(repo: Path) -> list[str]:
    return sorted(p.name for p in (repo / "changes").iterdir())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("+river-order.bugfix.md", "bugfix"),
        ("+v1.2.feature.1.md", "feature"),
        ("+no-extension.misc", "misc"),
        ("+other.feature.txt", "feature"),
        ("+misc.bugfix.md", "bugfix"),
        ("+thing.bugfixes.md", None),
    ],
)
def test_fragment_type(name, expected):
    types = nf.fragment_types(REPO / "pyproject.toml")
    assert nf.fragment_type(name, types) == expected


def test_pr_number_from_squash_subject(repo):
    _commit(repo, {"12.feature.md": "Numbered.\n"}, "Numbered (#12)")
    _commit(repo, {"+fill.bugfix.md": "A fix.\n"}, "Fix the fill (#40)")
    assert nf.number_fragments(repo, lookup=_no_lookup) == {
        "+fill.bugfix.md": "40.bugfix.md"
    }
    assert _names(repo) == ["12.feature.md", "40.bugfix.md", "README.md"]


def test_counters_avoid_existing_names(repo):
    _commit(
        repo,
        {"40.bugfix.md": "One.\n", "+a.bugfix.md": "Two.\n", "+b.bugfix.md": "3.\n"},
        "Fixes (#40)",
    )
    nf.number_fragments(repo, lookup=_no_lookup)
    assert _names(repo) == [
        "40.bugfix.1.md",
        "40.bugfix.2.md",
        "40.bugfix.md",
        "README.md",
    ]


def test_github_lookup_when_subject_has_no_number(repo):
    _commit(repo, {"+note.misc.md": "A note.\n"}, "A merge commit's branch commit")
    seen = []

    def lookup(sha):
        seen.append(sha)
        return 41

    assert nf.number_fragments(repo, lookup=lookup) == {"+note.misc.md": "41.misc.md"}
    assert len(seen) == 1


def test_unresolved_fragment_stops_before_any_rename(repo):
    _commit(repo, {"+ok.misc.md": "Fine.\n"}, "Fine (#42)")
    _commit(repo, {"+lost.misc.md": "Lost.\n"}, "No number here")
    (repo / "changes" / "+uncommitted.misc.md").write_text("New.\n")
    with pytest.raises(SystemExit, match=r"\+lost\.misc\.md.*\+uncommitted\.misc"):
        nf.number_fragments(repo, lookup=_no_lookup)
    assert "+ok.misc.md" in _names(repo)


def test_renamed_fragment_keeps_the_pr_that_wrote_it(repo):
    _commit(repo, {"+note.misc.md": "A longer note about something.\n"}, "Add (#43)")
    _git(repo, "mv", "changes/+note.misc.md", "changes/+note.bugfix.md")
    _git(repo, "commit", "-q", "-m", "Fix the note's type (#44)")
    assert nf.number_fragments(repo, lookup=_no_lookup) == {
        "+note.bugfix.md": "43.bugfix.md"
    }


def test_any_extension_and_untyped_files(repo):
    _commit(repo, {"+plain.misc": "No extension.\n"}, "Add (#45)")
    assert nf.number_fragments(repo, lookup=_no_lookup) == {"+plain.misc": "45.misc.md"}
    _commit(repo, {"+typo.bugfixes.md": "Misnamed.\n"}, "Add (#46)")
    with pytest.raises(SystemExit, match=r"No fragment type in changes/\+typo"):
        nf.number_fragments(repo, lookup=_no_lookup)


def test_failed_github_lookup_is_reported(repo, monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "HTTP 401: Bad credentials")

    monkeypatch.setattr(nf.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="abc123 failed: HTTP 401: Bad credentials"):
        nf.pr_from_github("abc123")


def test_release_commit_holds_only_notes_and_deletions(repo):
    pytest.importorskip("towncrier")
    _commit(repo, {"+fill.bugfix.md": "A fix.\n"}, "Fix the fill (#40)")
    nf.number_fragments(repo, lookup=_no_lookup)
    subprocess.run(
        [sys.executable, "-m", "towncrier", "build", "--yes", "--version", "5.4.0"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert f"* A fix. ([#40]({URL}/40))" in (repo / "docs" / "releases.md").read_text()
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == [
        "D  changes/+fill.bugfix.md",
        "M  docs/releases.md",
    ]
