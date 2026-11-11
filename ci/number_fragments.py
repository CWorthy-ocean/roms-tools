#!/usr/bin/env python3
"""
Name each unnumbered release-note fragment after the pull request that added it.

Contributors name a fragment ``changes/+<name>.<type>.md`` because the PR number
does not exist until the PR is opened. Before ``towncrier build``,
``finalize_release_notes.yaml`` runs this script, which ``git mv``s every such
file to ``changes/<PR>.<type>.md`` so the note links to its PR.

The PR is the one that added the file, looking through later renames: the
trailing ``(#N)`` of the squash-merge commit's subject, or, when the subject has
none, the merged PR that GitHub associates with that commit. A ``+`` file whose
type or PR cannot be found, or a failed GitHub lookup, stops the script with an
error, so a release never goes out with an unlinked note by accident.
Fragments already named ``<PR>.<type>.md`` are left alone.

Usage:
    python ci/number_fragments.py [--dry-run]

Environment:
    GH_TOKEN: token for the GitHub API fallback (the workflow's GITHUB_TOKEN).
    GITHUB_REPOSITORY: owner/repo to query; defaults to CWorthy-ocean/roms-tools.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORPHAN_PREFIX = "+"
_SUBJECT_PR_RE = re.compile(r"\(#(\d+)\)\s*$")


def fragment_types(pyproject: Path) -> list[str]:
    """Return the fragment type names (``directory`` keys) from *pyproject*."""
    config = tomllib.loads(pyproject.read_text())["tool"]["towncrier"]
    return [t["directory"] for t in config["type"]]


def fragment_type(name: str, types: list[str]) -> str | None:
    """
    Return the type of an unnumbered ``+<label>.<type>[...]`` fragment name.

    Mirrors towncrier: the last dot-separated part that is a known type, after
    the label; anything after it (``.md``, a counter, another extension) is
    ignored. Returns ``None`` when no type is found.
    """
    parts = name[len(ORPHAN_PREFIX) :].split(".")
    for part in reversed(parts[1:]):
        if part in types:
            return part
    return None


def pr_from_subject(subject: str) -> int | None:
    """Return *N* from a squash-merge subject ending in ``(#N)``, else ``None``."""
    m = _SUBJECT_PR_RE.search(subject)
    return int(m.group(1)) if m else None


def pr_from_github(sha: str) -> int | None:
    """Return the merged PR GitHub associates with commit *sha*, if any."""
    repo = os.environ.get("GITHUB_REPOSITORY", "CWorthy-ocean/roms-tools")
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/commits/{sha}/pulls",
                "--jq",
                "[.[] | select(.merged_at != null) | .number] | first // empty",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("The GitHub CLI (gh) is needed to look up pull requests.")
    if result.returncode != 0:
        # A bad token or rate limit is not "no pull request": say what failed.
        sys.exit(f"GitHub lookup for commit {sha} failed: {result.stderr.strip()}")
    out = result.stdout.strip()
    return int(out) if out.isdigit() else None


def adding_commit(repo: Path, path: str) -> tuple[str, str] | None:
    """
    Return *(sha, subject)* of the commit that added *path*, or ``None``.

    ``--follow`` looks through renames, so a fragment that a later PR renamed
    (to fix its type, say) still counts as added by the PR that wrote it.
    ``--literal-pathspecs`` keeps a ``*`` or ``[`` in a name from matching
    other fragments.
    """
    out = subprocess.run(
        [
            "git",
            "--literal-pathspecs",
            "log",
            "--follow",
            "--diff-filter=A",
            "--format=%H%x00%s",
            "-n",
            "1",
            "--",
            path,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not out:
        return None
    sha, subject = out.split("\0", 1)
    return sha, subject


def target_name(changes: Path, pr: int, ftype: str, taken: set[str]) -> str:
    """Return a free ``<pr>.<type>[.<n>].md`` name in *changes*."""
    name = f"{pr}.{ftype}.md"
    n = 0
    while name in taken or (changes / name).exists():
        n += 1
        name = f"{pr}.{ftype}.{n}.md"
    return name


def number_fragments(
    repo: Path,
    *,
    dry_run: bool = False,
    lookup: Callable[[str], int | None] = pr_from_github,
) -> dict[str, str]:
    """
    Rename every ``+<label>.<type>.md`` in ``changes/`` after its PR.

    Returns ``{old name: new name}``. Exits with an error, before renaming
    anything, listing every ``+`` file whose type or PR could not be found.

    Args:
        repo: Repository root holding ``pyproject.toml`` and ``changes/``.
        dry_run: Report the renames without making them.
        lookup: Commit sha -> PR number, used when the subject has no ``(#N)``.
    """
    types = fragment_types(repo / "pyproject.toml")
    changes = repo / "changes"
    plan: dict[str, str] = {}
    taken: set[str] = set()
    unresolved: list[str] = []
    untyped: list[str] = []
    for path in sorted(changes.iterdir()):
        if not path.name.startswith(ORPHAN_PREFIX):
            continue
        ftype = fragment_type(path.name, types)
        if ftype is None:
            untyped.append(path.name)
            continue
        commit = adding_commit(repo, f"changes/{path.name}")
        pr = None
        if commit is not None:
            sha, subject = commit
            pr = pr_from_subject(subject) or lookup(sha)
        if pr is None:
            unresolved.append(path.name)
            continue
        new = target_name(changes, pr, ftype, taken)
        taken.add(new)
        plan[path.name] = new

    problems = []
    if untyped:
        problems.append(
            "No fragment type in "
            + ", ".join(f"changes/{n}" for n in untyped)
            + f"; the types are {', '.join(types)}."
        )
    if unresolved:
        problems.append(
            "Could not find the pull request that added "
            + ", ".join(f"changes/{n}" for n in unresolved)
            + "."
        )
    if problems:
        sys.exit(
            " ".join(problems)
            + " Rename each to changes/<PR number>.<type>.md by hand."
        )
    for old, new in plan.items():
        print(f"changes/{old} -> changes/{new}")
        if not dry_run:
            subprocess.run(
                ["git", "mv", f"changes/{old}", f"changes/{new}"],
                cwd=repo,
                check=True,
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the renames only."
    )
    args = parser.parse_args()
    number_fragments(REPO_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
