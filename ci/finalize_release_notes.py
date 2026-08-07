#!/usr/bin/env python3
"""
Finalize the in-development release notes for a tagged release.

Locates the active "unreleased" ``## `` block in ``docs/releases.md``
(identified by its heading — exactly ``Unreleased``, or ``vX.Y.Z
(unreleased)`` / ``X.Y.Z (unreleased)`` — not by any external state), and
rewrites that heading to ``## <tag>``.

Release tags are written *without* a leading ``v`` (e.g. ``0.7.0``, not
``v0.7.0``) — this is the standardized convention going forward across all
CWorthy repos. If invoked with a ``v``-prefixed tag it is stripped
automatically.

Usage:
    python ci/finalize_release_notes.py <tag> [--dry-run]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

RELEASES_MD = Path(__file__).resolve().parent.parent / "docs" / "releases.md"

_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def normalize_tag(tag: str) -> str:
    """
    Strip a leading ``v``/``V`` from *tag*, if present.

    Args:
        tag: Raw tag as passed on the command line (e.g. ``"v0.7.0"`` or ``"0.7.0"``).
    """
    if re.match(r"^[vV]\d", tag):
        return tag[1:]
    return tag


def find_unreleased_heading(lines: list[str]) -> int | None:
    """
    Return the line index of the active unreleased ``## `` heading, or
    ``None`` if no such heading exists.

    A heading qualifies when it is exactly ``Unreleased`` (case-insensitive)
    or contains ``(unreleased)`` (e.g. ``vX.Y.Z (unreleased)``).

    Args:
        lines: Lines of ``releases.md``, as from ``str.splitlines(keepends=True)``.
    """
    for i, line in enumerate(lines):
        if line.startswith("###"):
            continue
        m = _H2_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip().lower()
        if title == "unreleased" or "(unreleased)" in title:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize the active unreleased release notes for a tagged release."
    )
    parser.add_argument("tag", help="Release tag to finalize to, e.g. 0.7.0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files.",
    )
    args = parser.parse_args()
    tag = normalize_tag(args.tag)

    text = RELEASES_MD.read_text()
    lines = text.splitlines(keepends=True)

    heading_idx = find_unreleased_heading(lines)
    if heading_idx is None:
        print("No unreleased release notes heading found — nothing to finalize.")
        return

    old_heading = lines[heading_idx].rstrip("\n")
    lines[heading_idx] = f"## {tag}\n"
    print(f"Finalizing '{old_heading}' -> '## {tag}'")

    new_text = "".join(lines)
    if args.dry_run:
        print(f"--- dry-run: would write {RELEASES_MD.relative_to(Path.cwd())} ---")
        print(new_text)
        return

    RELEASES_MD.write_text(new_text)
    print(f"\nFinalized release notes for {tag}.")


if __name__ == "__main__":
    main()
