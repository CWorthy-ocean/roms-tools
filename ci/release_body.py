#!/usr/bin/env python3
"""
Print the Markdown release-notes body for a finalized release *tag*.

Slices the ``## <tag>`` section out of ``docs/releases.md`` (from that heading
down to the next ``## `` heading), drops the heading line itself — the GitHub
release title already shows the tag — and prints the remainder to stdout, ready
for ``gh release create --notes-file``. The content is already GitHub-flavored
Markdown, so no conversion is needed.

With ``--check-only`` nothing is printed; instead the tag is validated as the
newest real release section in the file (any leading ``Unreleased`` block is
ignored), exiting non-zero on mismatch. This guards the publish workflow
against a retitled PR or a finalize step that never ran.

Release tags are stored without a leading ``v`` (e.g. ``0.5.0``); a ``v``-prefix
on the argument or in a heading is stripped before comparison.

Usage:
    python ci/release_body.py <tag> [--check-only]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RELEASES_MD = Path(__file__).resolve().parent.parent / "docs" / "releases.md"

_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def normalize_tag(tag: str) -> str:
    """Strip a leading ``v``/``V`` from *tag* if present (``v0.5.0`` -> ``0.5.0``)."""
    return tag[1:] if re.match(r"^[vV]\d", tag) else tag


def _is_unreleased(title: str) -> bool:
    """Return True if a ``## `` heading title marks an in-development block."""
    low = title.strip().lower()
    return low == "unreleased" or "(unreleased)" in low


def iter_sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """
    Return ``(title, start_idx, end_idx)`` for every ``## `` section in *lines*.

    ``start_idx`` is the heading line; ``end_idx`` is the next ``## `` heading
    or end of file. A ``## `` inside a fenced code block (```` ``` ````) is a
    code sample, not a heading, and is ignored.
    """
    heads: list[tuple[int, str]] = []
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _H2_RE.match(ln)
        if m:
            heads.append((i, m.group(1).strip()))
    out: list[tuple[str, int, int]] = []
    for n, (i, title) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        out.append((title, i, end))
    return out


def section_body(tag: str) -> str:
    """
    Return the Markdown body of the ``## <tag>`` section, heading excluded.

    Raises ``SystemExit`` if no matching section exists.
    """
    tag = normalize_tag(tag)
    lines = RELEASES_MD.read_text().splitlines(keepends=True)
    for title, start, end in iter_sections(lines):
        if normalize_tag(title) == tag:
            body = "".join(lines[start + 1 : end]).strip("\n")
            if not body.strip():
                sys.exit(f"Release section '## {tag}' in {RELEASES_MD} is empty.")
            return body + "\n"
    sys.exit(f"No release section '## {tag}' found in {RELEASES_MD}")


def check_newest(tag: str) -> None:
    """
    Validate that *tag* is the newest real (non-``Unreleased``) release section.

    Raises ``SystemExit`` if the file has no release sections, or the newest
    real section is some other version.
    """
    tag = normalize_tag(tag)
    lines = RELEASES_MD.read_text().splitlines(keepends=True)
    real = [t for t, _, _ in iter_sections(lines) if not _is_unreleased(t)]
    if not real:
        sys.exit(f"No release sections found in {RELEASES_MD}")
    newest = normalize_tag(real[0])
    if newest != tag:
        sys.exit(
            f"Tag '{tag}' is not the newest release section in "
            f"{RELEASES_MD.name} (top of file is '{newest}')."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit release-notes body for a tag.")
    parser.add_argument("tag", help="Release tag, e.g. 0.5.0")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the tag is the newest release section; print nothing.",
    )
    args = parser.parse_args()

    if args.check_only:
        check_newest(args.tag)
        return
    sys.stdout.write(section_body(args.tag))


if __name__ == "__main__":
    main()
