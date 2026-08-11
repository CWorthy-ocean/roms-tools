#!/usr/bin/env python3
"""
Finalize the in-development release notes for a tagged release.

Locates the active "unreleased" ``## `` block in ``docs/releases.md``
(identified by its heading — exactly ``Unreleased``, or ``vX.Y.Z
(unreleased)`` / ``X.Y.Z (unreleased)`` — not by any external state),
rewrites that heading to ``## <tag>``, and drops any ``### `` category
within that block that collected no notes this cycle (either empty or
holding nothing but an ``N/A``-style placeholder).

Only the block being finalized is touched — already-released blocks keep
whatever empty categories they were published with.

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
_H3_RE = re.compile(r"^###\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^[*-]\s+(.+?)\s*$")

# Bullet texts that are placeholders rather than real release notes.  A
# category holding only these is treated as having collected nothing.
_PLACEHOLDER_TEXTS = frozenset(
    {"n/a", "na", "none", "nothing", "no", "no change", "no changes"}
)


def _is_placeholder(text: str) -> bool:
    """
    Return True if *text* is an "empty" placeholder rather than a real note.

    Args:
        text: A bullet's text, with the leading ``*``/``-`` marker removed.
    """
    return text.strip().rstrip(".").strip().lower() in _PLACEHOLDER_TEXTS


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


def _block_bounds(lines: list[str], heading_idx: int) -> tuple[int, int]:
    """
    Return *(content_start, content_end)* line indices for the ``## `` block
    whose heading sits at *heading_idx* — content runs from the line after the
    heading up to the next ``## `` heading, or end of file.

    Args:
        lines: Lines of ``releases.md``.
        heading_idx: Line index of the block's ``## `` heading.
    """
    for i in range(heading_idx + 1, len(lines)):
        if _H2_RE.match(lines[i]) and not lines[i].startswith("###"):
            return heading_idx + 1, i
    return heading_idx + 1, len(lines)


def _section_is_empty(content: list[str]) -> bool:
    """
    Return True if *content* holds no real notes.

    Blank lines are ignored and placeholder bullets (``- N/A`` and friends)
    do not count as content.  Any other non-blank line — a real bullet, but
    also hand-written prose or a directive — counts, so we never drop
    something a maintainer wrote by hand.

    Args:
        content: The lines belonging to one ``### `` category, excluding
            its heading line.
    """
    for line in content:
        stripped = line.strip()
        if not stripped:
            continue
        m = _BULLET_RE.match(stripped)
        if m and _is_placeholder(m.group(1)):
            continue
        return False
    return True


def _splice_out(lines: list[str], drop: set[int]) -> list[str]:
    """
    Return *lines* with the indices in *drop* removed, re-inserting a blank
    line wherever a deletion left two non-blank lines flush against each other
    (dropping a block's last category takes with it the blank line that
    separated it from the next ``## `` heading).

    Only the junctions a deletion actually created are repaired, so untouched
    parts of the file — including any ``#`` that is not a heading — are
    reproduced byte for byte.

    Args:
        lines: Lines of ``releases.md``.
        drop: Line indices to remove.
    """
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        if i - 1 in drop and out and out[-1].strip() and line.strip():
            out.append("\n")
        out.append(line)
    return out


def drop_empty_sections(
    lines: list[str], heading_idx: int
) -> tuple[list[str], list[str]]:
    """
    Return *(new_lines, dropped_titles)* with every ``### `` category that
    collected no notes removed from the ``## `` block at *heading_idx*.

    Args:
        lines: Lines of ``releases.md``.
        heading_idx: Line index of the ``## `` heading being finalized.
    """
    block_start, block_end = _block_bounds(lines, heading_idx)
    sections: list[tuple[int, str]] = []
    for i in range(block_start, block_end):
        m = _H3_RE.match(lines[i])
        if m:
            sections.append((i, m.group(1)))

    drop: set[int] = set()
    dropped: list[str] = []
    for n, (start, title) in enumerate(sections):
        end = sections[n + 1][0] if n + 1 < len(sections) else block_end
        if _section_is_empty(lines[start + 1 : end]):
            dropped.append(title)
            drop.update(range(start, end))

    if not drop:
        return lines, []

    return _splice_out(lines, drop), dropped


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

    lines, dropped = drop_empty_sections(lines, heading_idx)
    for title in dropped:
        print(f"  Dropping empty category: ### {title}")

    new_text = "".join(lines)
    if args.dry_run:
        print(f"--- dry-run: would write {RELEASES_MD.relative_to(Path.cwd())} ---")
        print(new_text)
        return

    RELEASES_MD.write_text(new_text)
    print(f"\nFinalized release notes for {tag}.")


if __name__ == "__main__":
    main()
