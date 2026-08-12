#!/usr/bin/env python3
"""
Update release notes from merged GitHub pull requests.

Fetches all PRs merged since the last tagged release on
https://github.com/CWorthy-ocean/roms-tools and inserts their
categorised notes into the in-development ("Unreleased") section of
``docs/releases.md``.

To protect manually-curated release notes, this script is a no-op until a
stable release tag of at least ``GUARD_VERSION`` exists.  Automated notes only
begin accumulating once that release has been cut, so the last hand-written
release is never modified.

Usage:
    python ci/update_release_notes.py [--dry-run]

Environment:
    GITHUB_TOKEN: Optional personal access token for higher API rate limits
                  (unauthenticated requests are limited to 60/hour).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests
from packaging.version import Version

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_OWNER = "CWorthy-ocean"
REPO_NAME = "roms-tools"
REPO = f"{REPO_OWNER}/{REPO_NAME}"
GITHUB_API = "https://api.github.com"
RELEASES_MD = Path(__file__).resolve().parent.parent / "docs" / "releases.md"
PR_URL_BASE = f"https://github.com/{REPO}/pull"

# Automated notes only start once a stable release >= this version exists,
# leaving the last manually-curated release (and everything before it) untouched.
GUARD_VERSION = "4.0.0"

# PR template section heading  →  release-notes section heading.
# roms-tools uses the PR-template section names verbatim for all notes going
# forward, so this is an identity map.
SECTION_MAP: dict[str, str] = {
    "Breaking Changes": "Breaking Changes",
    "New Features": "New Features",
    "Bug Fixes": "Bug Fixes",
    "Improvements": "Improvements",
    "Miscellaneous": "Miscellaneous",
}

# PR template sections to skip entirely.
# Exact names are matched case-insensitively; any section whose name contains
# "checklist" is also skipped, catching variants like "Review Checklist".
_SKIP_SECTION_EXACT = frozenset({"summary", "code review checklist"})
_SKIP_SECTION_SUBSTRINGS = ("checklist",)


def _should_skip_section(name: str) -> bool:
    """
    Return True if a PR section with *name* should be excluded from release notes.

    Args:
        name: The section heading text (case-insensitive matching is applied).
    """
    lower = name.lower()
    return lower in _SKIP_SECTION_EXACT or any(
        s in lower for s in _SKIP_SECTION_SUBSTRINGS
    )


# Note texts that mean "nothing to report" rather than a real release note.
# Authors write these in place of, or instead of, the template's "- N/A".
_PLACEHOLDER_TEXTS = frozenset(
    {"n/a", "na", "none", "nothing", "no", "no change", "no changes"}
)


# Sections eligible for the un-bulleted-prose fallback in :func:`parse_pr_body`
# — only the categories we actually publish.
_RELEASE_NOTE_SECTIONS = frozenset(s.lower() for s in SECTION_MAP)

# Prose lines that are markup rather than a note: standalone HTML (typically
# an embedded screenshot) and Markdown table rows.
_NON_PROSE_LINE_RE = re.compile(r"^(?:<[^>]+>\s*$|\|)")


def _is_placeholder(text: str) -> bool:
    """
    Return True if *text* is an "empty" placeholder rather than a real note.

    Args:
        text: A note's text, with any leading bullet marker already removed.
    """
    return text.strip().rstrip(".").strip().lower() in _PLACEHOLDER_TEXTS


# Matches a PR link appended by this script: ([#NNN](url))
_LINK_SUFFIX_RE = re.compile(r"\s*\(\[#\d+\]\([^)]*\)\)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Version sorting
# ---------------------------------------------------------------------------


def _parse_version(s: str) -> Version:
    """
    Parse a version string like ``v0.5.0`` or ``v0.5.0-alpha`` into a
    ``packaging.version.Version``.

    Dash-separated pre-release labels (``-alpha``, ``-beta``, ``-rc``) are
    normalised to PEP 440 equivalents (``a0``, ``b0``, ``rc0``) so that
    ``packaging`` can order them correctly relative to the final release.

    Args:
        s: Version string to parse, with or without a leading ``v``
           (e.g. ``"v0.5.0"``, ``"0.5.0-alpha"``).
    """
    s = s.lstrip("v")
    s = re.sub(r"-alpha(\d*)$", lambda m: f"a{m.group(1) or '0'}", s)
    s = re.sub(r"-beta(\d*)$", lambda m: f"b{m.group(1) or '0'}", s)
    s = re.sub(r"-rc(\d*)$", lambda m: f"rc{m.group(1) or '0'}", s)
    return Version(s)


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    """Build a ``requests.Session`` with GitHub API headers."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        print(
            "Warning: GITHUB_TOKEN not set — unauthenticated requests are "
            "limited to 60/hour.",
            file=sys.stderr,
        )
    return session


def get_latest_stable_tag(session: requests.Session) -> tuple[str, str] | None:
    """
    Return *(tag_name, ISO-8601 commit date)* for the highest-versioned stable
    tag, or ``None`` if the repository has no stable semver tags.

    GitHub's tags API returns tags in commit-date order, which can surface
    old pre-release tags ahead of newer stable releases.  We fetch all tags,
    sort them by semantic version (reusing ``_parse_version``), and pick the
    highest one instead.

    Args:
        session: Authenticated ``requests.Session`` with GitHub API headers set.
    """
    versioned: list[tuple] = []
    page = 1
    while True:
        resp = session.get(
            f"{GITHUB_API}/repos/{REPO}/tags",
            params={"per_page": 100, "page": page},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for tag in batch:
            try:
                ver = _parse_version(tag["name"])
            except (ValueError, TypeError):
                continue  # skip non-semver tags
            # Skip pre-release tags (e.g. -alpha, 4.0.0a2, -rc); only stable
            # releases mark the boundary for collecting PR notes.
            if ver.is_prerelease:
                continue
            versioned.append((ver, tag))
        if len(batch) < 100:
            break
        page += 1

    if not versioned:
        return None

    versioned.sort(key=lambda x: x[0])
    latest = versioned[-1][1]
    tag_name: str = latest["name"]

    # Resolve the tag to its commit to get the commit date
    sha = latest["commit"]["sha"]
    resp = session.get(f"{GITHUB_API}/repos/{REPO}/commits/{sha}")
    resp.raise_for_status()
    date: str = resp.json()["commit"]["committer"]["date"]
    return tag_name, date


def get_merged_prs_since(session: requests.Session, since: str) -> list[dict]:
    """
    Return all PRs merged into ``main`` after *since* (ISO-8601 string),
    sorted by PR number ascending.

    Args:
        session: Authenticated ``requests.Session`` with GitHub API headers set.
        since: ISO-8601 timestamp; only PRs merged strictly after this date are returned.
    """
    results: list[dict] = []
    page = 1
    while True:
        params: dict[str, str | int] = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "base": "main",
            "per_page": 100,
            "page": page,
        }
        resp = session.get(f"{GITHUB_API}/repos/{REPO}/pulls", params=params)
        resp.raise_for_status()
        batch: list[dict] = resp.json()
        if not batch:
            break

        any_newer = False
        for pr in batch:
            merged_at: str | None = pr.get("merged_at")
            if merged_at and merged_at > since:
                results.append(pr)
                any_newer = True

        # If the oldest updated_at on this page is before *since* and we
        # found nothing newer, no further pages will have relevant PRs.
        if not any_newer and batch[-1]["updated_at"] <= since:
            break

        if len(batch) < 100:
            break
        page += 1

    return sorted(results, key=lambda p: p["number"])


# ---------------------------------------------------------------------------
# PR body parsing
# ---------------------------------------------------------------------------


def parse_pr_body(
    body: str | None,
) -> dict[str, list[tuple[str, list[str]]]]:
    """
    Parse a PR description into ``{section_name: [(text, [sub_item, …]), …]}``.

    Each top-level bullet is stored as a ``(text, sub_items)`` tuple where
    *sub_items* contains any indented child bullets directly beneath it.

    - Recognises both ``#`` and ``##`` level Markdown headings as section
      boundaries so that non-standard headers (e.g. ``# Review Checklist``)
      are handled correctly.
    - Skips the Summary and any section whose name contains "checklist".
    - Drops notes that are "nothing to report" placeholders — ``N/A``,
      ``None``, ``Nothing`` etc. (see :func:`_is_placeholder`).
    - Drops checklist-style bullets (``- [ ]`` / ``- [x]``) even if they
      appear under a content section, as a belt-and-suspenders guard.
    - Strips inline HTML comments (``<!-- … -->``) before processing, and
      ignores fenced code blocks entirely.
    - Falls back to prose when a section has **no** top-level bullets: some
      authors delete the template's ``- `` markers and just write sentences.
      Each blank-line-separated paragraph then becomes one note, with its
      lines joined by spaces.  The fallback is per-section and only applies
      when that section is bullet-free, so a section mixing an introductory
      paragraph with bullets still yields exactly its bullets, as before.

    Args:
        body: Raw Markdown text of the PR description, or ``None`` if the PR has no body.
    """
    if not body:
        return {}

    # Strip multiline HTML comments before line-by-line processing so that
    # comments spanning several lines don't leave orphaned <!-- or --> fragments.
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    result: dict[str, list[tuple[str, list[str]]]] = {}
    current: str | None = None
    # Each element: [top_level_text, [sub_item, …]] — mutable for sub-item appends
    items: list[list] = []
    # Un-bulleted prose in the current section, grouped into paragraphs
    paragraphs: list[list[str]] = []
    in_fence = False

    def _flush() -> None:
        if not current or _should_skip_section(current):
            return
        collected = items or _prose_fallback(current, paragraphs)
        good = [(t, s) for t, s in collected if not _is_placeholder(t)]
        if good:
            result[current] = good

    for raw in body.splitlines():
        cleaned = raw.rstrip()
        indent = len(cleaned) - len(cleaned.lstrip())
        line = cleaned.strip()

        # Fenced code blocks are never release-note content
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Match # or ## (and ###) level headings
        heading = re.match(r"^#{1,3}\s+(.+)$", line)
        if heading:
            _flush()
            current = heading.group(1).strip()
            items = []
            paragraphs = []
            continue

        if current is None or _should_skip_section(current):
            continue

        # Skip checklist-style bullets regardless of which section they appear in
        if re.match(r"^[-*]\s+\[[ xX]\]", line):
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            text = bullet.group(1).strip()
            if not text:
                continue
            if indent == 0:
                items.append([text, []])
            elif items:
                # Attach as a sub-bullet of the most recent top-level item
                items[-1][1].append(text)
            continue

        # Un-bulleted prose: buffer it in case this section has no bullets
        # at all, in which case _flush() promotes each paragraph to a note.
        if not line:
            if paragraphs and paragraphs[-1]:
                paragraphs.append([])
        else:
            if not paragraphs:
                paragraphs.append([])
            paragraphs[-1].append(line)

    _flush()
    return result


def _prose_fallback(section: str, paragraphs: list[list[str]]) -> list[list]:
    """
    Turn buffered prose paragraphs into ``[text, sub_items]`` note entries.

    Only the template's own categories are eligible.  PR authors routinely
    invent narrative headings ("Setup", "Testing", "Comparison figures", …)
    and write paragraphs, screenshots and tables under them; those keep the
    strict bullets-only behaviour so none of it is ever scraped.

    Within an eligible section, lines that are standalone HTML (``<img …>``)
    or Markdown table rows are dropped, as are paragraphs left with no
    alphanumeric content (horizontal rules such as ``---`` or ``***``).

    Args:
        section: Heading the paragraphs were collected under.
        paragraphs: Blank-line-separated groups of stripped prose lines.
    """
    if section.lower() not in _RELEASE_NOTE_SECTIONS:
        return []
    items: list[list] = []
    for para in paragraphs:
        text = " ".join(p for p in para if not _NON_PROSE_LINE_RE.match(p)).strip()
        if text and re.search(r"[A-Za-z0-9]", text):
            items.append([text, []])
    return items


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """
    Lowercase and collapse whitespace; also strips any appended PR link so
    that duplicate detection is link-agnostic.

    Args:
        text: A bullet's text, possibly including a trailing PR link suffix.
    """
    stripped = _LINK_SUFFIX_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", stripped).lower()


def get_known_pr_numbers(md_text: str) -> set[int]:
    """
    PR numbers already linked anywhere in *md_text*.

    Args:
        md_text: Full text of the ``releases.md`` file.
    """
    return {int(n) for n in re.findall(r"/pull/(\d+)", md_text)}


def _h2_positions(lines: list[str]) -> list[tuple[str, int]]:
    """Return ``[(title, line_index), …]`` for every ``## `` heading, in order."""
    positions: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and not line.startswith("###"):
            positions.append((m.group(1).strip(), i))
    return positions


def _h2_content_bounds(lines: list[str], title_idx: int) -> tuple[int, int]:
    """
    Return *(content_start, content_end)* line indices for the ``## `` block
    whose heading is at *title_idx* (content starts on the line after the heading
    and runs until the next ``## `` heading or end of file).
    """
    content_start = title_idx + 1
    content_end = len(lines)
    for _, idx in _h2_positions(lines):
        if title_idx < idx < content_end:
            content_end = idx
    return content_start, content_end


def find_unreleased_block(lines: list[str], latest_tag: str) -> tuple[list[str], str]:
    """
    Return *(lines, block_title)* for the in-development release block,
    creating a fresh ``## Unreleased`` block at the top if none is suitable.

    A block qualifies as the automated-notes target when its heading is exactly
    ``Unreleased`` (case-insensitive) or is a ``vX.Y.Z (unreleased)`` header
    whose version is strictly newer than *latest_tag*.  The version check keeps
    us from writing into a ``(unreleased)`` section that has already been tagged
    (e.g. a stale header the maintainer forgot to rename after release).

    Args:
        lines: Lines of ``releases.md`` (as from ``splitlines(keepends=True)``).
        latest_tag: The highest stable release tag name (e.g. ``"v4.0.0"``).
    """
    latest_ver = _parse_version(latest_tag)
    for title, _ in _h2_positions(lines):
        low = title.lower()
        if low == "unreleased":
            return lines, title
        if "(unreleased)" in low:
            ver_str = re.sub(r"\s*\(unreleased\)\s*", "", title, flags=re.IGNORECASE)
            try:
                if _parse_version(ver_str) > latest_ver:
                    return lines, title
            except (ValueError, TypeError):
                # Unparseable version in a "(unreleased)" header — treat as target.
                return lines, title

    # No suitable block — create "## Unreleased" with the standard sections.
    block = _unreleased_block_lines()

    # Insert after the "# Release notes" title (before the first "## " heading);
    # if neither exists, prepend to the file.
    insert_at = 0
    positions = _h2_positions(lines)
    if positions:
        insert_at = positions[0][1]
    else:
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = i + 1
                break
    new_lines = lines[:insert_at] + block + lines[insert_at:]
    return new_lines, "Unreleased"


def _unreleased_block_lines() -> list[str]:
    """Return the newline-terminated lines for a fresh ``## Unreleased`` block."""
    parts = ["## Unreleased\n", "\n"]
    for section in SECTION_MAP.values():
        parts.append(f"### {section}\n")
        parts.append("\n")
    return parts


def _section_positions_in_block(
    lines: list[str], block_start: int, block_end: int
) -> dict[str, int]:
    """Return ``{section_title: heading_line_index}`` for ``### `` headings in a block."""
    positions: dict[str, int] = {}
    for i in range(block_start, block_end):
        m = re.match(r"^###\s+(.+?)\s*$", lines[i])
        if m and lines[i].strip() not in positions:
            positions[m.group(1).strip()] = i
    return positions


def get_section_bullets(
    lines: list[str], block_title: str, section_title: str
) -> set[str]:
    """
    Normalised set of bullet texts already present in *section_title* within the
    ``## block_title`` block.
    """
    positions = _h2_positions(lines)
    block_idx = next((idx for t, idx in positions if t == block_title), None)
    if block_idx is None:
        return set()
    block_start, block_end = _h2_content_bounds(lines, block_idx)
    sections = _section_positions_in_block(lines, block_start, block_end)
    if section_title not in sections:
        return set()
    sec_start = sections[section_title] + 1
    sec_end = block_end
    for other_idx in sections.values():
        if sections[section_title] < other_idx < sec_end:
            sec_end = other_idx
    bullets: set[str] = set()
    for line in lines[sec_start:sec_end]:
        m = re.match(r"^[*-]\s+(.+)$", line.rstrip())
        if m:
            bullets.add(_normalize(m.group(1)))
    return bullets


def format_md_bullet(text: str, sub_items: list[str], pr_number: int) -> list[str]:
    """
    Return newline-terminated Markdown lines for one bullet.

    The PR link is appended to the top-level text; sub-items are rendered as a
    two-space-indented nested list.

    Args:
        text: Top-level bullet text (plain prose, no leading marker).
        sub_items: Indented child bullet texts, if any.
        pr_number: GitHub PR number used to build the hyperlink.
    """
    link = f"([#{pr_number}]({PR_URL_BASE}/{pr_number}))"
    out = [f"* {text} {link}\n"]
    out.extend(f"  * {s}\n" for s in sub_items)
    return out


def insert_bullets_into_section(
    lines: list[str],
    block_title: str,
    section_title: str,
    new_bullets: list[list[str]],
) -> list[str]:
    """
    Return a new line list with *new_bullets* appended to *section_title* inside
    the ``## block_title`` block.  If the section is absent, it is created at the
    end of the block.

    *new_bullets* is a list of pre-formatted bullets, each a list of
    newline-terminated lines as returned by ``format_md_bullet``.
    """
    flat: list[str] = [line for bullet in new_bullets for line in bullet]

    positions = _h2_positions(lines)
    block_idx = next((idx for t, idx in positions if t == block_title), None)
    if block_idx is None:
        print(f"  WARNING: block '{block_title}' not found — skipping.")
        return lines
    block_start, block_end = _h2_content_bounds(lines, block_idx)
    sections = _section_positions_in_block(lines, block_start, block_end)

    if section_title not in sections:
        # Create the section at the end of the block (after trailing blanks).
        insert_at = block_end
        while insert_at > block_start and not lines[insert_at - 1].strip():
            insert_at -= 1
        new_section = [f"\n### {section_title}\n", "\n", *flat]
        return lines[:insert_at] + new_section + lines[insert_at:]

    # Section exists: find where its content ends (next "### " or block end),
    # then insert after the last non-blank content line.
    sec_heading = sections[section_title]
    sec_end = block_end
    for other_idx in sections.values():
        if sec_heading < other_idx < sec_end:
            sec_end = other_idx

    insert_at = sec_end
    while insert_at > sec_heading + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    # If the heading is immediately followed by our insertion point (empty
    # section), ensure a blank line separates the heading from the bullets.
    prefix: list[str] = []
    if insert_at == sec_heading + 1:
        prefix = ["\n"]

    return lines[:insert_at] + prefix + flat + lines[insert_at:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the in-development release notes with notes from merged PRs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files.",
    )
    args = parser.parse_args()

    session = make_session()

    # Guard: do nothing until a stable release >= GUARD_VERSION exists, so the
    # last manually-curated release is never touched.
    latest = get_latest_stable_tag(session)
    guard = _parse_version(GUARD_VERSION)
    if latest is None:
        print(
            f"No stable release tags found — automated notes start at "
            f"v{GUARD_VERSION}. Nothing to do."
        )
        return
    tag_name, tag_date = latest
    if _parse_version(tag_name) < guard:
        print(
            f"Last stable release {tag_name} is below the v{GUARD_VERSION} "
            f"activation threshold — automated notes not yet enabled. Nothing to do."
        )
        return

    print(f"Last tagged release : {tag_name}  (commit date {tag_date})")

    if not RELEASES_MD.exists():
        print(f"ERROR: {RELEASES_MD} does not exist.", file=sys.stderr)
        sys.exit(1)

    rst_text = RELEASES_MD.read_text()
    lines = rst_text.splitlines(keepends=True)
    lines, block_title = find_unreleased_block(lines, tag_name)
    print(f"Target section      : ## {block_title}  in {RELEASES_MD.name}")

    # Fetch all PRs merged after that date
    print("Fetching merged PRs…")
    prs = get_merged_prs_since(session, tag_date)
    print(f"Found {len(prs)} PR(s) merged since {tag_name}.\n")

    if not prs:
        print("Nothing to do.")
        return

    known_prs = get_known_pr_numbers("".join(lines))
    total_added = 0

    for pr in prs:
        pr_num: int = pr["number"]
        pr_title: str = pr["title"]

        if pr_num in known_prs:
            print(f"  #{pr_num:5d} already in release notes — skipping.")
            continue

        body_sections = parse_pr_body(pr.get("body"))
        if not body_sections:
            print(f"  #{pr_num:5d} '{pr_title}' — no categorised notes found.")
            continue

        pr_added = 0
        for pr_section, md_section in SECTION_MAP.items():
            items = body_sections.get(pr_section, [])
            if not items:
                continue

            existing = get_section_bullets(lines, block_title, md_section)
            to_add: list[list[str]] = []
            for item_text, sub_items in items:
                if _normalize(item_text) in existing:
                    print(
                        f"  #{pr_num:5d} [{pr_section}] duplicate — '{item_text[:70]}'"
                    )
                    continue
                to_add.append(format_md_bullet(item_text, sub_items, pr_num))

            if to_add:
                lines = insert_bullets_into_section(
                    lines, block_title, md_section, to_add
                )
                pr_added += len(to_add)
                for b in to_add:
                    print(f"  #{pr_num:5d} + [{md_section}] {b[0].rstrip()[:90]}")

        if pr_added:
            total_added += pr_added
            known_prs.add(pr_num)

    if total_added == 0:
        print("\nNo new release notes to add.")
        return

    new_text = "".join(lines)
    if args.dry_run:
        print(
            f"\n--- dry-run: {total_added} bullet(s) would be added to {RELEASES_MD.name} ---"
        )
        print(new_text)
    else:
        RELEASES_MD.write_text(new_text)
        print(f"\nAdded {total_added} bullet(s) to {RELEASES_MD}.")


if __name__ == "__main__":
    main()
