#!/usr/bin/env python3
"""Consistency gates for the LLM wiki and the agent instruction files.

Run: uv run python scripts/check_docs.py
Exits 1 and names every failure; exits 0 when all gates pass.

These gates make the wiki's invariants enforceable rather than aspirational:
decision numbers are cited from source and rules files, so a renumbering or a
deletion must not silently orphan them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs/wiki/design-decisions.md"

SEARCH_GLOBS = (
    "src/**/*.py",
    "src/**/*.md",
    "tests/**/*.py",
    "scripts/**/*.py",
    "docs/**/*.md",
    ".claude/**/*.md",
    "CLAUDE.md",
)

# Timings that must appear in exactly one file. Regexes, because the same figure
# is written with and without a space before the unit ("~108 s" vs "~108s") and a
# literal-substring check silently misses the other spelling.
#
# Test COUNTS are deliberately absent here: they are measured by
# gate_test_counts() instead of pinned, because every commit that adds a test
# invalidated a hardcoded list and the single-homing gate could not tell a stale
# figure from a current one.
VOLATILE = (
    r"~?115\s*s\b",
    r"~?465\s*s\b",
    r"9\.6\s*min",
)
VOLATILE_HOME = ".claude/rules/testing-and-ci.md"
# Searched repo-wide, not over a fixed list: a figure reintroduced into a wiki page, a
# workflow or a docstring drifts just as silently as one left in CLAUDE.md.
VOLATILE_GLOBS = (
    "CLAUDE.md",
    ".claude/**/*.md",
    "docs/**/*.md",
    "pyproject.toml",
    ".github/workflows/*.yml",
    "tests/**/*.py",
)


def _files() -> list[Path]:
    seen: list[Path] = []
    for pattern in SEARCH_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and "superpowers" not in path.parts:
                seen.append(path)
    return seen


def gate_decision_citations() -> list[str]:
    """Every D<n> cited anywhere must be defined in the ledger."""
    defined = set(re.findall(r"^### (D\d+)", LEDGER.read_text(), re.M))
    failures = []
    for path in _files():
        if path == LEDGER:
            continue
        text = path.read_text(errors="ignore")
        # Match a decision citation in any of the forms actually used in this repo:
        # "(D23)", "wiki D38", "design-decisions.md D1", "D19/D20", "see D22.".
        # A bare "D1" in unrelated prose would be noise, so require a cue: either the
        # file names the ledger, or the token is bracketed//-joined/preceded by a cue word.
        names_ledger = "design-decisions" in text or "wiki D" in text
        cited = set()
        if names_ledger:
            cited |= set(re.findall(r"\bD([1-9]\d?)\b", text))
        cited |= set(re.findall(r"[(\[]D([1-9]\d?)[)\],;/]", text))
        cited |= set(re.findall(r"(?:wiki|see|per|decision)\s+D([1-9]\d?)\b", text, re.I))
        cited |= set(re.findall(r"\bD[1-9]\d?/D([1-9]\d?)\b", text))
        for n in sorted(cited, key=int):
            if f"D{n}" not in defined:
                failures.append(f"{path.relative_to(ROOT)} cites undefined D{n}")
    return failures


def gate_wiki_paths() -> list[str]:
    """Every docs/wiki/*.md path cited anywhere must exist."""
    failures = []
    for path in _files():
        text = path.read_text(errors="ignore")
        refs = set(re.findall(r"(?:docs/)?wiki/([a-z0-9_-]+\.md)", text))
        if path.parent == ROOT / "docs/wiki":
            # Inside the wiki, links are written relatively: [primer](deconv-primer.md).
            refs |= set(re.findall(r"\]\(([a-z0-9_-]+\.md)\)", text))
        for ref in sorted(refs):
            if not (ROOT / "docs/wiki" / ref).exists():
                failures.append(f"{path.relative_to(ROOT)} -> missing docs/wiki/{ref}")
    return failures


def gate_volatile_numbers() -> list[str]:
    """Test counts and timings live in exactly one file."""
    failures = []
    candidates = []
    for pattern in VOLATILE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and "superpowers" not in path.parts:
                candidates.append(path)
    for figure in VOLATILE:
        rx = re.compile(figure)
        homes = sorted(str(p.relative_to(ROOT)) for p in candidates if rx.search(p.read_text(errors="ignore")))
        if homes != [VOLATILE_HOME]:
            failures.append(f"{figure!r} in {homes or ['nowhere']}, want ['{VOLATILE_HOME}']")
    return failures


def gate_frontmatter() -> list[str]:
    """Every wiki page carries the OKF fields the maintenance rule depends on."""
    failures = []
    for path in sorted((ROOT / "docs/wiki").glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text()
        head = text.split("---")[1] if text.startswith("---") else ""
        for field in ("timestamp:", "last_verified_commit:"):
            if field not in head:
                failures.append(f"docs/wiki/{path.name} missing {field}")
    return failures


def _collect(marker: str) -> int:
    """Number of tests pytest collects under `-m <marker>`."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker, "tests"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    # The summary reads "N tests collected" or "N/M tests collected (K deselected)".
    # Not anchored to line start: pyproject's addopts carries --verbose, so pytest
    # wraps the summary in "=" padding and -q does not suppress it.
    match = re.search(r"(\d+)(?:/\d+)? tests collected", proc.stdout)
    if not match:
        raise RuntimeError(f"could not parse pytest collection output:\n{proc.stdout[-2000:]}")
    return int(match.group(1))


def gate_test_counts() -> list[str]:
    """Test counts in the rules file match what pytest actually collects.

    Counts used to be pinned in VOLATILE and single-homed. That caught a figure
    copied into a second file but not a figure that had simply gone stale, so
    every commit adding a test left the rules file quietly wrong while the gate
    stayed green. Measuring instead makes the gate self-correcting: it fails
    with the number to paste in.

    Only *collected* counts are checked, never pass/skip counts -- how many tests
    skip depends on which extras are installed, so a pinned pass count could not
    be true on every CI leg at once.

    Counts are not cross-file single-homed the way the timings are. A collected
    count is often a small, unremarkable integer (the slow count is 44, which is
    also a grid size in tests/test_weighting.py and a figure in two wiki pages),
    so "appears in exactly one file" produces false positives it cannot
    distinguish from real duplication. Staleness was the actual problem, and
    measurement fixes that at the source.
    """
    failures = []
    text = (ROOT / VOLATILE_HOME).read_text()
    measured = {
        "full": _collect(""),
        "slow": _collect("slow"),
        "fast": _collect("not slow"),
    }
    for label, count in measured.items():
        if not re.search(rf"\b{count}\b", text):
            failures.append(
                f"{VOLATILE_HOME} does not state the measured {label} count {count} "
                f"(suite changed -- update the figure there)"
            )
    return failures


def main() -> int:
    gates = {
        "decision citations resolve": gate_decision_citations,
        "wiki paths exist": gate_wiki_paths,
        "volatile timings single-homed": gate_volatile_numbers,
        "test counts match reality": gate_test_counts,
        "wiki frontmatter complete": gate_frontmatter,
    }
    bad = 0
    for name, gate in gates.items():
        failures = gate()
        if failures:
            bad += 1
            print(f"FAIL  {name}")
            for failure in failures:
                print(f"        {failure}")
        else:
            print(f"ok    {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
