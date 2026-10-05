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

# Figures that must appear in exactly one file. Regexes, because the same figure
# is written with and without a space before the unit ("~108 s" vs "~108s") and a
# literal-substring check silently misses the other spelling.
VOLATILE = (
    r"\b745\b",
    r"\b786\b",
    r"\b787\b",
    r"~?108\s*s\b",
    r"~?465\s*s\b",
    r"9\.5\s*min",
)
VOLATILE_HOME = ".claude/rules/testing-and-ci.md"
PROSE = (
    "CLAUDE.md",
    ".claude/rules/architecture.md",
    ".claude/rules/python-standards.md",
    ".claude/rules/testing-and-ci.md",
    "pyproject.toml",
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
        # Only treat a file as citing decisions when it refers to the ledger at all;
        # a bare "D1" in unrelated prose is noise.
        if "design-decisions" not in text and "wiki D" not in text:
            continue
        for cited in sorted(set(re.findall(r"\bD([1-9]\d?)\b", text)), key=int):
            if f"D{cited}" not in defined:
                failures.append(f"{path.relative_to(ROOT)} cites undefined D{cited}")
    return failures


def gate_wiki_paths() -> list[str]:
    """Every docs/wiki/*.md path cited anywhere must exist."""
    failures = []
    for path in _files():
        text = path.read_text(errors="ignore")
        for ref in sorted(set(re.findall(r"(?:docs/)?wiki/([a-z0-9-]+\.md)", text))):
            if not (ROOT / "docs/wiki" / ref).exists():
                failures.append(f"{path.relative_to(ROOT)} -> missing docs/wiki/{ref}")
    return failures


def gate_volatile_numbers() -> list[str]:
    """Test counts and timings live in exactly one file."""
    failures = []
    for figure in VOLATILE:
        pattern = re.compile(figure)
        homes = [p for p in PROSE if pattern.search((ROOT / p).read_text(errors="ignore"))]
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


def main() -> int:
    gates = {
        "decision citations resolve": gate_decision_citations,
        "wiki paths exist": gate_wiki_paths,
        "volatile numbers single-homed": gate_volatile_numbers,
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
