# Domain Docs

How the engineering skills should consume this repo's domain documentation. Layout:
**single-context**. This repo has no `docs/adr/`. The equivalents live in the LLM wiki
(`docs/wiki/`, Open Knowledge Format v0.1).

## Before exploring, read these

- **`docs/wiki/index.md`**, then the page it routes you to for the area you're working in.
- **`docs/wiki/design-decisions.md`**: the D-numbered decision ledger (this repo's ADRs).
  Read the entries that touch your area. `CLAUDE.md`'s "Where things are documented" table
  says which.
- **the wiki glossary** (a `glossary.md` page in `docs/wiki/`), if it exists.

If the glossary doesn't exist, **proceed silently**. The `/domain-modeling` skill creates it lazily.

## Where skills write

- **"Write an ADR"** → append `### D<next> — <title>` under `## Decisions` in
  `docs/wiki/design-decisions.md` with **Context / Decision / Rationale / Consequences / Source**.
  **Never renumber**: a retired decision becomes a tombstone (`### D<n> — (Retired <date>) ...`)
  that keeps its number.
- **"Update GLOSSARY.md"** → a `glossary.md` page in `docs/wiki/`, created on first need with OKF frontmatter
  (`type`, `title`, `description`, `tags`, `timestamp`, `last_verified_commit`) and a row in the
  `docs/wiki/index.md` table.
- After touching any wiki page, update its `timestamp` and `last_verified_commit` (only after you
  have read the page) and run `uv run python scripts/check_docs.py`.
- Sources cite code, tests, PRs, commits and issues. **Never** cite spec or plan paths
  (`docs/superpowers/` is gitignored and ephemeral).

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as the wiki defines it. Don't drift to synonyms.

If the concept you need isn't defined yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag decision conflicts

If your output contradicts a ledger entry, surface it explicitly rather than silently overriding:

> _Contradicts D22 (stored `BEAM` is `B/n`), but worth reopening because…_
