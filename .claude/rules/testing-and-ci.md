# Testing & CI/CD Guidelines

Read this when editing `tests/**/*.py` or `.github/workflows/*.yml` files.

## 1. Test Infrastructure

* Tests are parametrized with `pytest.mark.parametrize`.
* Test data in `tests/data/` is downloaded automatically from Google Drive on first run.
* Session-scoped fixtures in `conftest.py` for efficient data reuse.

### arcae / python-casacore coexistence

As of **arcae 0.5.2** (ratt-ru/arcae#211, #212) arcae and python-casacore coexist in one
process. Run the whole suite as a single command:

```bash
uv run pytest -v tests/
```

`tests/test_imager.py` (arcae / `pfb imager`) and the casacore-based tests (which pull in
python-casacore via `construct_mappings`/daskms in `ms_meta`) therefore run together in one
pytest session, sharing the session Ray fixture. New tests need no special placement. The
imaging-path modules (`operators/gridder`, `operators/hessian`, `utils/fits`, …) remain
casacore-free by design (see `.claude/rules/architecture.md` §3/§8) — a lightweight-install
preference, not an isolation requirement.

## 2. Commit Messages

* Use [Conventional Commits](https://www.conventionalcommits.org/) format: `<type>: <description>`
* Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `ci`, `deps`, `chore`
* Keep the first line under 72 characters. Use imperative mood.
* Optional scope: `feat(init): add support for new data column`
* Changelog is auto-generated from these prefixes via git-cliff.

## 3. Mandatory Linting

Always run linting after adding or modifying any code:
`uv run ruff format . && uv run ruff check . --fix`

### 3.1 Cab Sync Before Committing CLI Changes

The pre-commit `generate-cabs` hook needs `hip-cargo` on PATH and may fail in
environments where it isn't. Regenerate manually and confirm a clean diff instead:
`uv run hip-cargo generate-cabs --module 'src/pfb_imaging/cli/*.py' --output-dir src/pfb_imaging/cabs`
(a stale cab fails CI via `tests/test_roundtrip.py`; see python-standards §2.1 for the
help-text formatting constraints that test imposes).

## 4. CI Workflow and `[skip checks]`

The CI pipeline uses a custom `[skip checks]` tag (not GitHub's `[skip ci]`).
* The `update-cabs` workflow commits with `[skip checks]` after regenerating cab definitions on merge to main.
* Each CI job checks the commit message via `gh api` and sets `SKIP_CHECKS=true` to skip heavy steps while still reporting success for branch protection.

## 5. GitHub Actions Workflows

* **`ci.yml`**: Code quality (ruff) and tests across Python 3.11-3.13. The whole suite runs as a
  single `pytest tests/` invocation (arcae and python-casacore coexist as of arcae 0.5.2; see §1).
* **`publish.yml`**: PyPI publishing on version tags. Runs quality + tests before publishing.
* **`publish-container.yml`**: Build and push container images to GHCR.
* **`update-cabs.yml`**: Regenerate cab definitions on push to `main`. Uses `landman-ci-bot` GitHub App for auth.

## 6. Releases

```bash
tbump <new_version>
```

This generates a changelog (git-cliff), updates version strings and `_container_image.py` tag, regenerates cabs, creates a git tag, and triggers publish workflows.

## 7. Contributing Workflow

1. Create a feature branch: `git checkout -b your-feature-name`
2. Update `CONTAINER_IMAGE` tag in `src/pfb_imaging/_container_image.py` to match your branch name.
3. Make changes and ensure tests pass.
4. Commit using conventional commit messages.
5. Push and create a pull request.

The `update-cabs` workflow resets the tag to `latest` automatically on merge to `main`.
