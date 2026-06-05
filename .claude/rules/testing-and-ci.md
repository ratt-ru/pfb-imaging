# Testing & CI/CD Guidelines

Read this when editing `tests/**/*.py` or `.github/workflows/*.yml` files.

## 1. Test Infrastructure

* Tests are parametrized with `pytest.mark.parametrize`.
* Test data in `tests/data/` is downloaded automatically from Google Drive on first run.
* Session-scoped fixtures in `conftest.py` for efficient data reuse.

### arcae / python-casacore test isolation (CRITICAL)

`tests/test_imager.py` loads the `arcae` `xarray-ms` engine (it exercises `pfb imager`).
Most other tests pull in python-casacore (e.g. via `construct_mappings`/daskms in `ms_meta`).
arcae and python-casacore **cannot coexist in one process** (arcae#72), so **never run
`pytest tests/` as a single command** — it segfaults. Run the suite as two invocations:

```bash
uv run pytest -v tests/test_imager.py                 # arcae only
uv run pytest -v tests/ --ignore=tests/test_imager.py # everything else (casacore)
```

`ci.yml` and `publish.yml` already do exactly this. Guidance for new tests:

* Tests that load **arcae** (run `imager`, open an MS via `xr.open_datatree(engine="xarray-ms:msv2")`)
  must live in `tests/test_imager.py`. To compare against the casacore-based `init`/`grid`, run
  those in a **subprocess** (the `pfb` CLI) so casacore stays out of the arcae process — see the
  equivalence test in `tests/test_imager.py`.
* Tests that load **neither** arcae nor casacore (pure ducc0/numpy/zarr — e.g.
  `tests/test_imager_pass2.py`, `tests/test_hessian_tree.py`, `tests/test_fits_tree.py`) belong
  in the second (non-`test_imager`) group and may freely coexist with casacore tests.
* Do not import `operators/gridder`, `operators/hessian`, `utils/fits` etc. expecting them to be
  casacore-laden — they are deliberately casacore-free (see `.claude/rules/architecture.md` §3/§8).

## 2. Commit Messages

* Use [Conventional Commits](https://www.conventionalcommits.org/) format: `<type>: <description>`
* Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `ci`, `deps`, `chore`
* Keep the first line under 72 characters. Use imperative mood.
* Optional scope: `feat(init): add support for new data column`
* Changelog is auto-generated from these prefixes via git-cliff.

## 3. Mandatory Linting

Always run linting after adding or modifying any code:
`uv run ruff format . && uv run ruff check . --fix`

## 4. CI Workflow and `[skip checks]`

The CI pipeline uses a custom `[skip checks]` tag (not GitHub's `[skip ci]`).
* The `update-cabs` workflow commits with `[skip checks]` after regenerating cab definitions on merge to main.
* Each CI job checks the commit message via `gh api` and sets `SKIP_CHECKS=true` to skip heavy steps while still reporting success for branch protection.

## 5. GitHub Actions Workflows

* **`ci.yml`**: Code quality (ruff) and tests across Python 3.10-3.12. Tests run as two separate
  pytest invocations (`test_imager.py` alone, then the rest with `--ignore=tests/test_imager.py`)
  for the arcae/casacore isolation described in §1.
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
