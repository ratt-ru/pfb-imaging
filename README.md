# pfb-imaging

Radio interferometric imaging suite based on the preconditioned forward-backward algorithm.
The project follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format:
lightweight CLI installation with auto-generated [stimela](https://github.com/caracal-pipeline/stimela) cab definitions and containerised execution.

## Installation

**Lightweight (CLI + cabs only):**

```bash
pip install pfb-imaging
```

This installs the CLI and stimela cab definitions without the full scientific stack.
The cabs can be included in stimela recipes using:

```yaml
_include:
  - (pfb_imaging.cabs)init.yml
```

**Full stack:**

To run the code natively you need to install the full stack using

```bash
pip install "pfb-imaging[full]"
```

For maximum performance install `ducc0` in no-binary mode:

```bash
pip install ducc0 --no-binary ducc0
```

See the [Development](#development) section for instructions on how to set the package up in development mode and make contributions.

## Quick start

The easiest way to use `pfb-imaging` is via the `stimela` recipes given in the [recipes folder](recipes/).
Once the package is installed, a recipe can be queried for its input and output parameters using the `stimela doc` command.
For example, to see the inputs and outputs of the `sara` recipe, simply run

```bash
stimela doc 'pfb_imaging.recipes::sara.yaml'
```

The recipe can then be run with the `stimela run` command:

```bash
stimela run 'pfb_imaging.recipes::sara.yaml' sara \
  ms=path/to/data.ms \
  base-dir=path/to/base/output/directory \
  image-name=saraout
```

The recipe should contain sensible defaults for MeerKAT data at L-band.

## CLI documentation

The CLI is built with [Typer](https://typer.tiangolo.com/) and provides rich, auto-generated documentation.
To list all available commands:

```bash
pfb --help
```

To get detailed documentation for a specific command including all parameters, types, and defaults:

```bash
pfb init --help
```

This is often more useful than `stimela doc` as it shows the full parameter documentation with types and defaults directly in the terminal.

## CLI commands

The processing pipeline follows a modular pattern where each step is a separate command:

1. `pfb init` -- Parse measurement sets into xarray datasets
2. `pfb grid` -- Create dirty images, PSFs, and weights
3. `pfb kclean` -- Classical deconvolution (Hogbom/Clark)
4. `pfb sara` -- Advanced deconvolution with sparsity constraints
5. `pfb restore` -- Restore clean components to final image
6. `pfb degrid` -- Subtract model from visibilities

Additional commands:

- `pfb deconv` -- General deconvolution (replaces individual algorithm apps)
- `pfb hci` -- High cadence imaging
- `pfb fluxtractor` -- Flux extraction
- `pfb model2comps` -- Convert model to components

## Execution backends

Every command supports a `--backend` option that controls how the command is executed.
This is provided by [hip-cargo](https://github.com/landmanbester/hip-cargo) and enables container fallback execution: when the full scientific stack is not installed locally, commands automatically run inside a container.

Available backends:

- `auto` (default) -- Try native execution first; if the core module import fails (lightweight install), fall back to the best available container runtime.
- `native` -- Run natively using the locally installed Python environment. Fails with `ImportError` if dependencies are missing.
- `docker` -- Run inside a Docker container.
- `podman` -- Run inside a Podman container (daemonless, rootless).
- `apptainer` -- Run inside an Apptainer container (HPC-friendly, formerly Singularity).
- `singularity` -- Run inside a Singularity container.

An additional `--always-pull-images` flag forces re-pulling the container image before execution, useful for ensuring you have the latest version.

Example usage:

```bash
# Run natively (requires full install)
pfb init --ms data.ms --output-filename out --backend native

# Run in a Docker container (lightweight install only)
pfb init --ms data.ms --output-filename out --backend docker

# Auto-detect: native if available, otherwise container
pfb init --ms data.ms --output-filename out
```

Volume mounts are resolved automatically from the command's type hints: input paths are mounted read-only, output paths read-write.
Docker and Podman run as the current user to avoid root-owned output files.

## Default naming conventions

Output files follow consistent naming patterns using `--output-filename`, `--product`, and `--suffix`:

- XDS datasets: `{output_filename}_{product}.xds`
- DDS datasets: `{output_filename}_{product}_{suffix}.dds`
- Models: `{output_filename}_{product}_{suffix}_model.mds`
- FITS files: same convention with appropriate extensions

The `--suffix` parameter (default `main`) allows imaging multiple fields from a single set of corrected Stokes visibilities.
For example, the sun can be imaged by setting `--target sun --suffix sun`.
The `--target` parameter accepts any object recognised by `astropy` or `HH:MM:SS,DD:MM:SS` format.

## Parallelism settings

Two settings control parallelism:

- `--nworkers` controls how many chunks (usually imaging bands) are processed in parallel.
- `--nthreads` specifies threads available to each worker (gridding, FFTs, wavelet transforms).

By default a single worker is used for the smallest memory footprint and easy debugging.
Set `--nworkers` larger than one to use multiple Dask workers for parallel chunk processing.
The product of `--nworkers` and `--nthreads` should not exceed available resources.

## Weighting

Imaging weights control the tradeoff between point-source sensitivity and angular resolution.
`pfb grid` and `pfb hci` expose the same set of options under the **Weighting** help panel:

- `--robustness` -- Briggs robustness factor.
  Leaving this unset (the default) applies natural weighting, which simply uses the visibility weights from the measurement set and maximises point-source sensitivity.
  Setting an explicit value switches to Briggs weighting: `-2` is pure uniform (best resolution, lowest sidelobes, highest thermal noise) and larger values (e.g. `0.5`, `2`) taper back toward natural.
  Briggs weights are computed by binning visibilities onto a padded uv grid and then dividing by a per-cell factor derived from the robustness.
- `--npix-super` -- Super-uniform half-size in pixels.
  When non-zero and `--robustness` is set, each visibility is normalised by the sum of counts in a `(2*npix_super+1)^2` box around its uv-cell instead of a single cell.
  `0` (default) gives standard uniform/Briggs weighting; `1` uses a `3x3` box, which smooths sparse uv-coverage and reduces outer sidelobes.
  Combined with a non-default `--robustness`, this yields super-robust weighting.
- `--filter-counts-level` -- Floor cells with extremely low counts at `median / level` before normalising.
  This prevents a handful of nearly empty uv-cells from being up-weighted far above the rest.
- `--l2-reweight-dof` -- Degrees-of-freedom parameter for an optional Student's t reweighting pass that down-weights visibilities with large model residuals (useful for residual RFI).
  Requires a reference model (via `--transfer-model-from` or cached from a previous iteration).
  Small values reweight aggressively and should only be used once the model is reasonably complete.

Weights are computed by `pfb grid` and written to the `.dds` dataset; the dirty image, the PSF, and every subsequent forward/backward pass in `pfb kclean`, `pfb sara`, and `pfb hci` all grid with that same stored set.

**Note:** re-running `pfb grid` is the supported way to change the weighting scheme after `pfb init` -- the weighting options above only take effect at this step, and none of them require redoing the MS ingestion.
To keep multiple weighting choices side by side, pass a distinct `--suffix` to each `pfb grid` run (e.g. `--suffix robust0 --robustness 0` and `--suffix uniform --robustness -2`); the downstream deconvolution commands then pick a dataset by matching suffix.

## Package structure

The project follows the hip-cargo src layout:

```
pfb-imaging/
├── src/pfb_imaging/
│   ├── cli/          # Lightweight CLI wrappers (Typer)
│   ├── core/         # Core implementations (lazy-loaded)
│   ├── cabs/         # Generated Stimela cab definitions (YAML)
│   ├── deconv/       # Deconvolution algorithms
│   ├── operators/    # Mathematical operators (gridding, PSF, Psi)
│   ├── opt/          # Optimization algorithms (PCG, FISTA, primal-dual)
│   ├── prox/         # Proximal operators
│   ├── utils/        # Utility functions
│   └── wavelets/     # Wavelet transform implementations
├── scripts/          # Profiling and automation scripts
├── tests/
├── Dockerfile
└── pyproject.toml
```

**Key separation:** CLI modules (`cli/`) are lightweight with lazy imports so that `pfb --help` and cab generation don't pull in the full scientific stack.
Core implementations live in `core/` and are imported only when a command is executed.

## Container images

Container images are published to GitHub Container Registry at `ghcr.io/ratt-ru/pfb-imaging`.
The full image URL (including tag) is the single source of truth and lives in `src/pfb_imaging/_container_image.py` as the `CONTAINER_IMAGE` variable, loaded via `importlib` (no CWD dependency, no `uv sync` needed).

```python
CONTAINER_IMAGE = "ghcr.io/ratt-ru/pfb-imaging:<tag>"
```

The `<tag>` is managed by three mechanisms:

- **Feature branches:** the developer manually updates the tag in `_container_image.py` to match the branch name.
- **Merge to main:** the `update-cabs.yml` GitHub Action rewrites the tag to `latest`, regenerates cab definitions, and commits the changes.
- **Releases:** `tbump` rewrites the tag to the semantic version (e.g. `0.0.9`) via `before_commit` hooks in `tbump.toml`.

Cab definitions are auto-generated with the correct image tag via pre-commit hooks and the `update-cabs.yml` GitHub Action -- the image URL is read from `_container_image.py` at generation time, so the `--image` flag is not needed.

## Development

This project uses:
- [uv](https://github.com/astral-sh/uv) for dependency management
- [ruff](https://github.com/astral-sh/ruff) for linting and formatting (core dependency — `generate-function` runs `ruff format` and `ruff check --fix` on generated code)
- [typer](https://typer.tiangolo.com/) for the CLI
- [git-cliff](https://git-cliff.org/) for `CHANGELOG` automation


### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/ratt-ru/pfb-imaging.git
cd pfb-imaging

# Install dependencies with development tools
uv sync --extra full --group dev --group test

# Install pre-commit hooks (recommended)
uv run pre-commit install --hook-type commit-msg
```

This will automatically run the hooks before each commit.
If any checks fail, the commit will be blocked until you fix the issues.

#### Running Hooks Manually

You can run the hooks manually on all files:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run
```

#### Updating Hook Versions

To update hook versions to the latest:

```bash
uv run pre-commit autoupdate
```

### Manual Code Quality Checks

If you prefer to run checks manually without pre-commit:

```bash
# Format code
uv run ruff format .

# Check and auto-fix linting issues
uv run ruff check . --fix

# Run tests
uv run pytest -v

```

### Numba cache and the `PFB_FRESH_NUMBA_CACHE` env var

The test suite pins Numba's on-disk cache to `<repo>/.numba_cache/` (via `NUMBA_CACHE_DIR` in `tests/conftest.py`) so compiled kernels survive across runs. Numba keys its cache by per-function source hash, which is normally enough — **except** for functions decorated with `inline="always"`. Those get compiled into their callers, but Numba does not track the cross-function dependency. If you edit an inlined helper while its callers' source stays identical, the cached machine code for the callers goes stale and can segfault on load.

When iterating on any `inline="always"` function (see `src/pfb_imaging/{wavelets,utils,operators}/`), force a clean rebuild:

```bash
PFB_FRESH_NUMBA_CACHE=1 uv run pytest -v
```

A single run with the flag set is enough — subsequent runs can drop it and reuse the fresh cache.

### Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/) to enable automated changelog generation via [git-cliff](https://git-cliff.org/).

Every commit message should follow this format:

```
<type>: <description>

[optional body]
```

**Types:**

| Type | When to use | Changelog section |
|------|------------|-------------------|
| `feat` | New feature or capability | Added |
| `fix` | Bug fix | Fixed |
| `refactor` | Code change that neither fixes a bug nor adds a feature | Changed |
| `perf` | Performance improvement | Changed |
| `docs` | Documentation only | Documentation |
| `test` | Adding or updating tests | Testing |
| `ci` | CI/CD changes | CI |
| `deps` | Dependency updates | Dependencies |
| `chore` | Maintenance tasks (cab regeneration, formatting) | Miscellaneous |

**Examples:**

```bash
git commit -m "feat: add support for MS dtype in type inference"
git commit -m "fix: handle empty docstrings in introspector"
git commit -m "refactor: simplify generate_cabs output formatting"
git commit -m "docs: add container fallback section to README"
git commit -m "test: add roundtrip test for List types"
```

**Scoped commits** (optional): Use parentheses to specify the affected component:

```bash
git commit -m "feat(init): add --license-type option for BSD-3-Clause"
git commit -m "fix(runner): resolve volume mount for symlinked paths"
```

### Contributing Workflow


1. **Create a feature branch**:
   ```bash
   git checkout -b your-feature-name
   ```

2. **Update the container image tag** in `src/pfb_imaging/_container_image.py` to match your branch name.

   This ensures the cab definitions generated by pre-commit hooks use the correct branch-specific image tag during development. You do not need to reset the tag before merging — the `update-cabs` workflow handles that automatically on merge to `main`.

3. **Make your changes** and ensure tests pass:
   ```bash
   uv run pytest -v
   ```

4. **Commit using [conventional commit messages](#commit-message-convention)**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   # Pre-commit hooks run automatically
   ```

   The pre-commit hooks keep the CLI and corresponding cab definitions in sync, enforce code quality and conventional commits.

5. **Push and create a pull request**:
   ```bash
   git push origin your-feature-name
   ```

The GitHub actions workflow automates containerisation by pushing container images to the GitHub Container Registry. Once the PR is merged, they also sync the name of container image corresponding to the branch (i.e. tagged with `:latest`).

## Acknowledgement

If you find any of this useful please cite the [pfb-imaging paper](https://arxiv.org/abs/2412.10073/).
