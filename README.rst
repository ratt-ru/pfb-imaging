===========
pfb-imaging
===========

Radio interferometric imaging suite based on the preconditioned forward-backward algorithm.
The project follows the `hip-cargo <https://github.com/landmanbester/hip-cargo>`_ package format:
lightweight CLI installation with auto-generated `stimela <https://github.com/caracal-pipeline/stimela>`_ cab definitions and containerised execution.

Installation
~~~~~~~~~~~~

**Lightweight (CLI + cabs only):**

.. code-block:: bash

   pip install pfb-imaging

This installs the CLI and stimela cab definitions without the full scientific stack.
The cabs can be included in stimela recipes using:

.. code-block:: yaml

   _include:
     - (pfb_imaging.cabs)init.yml

**Full scientific stack:**

.. code-block:: bash

   pip install "pfb-imaging[full]"

Or for development:

.. code-block:: bash

   git clone https://github.com/ratt-ru/pfb-imaging.git
   cd pfb-imaging
   uv sync --extra full --extra dev
   uv run pre-commit install

For maximum performance it is strongly recommended to install ducc in no-binary mode:

.. code-block:: bash

   pip install ducc0 --no-binary ducc0


Quick start
~~~~~~~~~~~

The easiest way to use ``pfb-imaging`` is via the stimela recipes given in the `recipes folder <recipes/>`_.
Once the package is installed, a recipe can be queried for its input and output parameters using the ``stimela doc`` command.
For example, to see the inputs and outputs of the ``sara`` recipe, simply run

.. code-block:: bash

   stimela doc 'pfb_imaging.recipes::sara.yaml'

The recipe can then be run with the ``stimela run`` command:

.. code-block:: bash

   stimela run 'pfb_imaging.recipes::sara.yaml' sara \
     ms=path/to/data.ms \
     base-dir=path/to/base/output/directory \
     image-name=saraout

The recipe should contain sensible defaults for MeerKAT data at L-band.


CLI documentation
~~~~~~~~~~~~~~~~~

The CLI is built with `Typer <https://typer.tiangolo.com/>`_ and provides rich, auto-generated documentation.
To list all available commands:

.. code-block:: bash

   pfb --help

To get detailed documentation for a specific command including all parameters, types, and defaults:

.. code-block:: bash

   pfb init --help

This is often more useful than ``stimela doc`` as it shows the full parameter documentation with types and defaults directly in the terminal.


CLI commands
~~~~~~~~~~~~

The processing pipeline follows a modular pattern where each step is a separate command:

1. ``pfb init`` -- Parse measurement sets into xarray datasets
2. ``pfb grid`` -- Create dirty images, PSFs, and weights
3. ``pfb kclean`` -- Classical deconvolution (Hogbom/Clark)
4. ``pfb sara`` -- Advanced deconvolution with sparsity constraints
5. ``pfb restore`` -- Restore clean components to final image
6. ``pfb degrid`` -- Subtract model from visibilities

Additional commands:

* ``pfb deconv`` -- General deconvolution (replaces individual algorithm apps)
* ``pfb hci`` -- High cadence imaging
* ``pfb fluxtractor`` -- Flux extraction
* ``pfb model2comps`` -- Convert model to components


Execution backends
~~~~~~~~~~~~~~~~~~

Every command supports a ``--backend`` option that controls how the command is executed.
This is provided by `hip-cargo <https://github.com/landmanbester/hip-cargo>`_ and enables container fallback execution: when the full scientific stack is not installed locally, commands automatically run inside a container.

Available backends:

* ``auto`` (default) -- Try native execution first; if the core module import fails (lightweight install), fall back to the best available container runtime.
* ``native`` -- Run natively using the locally installed Python environment. Fails with ``ImportError`` if dependencies are missing.
* ``docker`` -- Run inside a Docker container.
* ``podman`` -- Run inside a Podman container (daemonless, rootless).
* ``apptainer`` -- Run inside an Apptainer container (HPC-friendly, formerly Singularity).
* ``singularity`` -- Run inside a Singularity container.

An additional ``--always-pull-images`` flag forces re-pulling the container image before execution, useful for ensuring you have the latest version.

Example usage:

.. code-block:: bash

   # Run natively (requires full install)
   pfb init --ms data.ms --output-filename out --backend native

   # Run in a Docker container (lightweight install only)
   pfb init --ms data.ms --output-filename out --backend docker

   # Auto-detect: native if available, otherwise container
   pfb init --ms data.ms --output-filename out

Volume mounts are resolved automatically from the command's type hints: input paths are mounted read-only, output paths read-write.
Docker and Podman run as the current user to avoid root-owned output files.


Default naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Output files follow consistent naming patterns using ``--output-filename``, ``--product``, and ``--suffix``:

* XDS datasets: ``{output_filename}_{product}.xds``
* DDS datasets: ``{output_filename}_{product}_{suffix}.dds``
* Models: ``{output_filename}_{product}_{suffix}_model.mds``
* FITS files: same convention with appropriate extensions

The ``--suffix`` parameter (default ``main``) allows imaging multiple fields from a single set of corrected Stokes visibilities.
For example, the sun can be imaged by setting ``--target sun --suffix sun``.
The ``--target`` parameter accepts any object recognised by ``astropy`` or ``HH:MM:SS,DD:MM:SS`` format.


Parallelism settings
~~~~~~~~~~~~~~~~~~~~

Two settings control parallelism:

* ``--nworkers`` controls how many chunks (usually imaging bands) are processed in parallel.
* ``--nthreads`` specifies threads available to each worker (gridding, FFTs, wavelet transforms).

By default a single worker is used for the smallest memory footprint and easy debugging.
Set ``--nworkers`` larger than one to use multiple Dask workers for parallel chunk processing.
The product of ``--nworkers`` and ``--nthreads`` should not exceed available resources.


Package structure
~~~~~~~~~~~~~~~~~

The project follows the hip-cargo src layout:

.. code-block::

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

**Key separation:** CLI modules (``cli/``) are lightweight with lazy imports so that ``pfb --help`` and cab generation don't pull in the full scientific stack.
Core implementations live in ``core/`` and are imported only when a command is executed.


Container images
~~~~~~~~~~~~~~~~

Container images are published to GitHub Container Registry at ``ghcr.io/ratt-ru/pfb-imaging``.
The full image URL (including tag) is the single source of truth and lives in ``pyproject.toml`` as a hip-cargo entry point::

    [project.entry-points."hip.cargo"]
    container-image = "ghcr.io/ratt-ru/pfb-imaging:<tag>"

The ``<tag>`` is managed by three mechanisms:

* **Feature branches:** the developer manually updates the tag to match the branch name and runs ``uv sync`` before committing.
* **Merge to main:** the ``update-cabs.yml`` GitHub Action rewrites the tag to ``latest``, regenerates cab definitions, and commits the updated ``pyproject.toml``, ``uv.lock``, and cab YAML files.
* **Releases:** ``tbump`` rewrites the tag to the semantic version (e.g. ``0.0.9``) via ``before_commit`` hooks in ``tbump.toml``.

Cab definitions are auto-generated with the correct image tag via pre-commit hooks and the ``update-cabs.yml`` GitHub Action — the image URL is read from the entry point at generation time, so the ``--image`` flag is not needed.


Development
~~~~~~~~~~~

.. code-block:: bash

   # Install with full and dev dependencies
   uv sync --extra full --extra dev

   # Install pre-commit hooks
   uv run pre-commit install

   # Run tests
   uv run pytest -v tests/

   # Format and lint
   uv run ruff format .
   uv run ruff check . --fix


Acknowledgement
~~~~~~~~~~~~~~~

If you find any of this useful please cite the `pfb-imaging paper <https://arxiv.org/abs/2412.10073/>`_.
