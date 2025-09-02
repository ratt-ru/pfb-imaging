===========
pfb-imaging
===========

Radio interferometric imaging suite base on the pre-conditioned forward-backward algorithm.

Installation
~~~~~~~~~~~~

It is best to install the packahe in a fresh virtual environment.
With the environment activated, update pip etc.

:code:`pip install -U pip setuptools wheel`

Now install the package 

:code:`pip install pfb-imaging`

For maximum performance it is strongly recommended to install ducc in no binary mode e.g.

:code:`pip install ducc0 --no-binary ducc0`

This might take some time to compile. 

Quick start
~~~~~~~~~~~

The easiest way to use ``pfb-imaging`` is via the stimela recipes given in the `recipes folder <recipes/>`_.
Once the package is installed, a recipe can be queried for its input and output parameters using the ``stimela doc`` command.
For example, to see the inputs and outputs of the ``sara`` recipe, simply run

:code:`stimela doc pfb-imaging/recipes/sara.yml`

The recipe can then be run with the ``stimela run`` command.
For example, to run the recipe with all parameters set to the defaults, use 

:code:`stimela run pfb-imaging/recipes/sara.yml sara ms=path/to/data.ms base-dir=path/to/base/output/directory image-name=saraout`

The recipe should contain sensible defaults for MeerKAT data at L-band. 
Note that the recipe exposes a minimal set of functional parameters.
More exotic parameters appear lower down in the list and some parameters are not exposed at all.
These can be exposed by passing the ``--obscure`` flag to `stimela doc`

:code:`stimela doc --obscure pfb-imaging/recipes/sara.yml`

These can be specified explicitly either by setting it from the command line or by adding (or adjusting) it in the recipe schema.
Usage through stimela is still a bit limited as the structure of the recipe is fixed (although steps can be run in isolation if required, see `stimela run --help` for further details).
Keep reading if you want to interacting directly with the applications from the command line.
Note that cabs for all the applications are also available through `cult-cargo <https://github.com/caracal-pipeline/cult-cargo/>`_.

Module of workers
~~~~~~~~~~~~~~~~~~~

Each worker module can be run as a standalone program.
Run

:code:`pfb --help`

for a list of available workers.

Documentation for each worker is listed under

:code:`pfb workername --help`

Default naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default outputs are named using a conbination of the following parameters

* ``--output-filename``
* ``--product``
* ``--suffix``

The output dataset of the ``init`` application (viz. the ``xds``) will be called

:code:`f"{output_filename}_{product}.xds"`

with standard python string substitutions. The grid worker creates another dataset (viz. the ``dds``) called

:code:`f"{output_filename}_{product}_{suffix}.dds"`

i.e. with the suffix appended (the default suffix being ``main``) which contains imaging data products such as the dirty and PSF images etc.
This is to allow imaging multiple fields from a single set of (possibly averaged) corrected Stokes visibilities produced by the ``init`` applcation.
For example, the sun can be imaged from and ``xds`` prepared for the main field by setting ``--target sun`` and, for example, ``--suffix sun``.
The ``--target`` parameter can be any object recognised by ``astropy`` and can also be specified using the ``HH:MM:SS,DD:MM:SS`` format.
All deconvolution algorithms interact with the ``dds`` produced by the ``grid`` application and will produce a component model called

:code:`f"{output_filename}_{product}_{suffix}_model.mds"`

as well as standard pixelated images which are stored in the ``dds``.
They also produce fits images but these are mainly for inspection with fits viewers.
By default logs and fits files are stored in the directory specifed by ``--output-filename`` but these can be overwritten with the

* ``--fits-output-folder`` and
* ``--log-directory``

parameters. Fits files are stored with the same naming convention but with the base output directory set to ``--fits-output-folder``.
Logs are stored by application name with a time stamp to prevent inadvertently overwriting them.
The output paths for all files are reported in the log.

Parallelism settings
~~~~~~~~~~~~~~~~~~~~~~

There are two settings controlling parallelism viz.

* ``--nworkers`` which controls how many chunks (usually corresponding to imaging bands) will be processed in parallel and
* ``--nthreads`` which specifies the number of threads available to each worker (eg. for gridding, FFT's and wavelet transforms).

By default only a single worker is used so that datasets will be processed one at a time.
This results in the smallest memory footprint but won't always result in optimal resource utilisation.
It also allows for easy debugging as there is no Dask cluster involved in this case.
However, for better resource utilisation and or distributing computations over a cluster, you may wish to set ``--nworkers`` larger than one.
This uses multiple Dask workers (processes) to process chunks in parallel and is especially useful for the ``init``, ``grid`` and ``fluxtractor`` applications.
It is usually advisable to set ``--nworkers`` to the number of desired imaging bands which is set by the ``--channels-per-image`` parameter when initialising corrected Stokes visibilities with the ``init`` application.
The product of ``--nworkers`` and ``--nthreads`` should not exceed the available resources.


Acknowledgement
~~~~~~~~~~~~~~~~~

If you find any of this useful please cite the `pfb-imaging paper <https://arxiv.org/abs/2412.10073/>`_.
