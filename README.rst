===========
pfb-imaging
===========

.. .. image:: /logo/Gemini_Generated_Image_m19n6gm19n6gm19n.jpg
..    :align: center

Radio interferometric imaging suite base on the pre-conditioned forward-backward algorithm.

Installation
~~~~~~~~~~~~

Install the package by cloning and running

:code:`$ pip install -e pfb-imaging/`

Note casacore needs to be installed on the system for this to work.

You will probably may also need to update pip eg.

:code:`$ pip install -U pip setuptools wheel`

For maximum performance it is strongly recommended to install ducc in
no binary mode eg

:code:`$ git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git`

:code:`$ pip install -e ducc`

Default naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default outputs are named using a conbination of the following parameters

* `--output-filename`
* `--product`
* `--suffix`

The output dataset of the `init` application (viz. the `xds`) will be called

:code:`f"{output_filename}_{product}.xds"`

with standard python string substitutions. The grid worker creates another dataset (viz. the `dds``) called

:code:`f"{output_filename}_{product}_{suffix}.dds"`

i.e. with the suffix appended (the default suffix being `main`) which contains imaging data products such as the dirty and PSF images etc.
This is to allow imaging multiple fields from a single set of (possibly averaged) corrected Stokes visibilities produced by the `init` applcation.
For example, the sun can be imaged from and `xds` prepared for the main field by setting `--target sun` and, for example, `--suffix sun`.
The `--target` parameter can be any object recognised by `astropy` and can also be specified using the `HH:MM:SS,DD:MM:SS` format.
All deconvolution algorithms interact with the `dds` produced by the `grid` application and will produce a component model called

:code:`f"{output_filename}_{product}_{suffix}_model.mds"`

as well as standard pixelated images which are stored in the `dds`.
They also produce fits images but these are mainly for inspection with fits viewers.
By default logs and fits files are stored in the directory specifed by `--output-filename` but these can be overwritten with the

* `--fits-output-folder` and
* `--log-directory`

parameters. Fits files are stored with the same naming convention but with the base output directory set to `--fits-output-folder`.
Logs are stored by application name with a time stamp to prevent inadvertently overwriting them.
The output paths for all files are reported in the log.

Parallelism settings
~~~~~~~~~~~~~~~~~~~~

There are two settings controlling parallelism viz.

* `--nworkers` which controls how many chunks (usually corresponding to imaging bands) will be processed in parallel and
* `--nthreads-per-worker` which specifies the number of threads available to each worker (eg. for gridding, FFT's and wavelet transforms).

By default only a single worker is used so that datasets will be processed one at a time.
This results in the smallest memory footprint but won't always result in optimal resource utilisation.
It also allows for easy debugging as there is no Dask cluster involved in this case.
However, for better resource utilisation and or distributing computations over a cluster, you may wish to set `--nworkers` larger than one.
This uses multiple Dask workers (processes) to process chunks in parallel and is especially useful for the `init`, `grid` and `fluxmop` applications.
It is usually advisable to set `--nworkers` to the number of desired imaging bands which is set by the `--channels-per-image` parameter when initialising corrected Stokes visibilities with the `init` application.
The product of `--nworkers` and `--nthreads-per-worker` should not exceed the available resources.

Acknowledgement
~~~~~~~~~~~~~~

If you find any of this useful please cite (for now)

https://arxiv.org/abs/2101.08072
