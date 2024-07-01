===========
pfb-imaging
===========

.. .. image:: /logo/Gemini_Generated_Image_m19n6gm19n6gm19n.jpg
..    :align: center

Radio interferometric imaging suite base on the pre-conditioned forward-backward algorithm.

Install the package by cloning and running

:code:`$ pip install -e pfb-imaging/`

Note casacore needs to be installed on the system for this to work.

You will probably may also need to update pip eg.

:code:`$ pip install -U pip setuptools wheel`

For maximum performance it is strongly recommended to install ducc in
no binary mode eg

:code:`$ git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git`

:code:`$ pip install -e ducc`

If you find any of this useful please cite (for now)

https://arxiv.org/abs/2101.08072

A word about naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default outputs are named using a conbination of the following parameters

* `--output-filename`
* `--product`
* `--suffix`

The output dataset of the `init` application will be called

:code:`f"{output_filename}_{product}.xds"`

with standard python string substitutions. The grid worker will creates another dataset called

:code:`f"{output_filename}_{product}_{suffix}.dds"`

i.e. with the suffix appended (the default suffix being main).
This is to allow imaging multiple fields from a single set of averaged Stokes visibilities produced by the `init` applcation (eg. the main field and the sun).
All deconvolution algorithms interact with the latter of these and they will produce a component model called

:code:`f"{output_filename}_{product}_{suffix}_model.mds"`
