Module of workers
==================

Each worker module can be run as a standalone program.
Run

$ pfb --help

for a list of available workers.

Documentation for each worker is listed under

$ pfb workername --help

All workers can be run on a distributed scheduler chunked over row and
imaging band.


.. Imaging weights
.. ---------------
.. Unless you know your instrument and its environment exactly, a typical
.. workflow starts with creating the imaging weights. See

.. $ pfb imweight --help

.. for options. This will write out a column (IMAGING_WEIGHT_SPECTRUM by default)
.. containing the imaging weights. These will be combined with the natural weights
.. column as described in imweights docs).


.. Gridding
.. -----------
.. The next step is to flip our data products into a more convenient form for
.. imaging. This is achievd with the grid worker, run

.. $ pfb grid --help

.. for available options. Your cell size and number of pixels needs to be
.. consistent what was provided to the imweight worker. These will be
.. determined automatically using the super-resolution-factor and
.. field-of-view parameters. Data products produced by the grid worker can
.. include a dirty image, PSF, weights and visibilities (obtained by setting
.. the dirty, psf, weight and vis options, respectively). The weights and
.. visibilities that are written out correspond to the product requested.
.. For example, if product is set to 'I' then the Stokes I weights and
.. visibilities will be computed and written to a chunked repositry called
.. output_filename_product.xds.zarr which can be opened using the experimental
.. dask-ms xds_from_zarr function. Importantly, the weights have been
.. pre-applied to the data and should not be applied during gridding.
.. They incorporate all of the imaging weights and gain solutions so that
.. gradient can be computed as

.. $ IR = R.H(V - W R x)

.. where R.H/R is the unweighted gridder/degridder, V/W are are the precomputed
.. visibilities/weights and x is the model. Alternatively, the gradient can also
.. be computed as

.. $ IR = ID - R.H W R x

.. which avoids loading the visibilities into memory. The grid worker will also
.. initialise a model dataset called output_filename_product.mds.zarr. These
.. two products (i.e. the xds and the mds) are used by the deconvolution
.. workers documented below. Fits files of the dirty image and PSF can also
.. optionally be written out.


.. Deconvolution
.. -------------

.. There are currently three deconvolution workers that can be used to get rid of
.. those pesky PSF sidelobes. These workers have all been designed to work with
.. the data products produced by the grid worker and to be compatible with the
.. forward-backward algorithmic structure. Importantly, the grid worker needs to
.. be rerun if the weights or calibration solutions are altered.

.. Clean
.. ~~~~~

.. The standard single-scale clean algorithm is implemented in the clean worker.
.. Run

.. $ pfb clean --help

.. for available options.

.. Forward step
.. ~~~~~~~~~~~~
.. The forward step uses the conjugate gradient algorithm to invert the linear
.. system

.. $ A x = dirty

.. where A can be convolution with the PSF (approximate image space solution)
.. or an application of the heessian. The latter is more accurate but can be
.. computationally prohibitive. Have a look at

.. $ pfb forward --help

.. for available options.

.. Backward step
.. ~~~~~~~~~~~~~
.. The backward step can be used to enforce a possitivity constraint and to
.. impose sparsity in a dictionary of wavelets (SARA prior). See

.. $ pfb backward --help

.. for available options.
