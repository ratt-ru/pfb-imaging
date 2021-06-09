Module of workers
==================

Each worker module can be run as a standalone program.
run

$ pfbworkers --help

for a list of available workers.

Documentation for each worker is listed under

$ pfbworkers workername --help

All workers can be run on a distributed scheduler chunked over row and
imaging band (the current limitation being that all images fit onto a single
worker).


.. Imaging weights
.. ---------------
.. Unless you know your instrument and its environment exactly, a typical
.. workflow starts with creating the imaging weights. See

.. $ pfbworkers imweight --help

.. for options. This will write out a column (IMAGIMG_WEIGHT_SPECTRUM by default)
.. containing the imaging weights (note these are combined with the WEIGHT
.. column as described in imweights docs).


.. Dirty image
.. -----------
.. The next step is getting a dirty image. Use

.. $ pfbworkers dirty --help

.. for specific options (note that your cell size and field of view needs to be
.. consistent when running any of imweight, dirty or psf). Most of the time it is
.. easiest to set the image size via the super-resolution-factor and
.. field-of-view options.

.. Point spread function
.. ---------------------
.. Once a dirty image has been created, you will need a PSF. See

.. $ pfbworkers psf --help

.. for options. By default the PSF is constructed at twice the size of the image.
.. This is usually what you want if you plan to use the fluxmop capabilities so
.. undersize at your own peril. The psf worker will write out a weight table
.. in .zarr format that makes "major" iterations significantly faster. The idea
.. here is to compute gradients using

.. $ IR = ID - R.H W R I

.. where W denotes the weights and R is the measurement operator. Compared to the
.. visibility space expression

.. $ IR = R.H W (V - R I)

.. the former needs only about a third of the IO (less if the weights are highly
.. compressible). We might also be able to exploit some special structure here
.. but that is still WIP.

.. Deconvolution
.. -------------

.. You can then run one fo the deconvolution algorithms to get rid of those pesky
.. PSF sidelobes. Have a look at (more on the way)

.. $ pfbworkers clean --help

.. for example. You have to use consistent settings for the dirty image, PSF and
.. weight table. There are good reasons for this. For example, CLEAN doesn't work
.. very well if you start off with natural weighting but you might want to change
.. the weighting as you get to the faint (and probably diffuse) background.
.. In this case, you might want to start with say Briggs -1 and progress all the
.. way through to natural weighting. Especially if you are using the fluxmop.

.. Flux mop
.. --------
.. The flux mop is a fairly simple utility. It basically uses the conjugate
.. gradient algorithm to invert a linear system of the form

.. $ Ax = b

.. where the operator A is some regularised approximation of the Hessian (eg.
.. convolution by the PSF + Identity) and b is the residual image. The solution
.. is in the same units as the model and will (probably) try to overfit the data
.. (which is where the name comes from) under the given prior. This is a
.. particularly useful tool when using Single Scale Clean (SSC). SSC is very
.. robust against imperfect data (calibration artefacts, RFI etc.) but suffers
.. from some very undesirable properties viz. it starts crawling to a halt if the
.. remaining faint emission is diffuse and it does not produce physical models
.. (it also sucks when using natural weighting). All of the above can be overcome
.. by following the simple instructions that follow.

.. Because of the way SSC works (viz. it adds components one pixel at a time
.. (or interpolates residual visibility amplitudes using a constant function)),
.. you might want to say "Hey, wait a sec. What if we try doing all them pixels
.. at the same time. You can do that (and more) with the flux mop. Have a look at

.. $ pfbworkers fluxmop --help

.. In particular, for SSC, try using the --model-as-mask feature (just note that
.. passing in the weight-table for this step might be prohibitively
.. computationally expensive).