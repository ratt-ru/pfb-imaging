# pfb-clean
Preconditioned forward/backward clean algorithm. 

Make sure MPI support is available by installing python3-mpi4py sytemwide.

The following dependencies are installed automatically

numpy
scipy
numba
astropy
python-casacore
dask
dask[array]
dask-ms[xarray]
codex-africanus

The following packages need to be installed manually from requirements

pypocketfft (git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft)
nifty_gridder (git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git)


## Outline of functionality
Since there is a bit of finetuning involved in the algo it has been decomposed
into a number of scripts.


### ```prep_data.py```
Pre-process data from measurement set into conveneient format.
Currently the data are read from a measurement set and the Stokes I
visibilities are written to a new table which contains only the necessary data.
The dirty image and psf are also computed if requested.
These are saved in an .npz file that can be read by ```solve_x0.py```.


### ```solve_x0.py```
A utility script that performs a single major cycle i.e. solve for a model image given
only the dirty/residual image and psf. In addition to finding a good starting point
this script should be used to fine tune the parameters of the algo. It can, for example,
be used to get an idea of what the regulariser strengths should be. 


### ```reweight_l2.py```
(Re-)Compute robust imaging weights given a model and data. Useful for getting rid of
calibration artefacts and low-leve RFI. Should be used with caution. If data are reweighted
pre-maturely it is possible to erase any structures not present in the model.


### ```pfb.py```
Performs pre-conditioned forward-backward clean algorithm. 


