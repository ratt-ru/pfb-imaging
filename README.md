# pfb-clean
Preconditioned forward/backward clean algorithm.

Make sure MPI support is available by installing python3-mpi4py sytemwide.

The following dependencies are installed automatically:

numpy\
scipy\
numba\
astropy\
python-casacore\
dask\
dask[array]\
dask-ms[xarray]\
codex-africanus

The following packages need to be installed manually from requirements

pypocketfft (git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft)\
nifty_gridder (git+https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git)


## Outline of functionality
The following standalone scripts are included:


### ```prep_data.py```
Pre-process data from measurement set into format required by gridder.
Currently data are read from a list of measurement sets and the Stokes I visibilities and weights are written to a new table which contains only the necessary data.
This table is read by the gridder which currently expects both the data and weights to be of shape ```(nrow, nchan)```. 
By default the gridder will only keep the data for one imaging band in memory at a time but chunking over row is also supported (in principle). 


### ```make_dirty.py``` 
Reads the output of ```prep_data.py``` and produces a dirty image and a psf of twice the size.
Only natural weighting is currently supported. 


### ```solve_x0.py```
A utility script that performs a single major cycle i.e. solve for a model image given only the dirty/residual image and psf in the form of a fits file. 
This is useful to tune the parameters of the algorithm since it is much faster than the full ```pfb.py``` script but has the same regularisers.
It can (and should) also be used to generate a good starting point for the algorithm. 
In principle the script doesn't care where the dirty image and psf come from but it is advised to use the ones produced by ```make_dirty.py```
as these will have the required precision, units and psf size to ensure a stable solution.
Note that the input dirty image and psf should not be normalised by wsum for this to work. 


### ```pfb.py```
Performs pre-conditioned forward-backward clean algorithm. 
In principle you just need to point it at the table produced by ```prep_data.py```.


