# pfb-clean
Preconditioned forward-backward clean algorithm.

Install the package by cloning and running

```
$ pip install -e pfb-clean/
```

Note casacore needs to be installed on the system for this to work. 

## Outline of functionality
The following standalone scripts are included:

### ```make_dirty.py``` 
Reads the output of ```prep_data.py``` and produces a dirty image and a psf of twice the size.
Only natural weighting is currently supported. 


### ```solve_x0.py```
A utility script that performs a single major cycle i.e. solve for a model image given only the dirty/residual image and psf in the form of a fits file. 
This is useful to tune the parameters of the algorithm since it is much faster than the full ```pfbclean.py``` script but has the same regularisers.
It can (and should) also be used to generate a good starting point for the algorithm.
You have to use the dirty image and psf produced by ```make_dirty.py``` as these will have the required precision, units and psf size to ensure a stable solution.
Note that the input dirty image and psf should not be normalised by wsum. 


### ```pfbclean.py```
Performs pre-conditioned forward-backward clean algorithm. 
In principle you just need to point it at a measurement set but it might be beneficial to to tune the regularisers (eg. sig_l21) using ```solve_x0.py```.
The result can be used as a starting point for ```pfbclean.py``` by passing in the model produced using the ```--x0``` option.

