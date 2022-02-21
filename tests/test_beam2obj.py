import numpy as np
import dask
import dask.array as da
from pfb.utils.beam import beam2obj

def test_beam2obj():
    x = np.linspace(-1, 1, 100)
    bvals = np.kron(np.cos(x)[None, :], np.cos(x)[:, None])
    bvals = da.from_array(bvals)
    x = da.from_array(x)


    b = dask.compute(beam2obj(bvals, x, x))[0]

    bvals_interp = b[0](x, x)

    assert (np.abs(bvals_interp - bvals) < 1e-14).all()
