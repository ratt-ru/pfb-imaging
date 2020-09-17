
import numpy as np
import dask.array as da
import time

def accumulate_dirty(dirty, this_dirty, bands, subbands):
    return da.blockwise(_accumulate_dirty, ('band', 'nx', 'ny'),
                        dirty, ('band', 'nx', 'ny'),
                        this_dirty, ('subband', 'nx', 'ny'),
                        bands, ('band',),
                        subbands, None,
                        dtype=dirty.dtype)

def _accumulate_dirty(dirty, this_dirty, bands, subbands):
    if bands in subbands:
        dirty += this_dirty[subbands.index(bands)]
    return dirty

if __name__=="__main__":
    nband = 8
    subband = 3
    nx = 12
    ny = 12

    subbands = (2, 4, 6)
    this_dirty = np.random.randn(subband, nx, ny)
    this_dirty_da = da.from_array(this_dirty, chunks=(1, nx, ny))
    
    bands = (0, 1, 2, 3, 4, 5, 6, 7)
    bands_da = da.from_array(bands, chunks=1)
    dirty  = np.zeros((nband, nx, ny))
    dirty_da = da.from_array(dirty, chunks=(1, nx, ny))

    result = accumulate_dirty(dirty_da, this_dirty_da, bands_da, subbands).compute()

    for i in subbands:
        result[i] -= this_dirty[subbands.index(i)]

    print(np.abs(result).max())