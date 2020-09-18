
import numpy as np
import dask.array as da
import time
from pfb.operators import Gridder

if __name__=="__main__":
    nband = 5
    cell = 0.15
    nx = 2048
    ny = 2048

    ms = ['/home/landman/Data/VLA/CYGA/CygA-S-D-HI.MS']

    R = Gridder(ms, nx, ny, cell, nband)

    x = np.zeros((nband, nx, ny), dtype=np.float64)
    x[:, nx//2, ny//2] = 100.0
    
    residual = R.make_residual(x)

    import matplotlib.pyplot as plt
    for i in range(nband):
        plt.imshow(residual[i])
        plt.colorbar()
        plt.show()
