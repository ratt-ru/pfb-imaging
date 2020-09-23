
import numpy as np
import time
from africanus.gps.kernels import exponential_squared as expsq
from pfb.operators import freqmul, Prior
import matplotlib.pyplot as plt
import dask.array as da


if __name__=="__main__":
    nv = 12
    nx = 4096
    ny = 4096

    v = np.arange(nv).astype(np.float64)

    sigma0 = 0.01
    l = 0.25*nv
    Kmat = expsq(v, v, sigma0, l)

    x = np.random.randn(nv, nx, ny)

    K = Prior(sigma0, nv, nx, ny)

    ti = time.time()
    res1 = Kmat.dot(x.reshape(nv, nx*ny)).reshape(nv, nx, ny)
    print(time.time() - ti)
    ti = time.time()
    res2 = K.dot(x)
    print(time.time() - ti)
    ti = time.time()
    res3 = np.einsum('ij,jkl->ikl', Kmat, x)
    print(time.time() - ti)


    print(np.abs(res1-res2).max())
    print(np.abs(res1-res3).max())


    