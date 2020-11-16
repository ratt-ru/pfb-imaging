import numpy as np
from pfb.operators import kron_matvec, kron_matvec2
from africanus.gps.kernels import exponential_squared as expsq
from time import time
from ducc0.fft import r2c

def test_matvec():
    nv = 8
    nx = 1024
    ny = 1024

    xi = np.random.randn(nv,nx,ny)

    v = np.linspace(-0.5, 0.5, nv)
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)

    A1 = expsq(v, v, 1.0, 0.1)
    A2 = expsq(x, x, 1.0, 0.1)
    A3 = expsq(y, y, 1.0, 0.01)

    A = (A1, A2, A3)

    # make sure they have compiled
    res1 = kron_matvec(A, xi)
    res2 = kron_matvec2(A, xi)


    ti = time()
    res1 = kron_matvec(A, xi)
    print(time() - ti)

    ti = time()
    res2 = kron_matvec2(A, xi)
    print(time() - ti)


    # FFT for reference
    # ti = time()
    # res2 = r2c(xi, axes=(0,1,2), nthreads=2, forward=True, inorm=0)
    # print(time() - ti)


test_matvec()