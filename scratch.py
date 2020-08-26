import numpy as np
np.random.seed(420)
from pfb.operators import Prior
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt

if __name__=="__main__":
    nx = 16
    ny = 16
    nchan = 4
    N = nchan*nx*ny

    v = np.linspace(0.5, 1.5, nchan)
    Kop = Prior(v, 1.0, 0.1, nx, ny, 1e-5)
    
    K = np.kron(Kop.Kv, np.kron(Kop.Kl, Kop.Km))

    xi = np.random.randn(N)
    res1 = Kop.dot(xi)
    res2 = K.dot(xi)

    print(np.abs(res1 - res2).max())

    Kinv = np.kron(Kop.Kvinv, np.kron(Kop.Klinv, Kop.Kminv))

    xi = np.random.randn(N)

    res1 = Kop.idot(xi)
    res2 = Kinv.dot(xi)

    print(np.abs(res1 - res2).max())


    



