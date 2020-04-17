import numpy as np
from scipy.linalg import norm
from pfb.utils import prox_21
from pfb.operators import PSI
import pywt

if __name__=="__main__":
    nx = 512
    ny = 512
    nchan = 8
    nlevels = 3

    # construct operator (frequency axis dealt with in dot and hdot)
    psi = PSI(nchan, nx, ny, nlevels=nlevels)
    nbasis = psi.nbasis
    ntot = psi.ntot  # number of wavelet coefficients

    p = np.ones((nchan, nx, ny), dtype=np.float64)
    weights_21 = np.empty(nbasis, dtype=object)
    weights_21[0] = np.ones(nx*ny, dtype=real_type)  # for dirac basis
    for m in range(1, psi.nbasis):
        weights_21[m] = np.ones(psi.ntot, dtype=real_type)  # all other wavelet bases
    
    sig_21 = 1e-5

    x = prox_21(p, sig_21, weights_21, psi=psi)

            