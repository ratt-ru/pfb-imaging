import numpy as np
import numba
import pywt
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pfb.wavelets import dwt2d, idwt2d, coeff_size, signal_size
import pytest
pmp = pytest.mark.parametrize

# TODO - return shape is currently incorrect for certain odd valued inputs
# https://github.com/PyWavelets/pywt/issues/725
@pmp("wavelet", ["db1", "db4", "db5"])
@pmp("data_shape", [(128,256), (512,128)])  #, (129,255), (511,257)])
@pmp('nlevel', [1, 2, 3])
def test_dwt_idwt_pywt(wavelet, data_shape, nlevel):
    nx, ny = data_shape
    data = np.random.random(size=data_shape)

    # pywt comparison
    alpha = pywt.wavedec2(data, wavelet, mode='zero', level=nlevel)
    xrec = pywt.waverec2(alpha, wavelet, mode='zero')
    assert_array_almost_equal(data, xrec[0:data.shape[0], 0:data.shape[1]])

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    # bookeeping
    N2Cx = {}
    N2Cy = {}
    Nx = nx
    Ny = ny
    Ntotx = 0
    Ntoty = 0
    sx = ()
    sy = ()
    spx = ()
    spy = ()
    F = int(wavelet[-1])*2  # filter length
    for k in range(nlevel):
        Cx = coeff_size(Nx, F)
        Cy = coeff_size(Ny, F)
        N2Cx[k] = (signal_size(Cx, F), Cx)
        N2Cy[k] = (signal_size(Cy, F), Cy)
        Ntotx += Cx
        Ntoty += Cy
        sx += (Cx,)
        sy += (Cy,)
        Nx = Cx + Cx%2
        Ny = Cy + Cy%2
        spx += (signal_size(Cx, F),)
        spy += (signal_size(Cy, F),)
    Ntotx += Cx  # last approx coeffs
    Ntoty += Cy

    ix = numba.typed.Dict()
    iy = numba.typed.Dict()
    lowx = N2Cx[nlevel-1][1]
    lowy = N2Cy[nlevel-1][1]
    ix[nlevel-1] = (lowx, 2*lowx)
    iy[nlevel-1] = (lowy, 2*lowy)
    lowx *= 2
    lowy *= 2
    for k in reversed(range(nlevel-1)):
        highx = N2Cx[k][1]
        highy = N2Cy[k][1]
        ix[k] = (lowx, lowx + highx)
        iy[k] = (lowy, lowy + highy)
        lowx += highx
        lowy += highy

    alpha2 = np.zeros((Ntoty, Ntotx))
    cbuff = np.zeros((Ntotx, Ntoty))
    cbuffT = np.zeros((Ntoty, Ntotx))
    dwt2d(data, alpha2, cbuff, cbuffT, ix, iy, sx, sy, dec_lo, dec_hi, nlevel)
    xrec2 = np.zeros((nx, ny))
    coeffs = np.zeros((Ntoty, Ntotx))
    idwt2d(alpha2, xrec2, coeffs, cbuff, cbuffT, ix, iy, sx, sy,
        spx, spy, rec_lo, rec_hi, nlevel)

    assert_array_almost_equal(data, xrec2)


    # pack pywt into array
    alpha3 = np.zeros_like(alpha2.T)
    Nx = ix[nlevel-1][0]
    Ny = iy[nlevel-1][0]
    alpha3[0:Nx,0:Ny] = alpha[0]
    for i, j in zip(range(1, nlevel+1), reversed(range(nlevel))):
        # bottom left
        lowx, highx = ix[j]
        nax = sx[j]
        lowx2 = highx - 2*nax
        npix_x = highx - lowx
        lowy, highy = iy[j]
        nay = sy[j]
        lowy2 = highy - 2*nay
        npix_y = highy - lowy
        # import ipdb; ipdb.set_trace()
        a, b, c = alpha[i]
        # diagonal
        alpha3[lowx:highx, lowy:highy] = c
        # upper right
        alpha3[lowx2:lowx2+npix_x, lowy:highy] = b
        # lower left
        alpha3[lowx:highx, lowy2:lowy2+npix_y] = a

    assert_array_almost_equal(alpha2.T, alpha3)
