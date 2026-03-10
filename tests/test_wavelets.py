import numpy as np
import pytest
import pywt
from numpy.testing import assert_array_almost_equal

from pfb_imaging.wavelets import coeff_size, dwt2d, idwt2d, signal_size

pmp = pytest.mark.parametrize


def build_wavelet_arrays(nxi, nyi, wavelet, nlevel):
    """Build array-based bookkeeping for dwt2d/idwt2d."""
    n2cx = {}
    n2cy = {}
    nx = nxi
    ny = nyi
    ntotx = 0
    ntoty = 0
    sx = np.zeros(nlevel, dtype=np.int64)
    sy = np.zeros(nlevel, dtype=np.int64)
    spx = np.zeros(nlevel, dtype=np.int64)
    spy = np.zeros(nlevel, dtype=np.int64)
    filter_length = int(wavelet[-1]) * 2
    for k in range(nlevel):
        cx = coeff_size(nx, filter_length)
        cy = coeff_size(ny, filter_length)
        n2cx[k] = (signal_size(cx, filter_length), cx)
        n2cy[k] = (signal_size(cy, filter_length), cy)
        ntotx += cx
        ntoty += cy
        sx[k] = cx
        sy[k] = cy
        nx = cx + cx % 2
        ny = cy + cy % 2
        spx[k] = signal_size(cx, filter_length)
        spy[k] = signal_size(cy, filter_length)
    ntotx += cx  # last approx coeffs
    ntoty += cy

    ix = np.zeros((nlevel, 2), dtype=np.int64)
    iy = np.zeros((nlevel, 2), dtype=np.int64)
    lowx = n2cx[nlevel - 1][1]
    lowy = n2cy[nlevel - 1][1]
    ix[nlevel - 1, 0] = lowx
    ix[nlevel - 1, 1] = 2 * lowx
    iy[nlevel - 1, 0] = lowy
    iy[nlevel - 1, 1] = 2 * lowy
    lowx *= 2
    lowy *= 2
    for k in reversed(range(nlevel - 1)):
        highx = n2cx[k][1]
        highy = n2cy[k][1]
        ix[k, 0] = lowx
        ix[k, 1] = lowx + highx
        iy[k, 0] = lowy
        iy[k, 1] = lowy + highy
        lowx += highx
        lowy += highy

    return ix, iy, sx, sy, spx, spy, ntotx, ntoty


# TODO - return shape is currently incorrect for certain odd valued inputs
# https://github.com/PyWavelets/pywt/issues/725
@pmp("wavelet", ["db1", "db4", "db5"])
@pmp("data_shape", [(128, 256), (512, 128)])  # , (129,255), (511,257)])
@pmp("nlevel", [1, 2, 3])
def test_dwt_idwt_pywt(wavelet, data_shape, nlevel):
    nxi, nyi = data_shape
    data = np.random.random(size=data_shape)

    # pywt comparison
    alpha = pywt.wavedec2(data, wavelet, mode="zero", level=nlevel)
    xrec = pywt.waverec2(alpha, wavelet, mode="zero")
    assert_array_almost_equal(data, xrec[0 : data.shape[0], 0 : data.shape[1]])

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_wavelet_arrays(nxi, nyi, wavelet, nlevel)

    alpha2 = np.zeros((ntoty, ntotx))
    cbuff_flat = np.zeros(ntotx * ntoty)
    cbufft_flat = np.zeros(ntoty * ntotx)
    approx = np.zeros((ntotx, ntoty))
    dwt2d(data, alpha2, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)
    xrec2 = np.zeros((nxi, nyi))
    coeffs = np.zeros((ntoty, ntotx))
    idwt2d(alpha2, xrec2, coeffs, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    assert_array_almost_equal(data, xrec2)

    # pack pywt into array
    alpha3 = np.zeros_like(alpha2.T)
    nx = ix[nlevel - 1, 0]
    ny = iy[nlevel - 1, 0]
    alpha3[0:nx, 0:ny] = alpha[0]
    for i, j in zip(range(1, nlevel + 1), reversed(range(nlevel))):
        # bottom left
        lowx = ix[j, 0]
        highx = ix[j, 1]
        nax = sx[j]
        lowx2 = highx - 2 * nax
        npix_x = highx - lowx
        lowy = iy[j, 0]
        highy = iy[j, 1]
        nay = sy[j]
        lowy2 = highy - 2 * nay
        npix_y = highy - lowy
        a, b, c = alpha[i]
        # diagonal
        alpha3[lowx:highx, lowy:highy] = c
        # upper right
        alpha3[lowx2 : lowx2 + npix_x, lowy:highy] = b
        # lower left
        alpha3[lowx:highx, lowy2 : lowy2 + npix_y] = a

    assert_array_almost_equal(alpha2.T, alpha3)
