"""Tests for wavelet optimization correctness and Psi operator properties."""

import numpy as np
import pytest
import pywt
from numpy.testing import assert_array_almost_equal

from pfb_imaging.operators.psi import Psi, PsiBasis
from pfb_imaging.wavelets import coeff_size, dwt2d, dwt2d_seq, idwt2d, idwt2d_seq, signal_size

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
    ntotx += cx
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


# --- Approx buffer correctness ---


@pmp("wavelet", ["db1", "db4", "db5"])
@pmp("data_shape", [(128, 256), (512, 128), (64, 64)])
@pmp("nlevel", [1, 2, 3])
def test_approx_buffer_roundtrip(wavelet, data_shape, nlevel):
    """Verify pre-allocated approx buffer gives exact round-trip reconstruction."""
    nxi, nyi = data_shape
    max_level = pywt.dwt_max_level(min(nxi, nyi), wavelet)
    if nlevel > max_level:
        pytest.skip(f"nlevel {nlevel} > max_level {max_level} for {wavelet}")

    np.random.seed(42)
    data = np.random.randn(*data_shape)

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_wavelet_arrays(nxi, nyi, wavelet, nlevel)

    alpha = np.zeros((ntoty, ntotx))
    cbuff_flat = np.zeros(ntotx * ntoty)
    cbufft_flat = np.zeros(ntoty * ntotx)
    approx = np.zeros((ntotx, ntoty))

    dwt2d(data, alpha, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)

    xrec = np.zeros((nxi, nyi))
    alpha_buf = np.zeros((ntoty, ntotx))
    idwt2d(alpha, xrec, alpha_buf, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    assert_array_almost_equal(data, xrec, decimal=10)


# --- Psi adjoint test ---


@pmp("nx", [64, 128])
@pmp("ny", [48, 96])
@pmp("nlevels", [1, 2])
def test_psi_adjoint(nx, ny, nlevels):
    """Test adjoint property: <Psi x, alpha> == <x, Psi^H alpha>."""
    np.random.seed(123)
    nband = 1
    bases = ["self", "db1", "db3"]
    nbasis = len(bases)

    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha_rand = np.random.randn(nband, nbasis, nymax, nxmax)

    # forward: Psi x
    alpha_out = np.zeros((nband, nbasis, nymax, nxmax))
    psi.dot(x, alpha_out)

    # adjoint: Psi^H alpha_rand
    x_out = np.zeros((nband, nx, ny))
    psi.hdot(alpha_rand, x_out)

    # <Psi x, alpha_rand> should equal <x, Psi^H alpha_rand>
    lhs = np.sum(alpha_out * alpha_rand)
    rhs = np.sum(x * x_out)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# --- Psi multi-band consistency ---


@pmp("nband", [2, 4])
@pmp("nlevels", [1, 2])
def test_psi_multiband_consistency(nband, nlevels):
    """Multi-band Psi should give same results as single-band applied independently."""
    np.random.seed(456)
    nx, ny = 64, 48
    bases = ["db1", "db3"]
    nbasis = len(bases)

    psi_multi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi_multi.nxmax
    nymax = psi_multi.nymax

    x = np.random.randn(nband, nx, ny)

    # multi-band forward
    alpha_multi = np.zeros((nband, nbasis, nymax, nxmax))
    psi_multi.dot(x, alpha_multi)

    # single-band forward per band
    for b in range(nband):
        psi_single = Psi(1, nx, ny, bases, nlevels, 1)
        alpha_single = np.zeros((1, nbasis, nymax, nxmax))
        psi_single.dot(x[b : b + 1], alpha_single)
        assert_array_almost_equal(alpha_multi[b], alpha_single[0], decimal=12)


# --- Self basis test ---


@pmp("nx", [32, 64])
@pmp("ny", [48, 32])
def test_self_basis_roundtrip(nx, ny):
    """The 'self' (identity) basis should round-trip exactly."""
    np.random.seed(789)
    nband = 1
    bases = ["self"]
    nbasis = 1
    nlevels = 1  # not used for self basis

    psi = Psi(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    # nbasis=1 so no scaling needed
    assert_array_almost_equal(x, xrec, decimal=12)


# --- Minimum size input ---


@pmp("wavelet", ["db1", "db2"])
def test_minimum_size_input(wavelet):
    """Test with smallest possible input sizes."""
    np.random.seed(101)
    nband = 1
    nlevel = 1
    bases = [wavelet]
    nbasis = 1

    # minimum size for a 1-level decomposition
    filter_length = int(wavelet[-1]) * 2
    nx = filter_length + 2
    ny = filter_length + 2

    psi = Psi(nband, nx, ny, bases, nlevel, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    assert_array_almost_equal(x, xrec, decimal=10)


# --- Highly non-square inputs ---


@pmp("data_shape", [(16, 256), (256, 16)])
@pmp("wavelet", ["db1", "db4"])
def test_nonsquare_input(data_shape, wavelet):
    """Test with highly non-square inputs."""
    np.random.seed(202)
    nband = 1
    nlevel = 1
    bases = [wavelet]
    nbasis = 1
    nx, ny = data_shape

    psi = Psi(nband, nx, ny, bases, nlevel, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    assert_array_almost_equal(x, xrec, decimal=10)


# --- Fastmath numerical accuracy ---


@pmp("wavelet", ["db1", "db4", "db5"])
@pmp("data_shape", [(128, 128), (256, 128)])
def test_fastmath_accuracy_vs_pywt(wavelet, data_shape):
    """Verify fastmath-enabled convolutions match PyWavelets to reasonable precision."""
    np.random.seed(303)
    nxi, nyi = data_shape
    nlevel = 2
    data = np.random.randn(*data_shape)

    # pywt reference
    alpha_pywt = pywt.wavedec2(data, wavelet, mode="zero", level=nlevel)
    xrec_pywt = pywt.waverec2(alpha_pywt, wavelet, mode="zero")

    # our implementation
    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_wavelet_arrays(nxi, nyi, wavelet, nlevel)

    alpha = np.zeros((ntoty, ntotx))
    cbuff_flat = np.zeros(ntotx * ntoty)
    cbufft_flat = np.zeros(ntoty * ntotx)
    approx = np.zeros((ntotx, ntoty))

    dwt2d(data, alpha, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)

    xrec = np.zeros((nxi, nyi))
    alpha_buf = np.zeros((ntoty, ntotx))
    idwt2d(alpha, xrec, alpha_buf, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    # round-trip should be exact (within floating point)
    assert_array_almost_equal(data, xrec, decimal=10)
    # pywt round-trip should also match
    assert_array_almost_equal(data, xrec_pywt[0:nxi, 0:nyi], decimal=10)


# --- PsiBand with single wavelet basis ---


@pmp("wavelet", ["db1", "db4"])
@pmp("nlevel", [1, 2, 3])
def test_psiband_single_wavelet(wavelet, nlevel):
    """Test PsiBand with a single wavelet basis (no 'self')."""
    np.random.seed(404)
    nx, ny = 64, 48
    max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
    if nlevel > max_level:
        pytest.skip(f"nlevel {nlevel} > max_level {max_level} for {wavelet}")

    nband = 1
    bases = [wavelet]
    nbasis = 1

    psi = Psi(nband, nx, ny, bases, nlevel, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    assert_array_almost_equal(x, xrec, decimal=10)


# --- Serial dwt2d_seq / idwt2d_seq round-trip ---


@pmp("wavelet", ["db1", "db4", "db5"])
@pmp("data_shape", [(128, 256), (64, 64)])
@pmp("nlevel", [1, 2, 3])
def test_seq_roundtrip(wavelet, data_shape, nlevel):
    """Verify serial dwt2d_seq/idwt2d_seq gives exact round-trip."""
    nxi, nyi = data_shape
    max_level = pywt.dwt_max_level(min(nxi, nyi), wavelet)
    if nlevel > max_level:
        pytest.skip(f"nlevel {nlevel} > max_level {max_level} for {wavelet}")

    np.random.seed(42)
    data = np.random.randn(*data_shape)

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])
    rec_lo = np.array(wvlt.filter_bank[2])
    rec_hi = np.array(wvlt.filter_bank[3])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_wavelet_arrays(nxi, nyi, wavelet, nlevel)

    alpha = np.zeros((ntoty, ntotx))
    cbuff_flat = np.zeros(ntotx * ntoty)
    cbufft_flat = np.zeros(ntoty * ntotx)
    approx = np.zeros((ntotx, ntoty))

    dwt2d_seq(data, alpha, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)

    xrec = np.zeros((nxi, nyi))
    alpha_buf = np.zeros((ntoty, ntotx))
    idwt2d_seq(alpha, xrec, alpha_buf, cbuff_flat, cbufft_flat, ix, iy, sx, sy, spx, spy, rec_lo, rec_hi, nlevel)

    assert_array_almost_equal(data, xrec, decimal=10)


# --- Serial vs parallel DWT consistency ---


@pmp("wavelet", ["db1", "db4"])
@pmp("data_shape", [(128, 128)])
def test_seq_matches_parallel(wavelet, data_shape):
    """Serial and parallel DWT should produce identical coefficients."""
    np.random.seed(55)
    nxi, nyi = data_shape
    nlevel = 2
    data = np.random.randn(*data_shape)

    wvlt = pywt.Wavelet(wavelet)
    dec_lo = np.array(wvlt.filter_bank[0])
    dec_hi = np.array(wvlt.filter_bank[1])

    ix, iy, sx, sy, spx, spy, ntotx, ntoty = build_wavelet_arrays(nxi, nyi, wavelet, nlevel)

    # parallel version
    alpha_par = np.zeros((ntoty, ntotx))
    cbuff_flat = np.zeros(ntotx * ntoty)
    cbufft_flat = np.zeros(ntoty * ntotx)
    approx = np.zeros((ntotx, ntoty))
    dwt2d(data, alpha_par, cbuff_flat, cbufft_flat, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx)

    # serial version
    alpha_seq = np.zeros((ntoty, ntotx))
    cbuff_flat2 = np.zeros(ntotx * ntoty)
    cbufft_flat2 = np.zeros(ntoty * ntotx)
    approx2 = np.zeros((ntotx, ntoty))
    dwt2d_seq(data, alpha_seq, cbuff_flat2, cbufft_flat2, ix, iy, sx, sy, dec_lo, dec_hi, nlevel, approx2)

    assert_array_almost_equal(alpha_par, alpha_seq, decimal=12)


# --- PsiBasis round-trip ---


@pmp("wavelet", ["db1", "db4"])
@pmp("nlevel", [1, 2, 3])
def test_psi_basis_single_wavelet(wavelet, nlevel):
    """PsiBasis with single wavelet basis round-trips exactly."""
    np.random.seed(500)
    nx, ny = 64, 48
    max_level = pywt.dwt_max_level(min(nx, ny), wavelet)
    if nlevel > max_level:
        pytest.skip(f"nlevel {nlevel} > max_level {max_level} for {wavelet}")

    nband = 1
    bases = [wavelet]
    nbasis = 1

    psi = PsiBasis(nband, nx, ny, bases, nlevel, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    assert_array_almost_equal(x, xrec, decimal=10)


# --- PsiBasis adjoint ---


@pmp("nx", [64, 128])
@pmp("ny", [48, 96])
@pmp("nlevels", [1, 2])
def test_psi_basis_adjoint(nx, ny, nlevels):
    """PsiBasis adjoint property: <Psi x, alpha> == <x, Psi^H alpha>."""
    np.random.seed(600)
    nband = 1
    bases = ["self", "db1", "db3"]
    nbasis = len(bases)

    psi = PsiBasis(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha_rand = np.random.randn(nband, nbasis, nymax, nxmax)

    alpha_out = np.zeros((nband, nbasis, nymax, nxmax))
    psi.dot(x, alpha_out)

    x_out = np.zeros((nband, nx, ny))
    psi.hdot(alpha_rand, x_out)

    lhs = np.sum(alpha_out * alpha_rand)
    rhs = np.sum(x * x_out)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# --- PsiBasis matches Psi (PsiBand) ---


@pmp("nx", [64])
@pmp("ny", [48])
@pmp("nlevels", [1, 2])
def test_psi_basis_matches_psi_band(nx, ny, nlevels):
    """PsiBasis produces identical results to PsiBand (Psi)."""
    np.random.seed(700)
    nband = 1
    bases = ["self", "db1", "db4"]
    nbasis = len(bases)

    psi_band = Psi(nband, nx, ny, bases, nlevels, 1)
    psi_basis = PsiBasis(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi_band.nxmax
    nymax = psi_band.nymax

    x = np.random.randn(nband, nx, ny)

    # forward
    alpha_band = np.zeros((nband, nbasis, nymax, nxmax))
    alpha_basis = np.zeros((nband, nbasis, nymax, nxmax))
    psi_band.dot(x, alpha_band)
    psi_basis.dot(x, alpha_basis)
    assert_array_almost_equal(alpha_band, alpha_basis, decimal=12)

    # adjoint
    xo_band = np.zeros((nband, nx, ny))
    xo_basis = np.zeros((nband, nx, ny))
    psi_band.hdot(alpha_band, xo_band)
    psi_basis.hdot(alpha_band, xo_basis)
    assert_array_almost_equal(xo_band, xo_basis, decimal=12)


# --- PsiBasis self-only ---


@pmp("nx", [32, 64])
@pmp("ny", [48, 32])
def test_psi_basis_self_only(nx, ny):
    """PsiBasis with only 'self' basis round-trips exactly."""
    np.random.seed(800)
    nband = 1
    bases = ["self"]
    nbasis = 1
    nlevels = 1

    psi = PsiBasis(nband, nx, ny, bases, nlevels, 1)
    nxmax = psi.nxmax
    nymax = psi.nymax

    x = np.random.randn(nband, nx, ny)
    alpha = np.zeros((nband, nbasis, nymax, nxmax))
    xrec = np.zeros((nband, nx, ny))

    psi.dot(x, alpha)
    psi.hdot(alpha, xrec)

    assert_array_almost_equal(x, xrec, decimal=12)
