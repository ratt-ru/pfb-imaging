import numpy as np
from africanus.model.spi import fit_spi_components
from numpy.testing._private.utils import assert_allclose
from pfb.utils.misc import convolve2gaussres, Gaussian2D
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [128])
@pmp("ny", [80, 220])
@pmp("nband", [4, 8])
@pmp("alpha", [-0.5, 0.0, 0.5])
def test_convolve2gaussres(nx, ny, nband, alpha):
    np.random.seed(420)
    freq = np.linspace(0.5e9, 1.5e9, nband)
    ref_freq = freq[0]

    Gausspari = ()
    es = np.linspace(15, 5, nband)
    for v in range(nband):
        Gausspari += ((es[v], es[v], 0.0),)

    x = np.arange(-nx/2, nx/2)
    y = np.arange(-ny/2, ny/2)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    restored = np.zeros((nband, nx, ny))
    conv_model = np.zeros((nband, nx, ny))
    for v in range(nband):
        restored[v] = Gaussian2D(xx, yy, Gausspari[v],
                                 normalise=False) * (freq[v]/ref_freq)**alpha

        conv_model[v] = convolve2gaussres(restored[v], xx, yy, Gausspari[0], 8,
                                          gausspari=Gausspari[v])

    Ix, Iy = np.where(conv_model[-1] > 0.05)

    comps = conv_model[:, Ix, Iy]
    weights = np.ones((nband))

    out = fit_spi_components(comps.T, weights, freq, ref_freq,
                             tol=1e-7, maxiter=250)

    # offset for relative difference
    assert_allclose(1+alpha, 1+out[0, :], atol=5e-4, rtol=5e-4)
    assert_allclose(out[2, :], restored[0, Ix, Iy], atol=5e-4, rtol=5e-4)
