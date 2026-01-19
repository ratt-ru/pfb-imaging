import numpy as np
import pytest
from africanus.model.spi import fit_spi_components
from numpy.testing._private.utils import assert_allclose

from pfb_imaging.utils.misc import Gaussian2D, convolve2gaussres, fitcleanbeam

pmp = pytest.mark.parametrize


@pmp("nx", [128])
@pmp("ny", [80, 220])
@pmp("nband", [4, 8])
@pmp("alpha", [-0.5, 0.0, 0.5])
def test_convolve2gaussres(nx, ny, nband, alpha):
    np.random.seed(420)
    freq = np.linspace(0.5e9, 1.5e9, nband)
    ref_freq = freq[0]

    gausspari = ()
    es = np.linspace(15, 5, nband)
    for v in range(nband):
        gausspari += ((es[v], es[v], 0.0),)

    x = np.arange(-nx / 2, nx / 2)
    y = np.arange(-ny / 2, ny / 2)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    restored = np.zeros((nband, nx, ny))
    conv_model = np.zeros((nband, nx, ny))
    for v in range(nband):
        restored[v] = Gaussian2D(xx, yy, gausspari[v], normalise=False) * (freq[v] / ref_freq) ** alpha

        conv_model[v] = convolve2gaussres(
            restored[v][None], xx, yy, gausspari[0], nthreads=8, gausspari=(gausspari[v],)
        )

    x_index, y_index = np.where(conv_model[-1] > 0.05)

    comps = conv_model[:, x_index, y_index]
    weights = np.ones((nband))

    out = fit_spi_components(comps.T, weights, freq, ref_freq, tol=1e-7, maxiter=250)

    # offset for relative difference
    assert_allclose(1 + alpha, 1 + out[0, :], atol=5e-4, rtol=5e-4)
    assert_allclose(out[2, :], restored[0, x_index, y_index], atol=5e-4, rtol=5e-4)


@pmp("nx", [128, 256])
@pmp("ny", [80, 220])
@pmp("gpars", [(10.0, 5.0, 0.0), (7.0, 4.0, 1.0), (7.5, 3.0, 2.0), (7.0, 1.1, 3.0)])
def test_fitcleanbeam(nx, ny, gpars):
    # generate a grid
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    print(gpars)
    gauss = Gaussian2D(xx, yy, GaussPar=gpars, normalise=False)

    gpars_fit = fitcleanbeam(gauss[None, :, :])[0]

    assert np.abs(gpars[0] - gpars_fit[0]) < 1e-4
    assert np.abs(gpars[1] - gpars_fit[1]) < 1e-4
    # if 0 and pi are equivalent
    padiff = np.abs(gpars[2] - gpars_fit[2])
    assert np.sin(padiff) < 1e-4
