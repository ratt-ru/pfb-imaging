import numpy as np
import pytest
from africanus.model.spi import fit_spi_components
from astropy.modeling.functional_models import Gaussian2D
from numpy.testing._private.utils import assert_allclose

from pfb_imaging.utils.misc import convolve2gaussres, fitcleanbeam, gaussian2d

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
        restored[v] = gaussian2d(xx, yy, gausspari[v], normalise=False) * (freq[v] / ref_freq) ** alpha

        conv_model[v] = convolve2gaussres(
            restored[v][None], xx, yy, gausspari[0], nthreads=2, gausspari=(gausspari[v],)
        )

    x_index, y_index = np.where(conv_model[-1] > 0.05)

    comps = conv_model[:, x_index, y_index]
    weights = np.ones((nband))

    out = fit_spi_components(comps.T, weights, freq, ref_freq, tol=1e-7, maxiter=250)

    # offset for relative difference
    assert_allclose(1 + alpha, 1 + out[0, :], atol=5e-4, rtol=5e-4)
    assert_allclose(out[2, :], restored[0, x_index, y_index], atol=5e-4, rtol=5e-4)


@pmp("nx", [128, 256])
@pmp("ny", [128, 256])
@pmp("gpars", [(10.0, 5.0, 0.0), (7.0, 4.0, 1.0), (7.5, 3.0, 2.0), (7.0, 1.1, 3.0)])
def test_gaussian_pfb_vs_astropy(nx, ny, gpars):
    emaj, emin, pa = gpars

    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # gpars = (emaj, emin, pa), where emaj and emin are the expected FWHM values of the major
    # and minor axes respectively. pa is the position angle in radians and is given as an
    # anticlockwise rotation of the major axis from the positive vertical axis. These are
    # consistent with the manner in which the beam information is stored in FITS files.
    pfb_gauss = gaussian2d(xx, yy, gausspar=gpars, normalise=False)

    # Astropy uses a conventional gaussian formulation i.e. the Gaussian is parameterised in terms
    # of the standard deviation along the x and y axes, and a position angle. We convert the FWHM
    # values using the standard formula sigma = FWHM / k where k = 2 * np.sqrt(2 * np.log(2)).
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    sigma_maj = emaj / fwhm_conv
    sigma_min = emin / fwhm_conv

    # The position angle in astropy follows the standard conventions i.e. it is an anti-clockwise
    # rotation of the major axis from the positive horizontal axis. The discrepancy between the
    # two conventions means we need to add pi / 2 to shift the major axis to align with the
    # positive vertical axis, then add the pa such that the rotation is anti-clockwise from the
    # positive vertical axis.
    theta = np.pi / 2 + pa

    astropy_gauss = Gaussian2D(
        amplitude=1.0,
        x_mean=0.0,
        y_mean=0.0,
        x_stddev=sigma_maj,
        y_stddev=sigma_min,
        theta=theta,
    )(xx, yy)

    # Compare where signal is significant
    mask = pfb_gauss > 1e-12
    assert mask.any(), "No significant values in gaussian2d output"
    assert_allclose(pfb_gauss[mask], astropy_gauss[mask], atol=1e-12, rtol=1e-12)


@pmp("nx", [128, 256])
@pmp("ny", [128, 256])
@pmp("gpars", [(10.0, 5.0, 0.0), (7.0, 4.0, 1.0), (7.5, 3.0, 2.0), (7.0, 1.1, 3.0)])
def test_fitcleanbeam_vs_astropy(nx, ny, gpars):
    """Fit an astropy Gaussian2D with fitcleanbeam and check parameter recovery."""
    emaj, emin, pa = gpars

    # Convert (emaj, emin, pa) to astropy parameters - see test_gaussian_pfb_vs_astropy.
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    sigma_maj = emaj / fwhm_conv
    sigma_min = emin / fwhm_conv
    theta = np.pi / 2 + pa

    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    gauss = Gaussian2D(
        amplitude=1.0,
        x_mean=0.0,
        y_mean=0.0,
        x_stddev=sigma_maj,
        y_stddev=sigma_min,
        theta=theta,
    )(xx, yy)

    gpars_fit = fitcleanbeam(gauss[None, :, :])[0]

    assert np.abs(emaj - gpars_fit[0]) < 1e-4
    assert np.abs(emin - gpars_fit[1]) < 1e-4
    padiff = np.abs(pa - gpars_fit[2])
    assert np.sin(padiff) < 1e-4


@pmp("nx", [128, 256])
@pmp("ny", [80, 220])
@pmp("gpars", [(10.0, 5.0, 0.0), (7.0, 4.0, 1.0), (7.5, 3.0, 2.0), (7.0, 1.1, 3.0)])
def test_fitcleanbeam(nx, ny, gpars):
    # generate a grid
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gauss = gaussian2d(xx, yy, gausspar=gpars, normalise=False)

    gpars_fit = fitcleanbeam(gauss[None, :, :])[0]

    assert np.abs(gpars[0] - gpars_fit[0]) < 1e-4
    assert np.abs(gpars[1] - gpars_fit[1]) < 1e-4
    # if 0 and pi are equivalent
    padiff = np.abs(gpars[2] - gpars_fit[2])
    assert np.sin(padiff) < 1e-4
