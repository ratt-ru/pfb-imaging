import numpy as np
import pytest
from africanus.model.spi import fit_spi_components
from astropy.modeling.functional_models import Gaussian2D
from numpy.testing import assert_allclose

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


@pmp("sidelobe_amp", [0.0, 0.3])
def test_init_rotated_bbox_vs_axis_aligned(sidelobe_amp):
    """Compare the rotated bounding box initialization against the old
    axis-aligned bounding box method. Both methods share the same PA
    estimate from weighted second moments; the difference is how
    emaj0/emin0 are derived from the main lobe shape.

    We test on clean Gaussians and on Gaussians with synthetic sidelobes
    to simulate a realistic PSF. The rotated method should produce lower
    total error across a sweep of position angles and eccentricities.
    """
    from scipy.ndimage import label

    nx = 128
    x = -(nx // 2) + np.arange(nx)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    level = 0.5

    # place here instead of using pmp so we can look at aggregate error across all cases
    gpars_list = [
        (10.0, 5.0, 0.0),
        (7.0, 4.0, 1.0),
        (7.5, 3.0, 2.0),
        (7.0, 1.1, 3.0),
        (8.0, 2.0, 0.5),
        (9.0, 3.0, 1.5),
        (6.0, 1.5, 2.5),
        (10.0, 2.0, 0.8),
    ]

    total_err_old = 0.0
    total_err_new = 0.0

    for gpars in gpars_list:
        emaj, emin, pa = gpars
        gauss = gaussian2d(xx, yy, gausspar=gpars, normalise=False)

        if sidelobe_amp > 0:
            rr = np.sqrt(xx**2 + yy**2)
            ring_radius = 1.5 * emaj / fwhm_conv
            ring = sidelobe_amp * np.exp(-0.5 * ((rr - ring_radius) / 1.0) ** 2)
            gauss = gauss + ring
            gauss /= gauss.max()

        psfv = gauss / gauss.max()
        mask = np.where(psfv > level, 1.0, 0)
        islands, _ = label(mask)
        ncenter = islands[nx // 2, nx // 2]

        xl = xx[islands == ncenter]
        yl = yy[islands == ncenter]
        psftmp = psfv[islands == ncenter]
        wsum = psftmp.sum()
        dxl = xl - np.sum(psftmp * xl) / wsum
        dyl = yl - np.sum(psftmp * yl) / wsum
        mxx = np.sum(psftmp * dxl**2) / wsum
        myy = np.sum(psftmp * dyl**2) / wsum
        mxy = np.sum(psftmp * dxl * dyl) / wsum
        pa0 = np.pi / 2 + 0.5 * np.arctan2(2 * mxy, mxx - myy)
        pa0 = float(np.clip(pa0, 0.0, np.pi))

        # old method: axis-aligned bounding box
        xdiff_old = np.maximum(yl.max() - yl.min(), 1)
        ydiff_old = np.maximum(xl.max() - xl.min(), 1)
        if xdiff_old > ydiff_old:
            emaj_old, emin_old = xdiff_old, ydiff_old
        else:
            emaj_old, emin_old = ydiff_old, xdiff_old

        # new method: rotated bounding box
        t = np.pi / 2 + pa0
        ct, st = np.cos(t), np.sin(t)
        dx_rot = ct * dxl + st * dyl
        dy_rot = -st * dxl + ct * dyl
        emaj_new = np.maximum(dx_rot.max() - dx_rot.min(), 1.0)
        emin_new = np.maximum(dy_rot.max() - dy_rot.min(), 1.0)

        total_err_old += (emaj_old - emaj) ** 2 + (emin_old - emin) ** 2
        total_err_new += (emaj_new - emaj) ** 2 + (emin_new - emin) ** 2

    assert total_err_new < total_err_old, (
        f"rotated bbox total SSE ({total_err_new:.4f}) not better than "
        f"axis-aligned ({total_err_old:.4f}), sidelobe_amp={sidelobe_amp}"
    )


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
