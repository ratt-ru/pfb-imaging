import numpy as np
import pytest
from africanus.constants import c as lightspeed
from daskms import xds_from_ms, xds_from_table
from ducc0.fft import good_size
from numpy.testing import assert_allclose

from pfb_imaging.utils.misc import gaussian2d, give_edges
from pfb_imaging.utils.modelspec import eval_coeffs_to_cube, eval_coeffs_to_slice, fit_image_cube

pmp = pytest.mark.parametrize


def test_model2comps(ms_name):
    """
    TODO - This only tests the separate function implementations
    not the workers. Tested by spotless workflow?
    """
    np.random.seed(420)
    xds = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1})[0]
    spw = xds_from_table(f"{ms_name}::SPECTRAL_WINDOW")[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    nchan = freq.size

    uvw = xds.UVW.values
    max_blength = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max()

    # image size
    cell_n = 1.0 / (2 * max_blength * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_n / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    # the test will fail in intrinsic if sources fall near beam sidelobes
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    print("Image size set to (%i, %i, %i)" % (nchan, nx, ny))

    # model
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    nsource = 25
    border = np.maximum(int(0.15 * nx), int(0.15 * ny))
    x_index = np.random.randint(border, npix - border, nsource)
    y_index = np.random.randint(border, npix - border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    ref_flux = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1 * nx), nsource)
    extenty = np.random.randint(3, int(0.1 * nx), nsource)
    pas = np.random.random(nsource) * 180
    x = -(nx / 2) + np.arange(nx)
    y = -(nx / 2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing="ij")
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = gaussian2d(xin, yin, gausspar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(x_index[i], y_index[i], nx, ny, nx, ny)
        spectrum = ref_flux[i] * (freq / freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]

    mfreqs = freq
    mtimes = utime[0:1]  # arbitrary for now

    coeffs, x_index, y_index, expr, params, tfunc, ffunc = fit_image_cube(
        mtimes, mfreqs, model[None, :, :, :], nbasisf=nchan, sigmasq=0.0, method="Legendre"
    )
    image = eval_coeffs_to_cube(mtimes, mfreqs, nx, ny, coeffs, x_index, y_index, expr, params, tfunc, ffunc)

    image = image[0]  # no time axis for now
    mask = model > 0
    assert_allclose(image[mask], model[mask], atol=1e-10)

    # test spatial interpolation
    # if we shift the center by an integer number of pixels
    # the pixel centers should stay the same
    xshift = 25
    x0 = cell_deg * xshift
    yshift = -10
    y0 = cell_deg * yshift
    for npix in [100, nx, 2 * nx]:
        imout = eval_coeffs_to_slice(
            mtimes[0],
            mfreqs[0],
            coeffs,
            x_index,
            y_index,
            expr,
            params,
            tfunc,
            ffunc,
            nx,
            ny,
            cell_deg,
            cell_deg,
            0.0,
            0.0,
            npix,
            npix,
            cell_deg,
            cell_deg,
            x0,
            y0,
        )

        mx, my, gx, gy = give_edges(npix // 2 - xshift, npix // 2 - yshift, npix, npix, nx, ny)

        assert_allclose(1.0 + imout[mx, my], 1.0 + model[0, gx, gy])
