import pytest
from pathlib import Path
pmp = pytest.mark.parametrize


def test_model2comps(ms_name):
    '''
    TODO - This only tests the separate function implementations
    not the workers. Tested by spotless workflow?
    '''

    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_ms, xds_from_table
    from pfb.utils.misc import Gaussian2D, give_edges
    from africanus.constants import c as lightspeed
    from pfb.utils.misc import (fit_image_cube,
                                eval_coeffs_to_cube,
                                eval_coeffs_to_slice)

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA1.values.max()) + 1

    ncorr = xds.corr.size

    uvw = xds.UVW.values
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)

    from ducc0.fft import good_size
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
    border = np.maximum(int(0.15*nx), int(0.15*ny))
    Ix = np.random.randint(border, npix-border, nsource)
    Iy = np.random.randint(border, npix-border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    I0 = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1*nx), nsource)
    extenty = np.random.randint(3, int(0.1*nx), nsource)
    pas = np.random.random(nsource) * 180
    x = -(nx/2) + np.arange(nx)
    y = -(nx/2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing='ij')
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = Gaussian2D(xin, yin, GaussPar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(Ix[i], Iy[i], nx, ny, nx, ny)
        spectrum = I0[i] * (freq/freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]

    mfreqs = freq
    mtimes = utime[0:1]  # arbitrary for now



    coeffs, Ix, Iy, expr, params, tfunc, ffunc = \
        fit_image_cube(mtimes, mfreqs, model[None, :, :, :], nbasisf=nchan,
                       sigmasq=0.0, method='Legendre')
    image = eval_coeffs_to_cube(mtimes, mfreqs, nx, ny, coeffs, Ix, Iy,
                                expr, params, tfunc, ffunc)

    image = image[0]  # no time axis for now
    mask = model > 0
    assert_allclose(image[mask], model[mask], atol=1e-10)

    # test spatial interpolation
    # if we shift the center by an integer number of pixels
    # the pixel centers should stay the same
    xshift = 25
    x0 = cell_deg*xshift
    yshift = -10
    y0 = cell_deg*yshift
    for npix in [100, nx, 2*nx]:
        imout = eval_coeffs_to_slice(mtimes[0],
                                    mfreqs[0],
                                    coeffs,
                                    Ix, Iy,
                                    expr,
                                    params,
                                    tfunc,
                                    ffunc,
                                    nx, ny,
                                    cell_deg, cell_deg,
                                    0.0, 0.0,
                                    npix, npix,
                                    cell_deg, cell_deg,
                                    x0, y0)

        mx, my, gx, gy = give_edges(npix//2 - xshift,
                                    npix//2 - yshift, npix, npix, nx, ny)

        assert_allclose(1.0 + imout[mx, my], 1.0 + model[0, gx, gy])

