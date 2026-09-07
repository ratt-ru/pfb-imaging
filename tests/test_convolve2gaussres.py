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


def test_convolve2gaussres_yx_order_matches_manual_transpose():
    """yx_order=True must equal transposing to x-major, convolving, and
    transposing back. gaussian2d's PA is not transpose-invariant, so the
    .dt's (corr, y, x) arrays (wiki D19/D20) cannot be fed in directly.
    """
    from pfb_imaging.utils.misc import convolve2gaussres

    nx, ny = 64, 48
    img_xy = np.zeros((1, nx, ny))
    img_xy[0, nx // 2 + 5, ny // 2 - 3] = 1.0
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gpar = np.array([[6.0, 3.0, 0.7]])

    ref = convolve2gaussres(img_xy, xx, yy, gpar, nthreads=1, pfrac=0.2)
    got = convolve2gaussres(img_xy.transpose(0, 2, 1), xx, yy, gpar, nthreads=1, pfrac=0.2, yx_order=True)

    assert got.shape == (1, ny, nx)
    np.testing.assert_allclose(got, ref.transpose(0, 2, 1), rtol=0, atol=1e-10)


def test_convolve2gaussres_yx_order_is_not_a_noop():
    """On a square image the missing transpose does not raise, it silently
    mirrors the position angle. Guard that the flag actually does something.
    """
    from pfb_imaging.utils.misc import convolve2gaussres

    n = 64
    img = np.zeros((1, n, n))
    img[0, n // 2 + 5, n // 2 - 3] = 1.0
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    gpar = np.array([[8.0, 3.0, 0.6]])  # elongated, PA off the axes

    naive = convolve2gaussres(img, xx, yy, gpar, nthreads=1, pfrac=0.2)
    yx = convolve2gaussres(img, xx, yy, gpar, nthreads=1, pfrac=0.2, yx_order=True)

    assert not np.allclose(naive, yx)


def test_convolve2gaussres_per_plane_gausspar_for_three_planes():
    """A (3, 3) gaussparf is three per-plane triples, not one triple.
    The old `len(gaussparf) == 3` test took the single-resolution branch for
    any 3-plane input -- reachable via restore with 3 correlations.
    """
    from pfb_imaging.utils.misc import convolve2gaussres

    n = 48
    img = np.zeros((3, n, n))
    img[:, n // 2, n // 2] = 1.0
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    gpars = np.array([[4.0, 4.0, 0.0], [8.0, 8.0, 0.0], [12.0, 12.0, 0.0]])

    out = convolve2gaussres(img, xx, yy, gpars, nthreads=1, pfrac=0.2)

    # peak-normalised kernels: total flux scales with emaj*emin, so wider
    # planes must integrate to more. Equal sums would mean one shared kernel.
    sums = [float(out[i].sum()) for i in range(3)]
    assert sums[0] < sums[1] < sums[2]


def test_convolve2gaussres_single_triple_still_shared():
    """A 1-D gaussparf stays the single-shared-resolution branch."""
    from pfb_imaging.utils.misc import convolve2gaussres

    n = 48
    img = np.zeros((3, n, n))
    img[:, n // 2, n // 2] = 1.0
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")

    out = convolve2gaussres(img, xx, yy, np.array([6.0, 3.0, 0.3]), nthreads=1, pfrac=0.2)

    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out[0], out[2], rtol=0, atol=1e-12)


def test_convolve2gaussres_gausspari_shared_triple_matches_per_plane():
    """A shared (3,) gausspari must give the same result as passing the
    equivalent (nplane, 3) per-plane array with identical rows -- this is
    the `gpi = gausspari if gausspari.ndim == 1 else gausspari[b]` branch.
    """
    from pfb_imaging.utils.misc import convolve2gaussres

    n = 48
    img = np.zeros((3, n, n))
    img[:, n // 2 + 3, n // 2 - 2] = 1.0
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    gaussparf = np.array([6.0, 3.0, 0.3])
    gausspari_shared = np.array([2.0, 2.0, 0.0])
    gausspari_per_plane = np.tile(gausspari_shared, (3, 1))

    shared = convolve2gaussres(img, xx, yy, gaussparf, nthreads=1, pfrac=0.2, gausspari=gausspari_shared)
    per_plane = convolve2gaussres(img, xx, yy, gaussparf, nthreads=1, pfrac=0.2, gausspari=gausspari_per_plane)

    np.testing.assert_allclose(shared, per_plane, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Resolution changes (gausspari given). Issue #312: these all passed vacuously
# before, because every test above uses gausspari None, equal to gaussparf, or
# a smooth centred Gaussian whose spectrum dies exactly where the old sampled
# division went wrong.
# ---------------------------------------------------------------------------


@pmp("npix", [128, 512])
def test_convolve2gaussres_preserves_position_when_deconvolving(npix):
    """A delta must land where it started after a resolution change.

    The old sampled division put it elsewhere entirely -- at npix=512 a source
    at (170, 260) came out at (146, 260), displaced by twice its offset from
    the image centre, with -0.81 of peak in ringing (issue #312).
    """
    y0, x0 = npix // 3, npix // 2 + 4
    image = np.zeros((1, npix, npix))
    image[0, y0, x0] = 1.0
    coord = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")

    out = convolve2gaussres(
        image,
        xx,
        yy,
        np.array([[6.3, 4.2, 0.0]]),
        nthreads=1,
        gausspari=np.array([[6.0, 4.0, 0.0]]),
        norm_kernel=False,
        yx_order=True,
    )[0]

    assert np.unravel_index(np.argmax(out), out.shape) == (y0, x0)
    # a small broadening of a delta is a near-identity: no deep negatives
    assert out.min() > -0.05 * out.max()


@pmp("norm_kernel", [False, True])
def test_convolve2gaussres_conserves_flux_through_a_resolution_change(norm_kernel):
    """Total flux scales by the ratio of the kernels' volumes, exactly.

    With norm_kernel both kernels carry unit volume, so the gain is 1; without
    it the gain is (emaj_f * emin_f) / (emaj_i * emin_i). The old sampled
    division missed this by +6 percent at a 6 px beam and -25 percent at 12 px.
    """
    npix = 256
    coord = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    image = np.zeros((1, npix, npix))
    image[0, npix // 3, npix // 2 + 4] = 1.0
    gausspari = np.array([5.0, 3.0, 0.4])
    gaussparf = np.array([11.0, 7.0, 0.4])

    out = convolve2gaussres(image, xx, yy, gaussparf, nthreads=1, gausspari=gausspari, norm_kernel=norm_kernel)

    expected = 1.0 if norm_kernel else (gaussparf[0] * gaussparf[1]) / (gausspari[0] * gausspari[1])
    assert out.sum() == pytest.approx(expected, rel=1e-6)


@pmp("srf", [2.0, 3.0, 4.0])
def test_convolve2gaussres_two_step_matches_direct_across_srf(srf):
    """Going to gf via gi must equal going to gf directly.

    The beam is 2 * srf pixels across and the sky is band limited to the uv
    coverage that implies, so this is the invariant a restore run actually
    depends on. The old sampled division broke it by 1.8e-3 and 1.4e-4 of peak
    at srf 3 and 4, where the beam is well enough sampled that its transform
    sinks into round-off before Nyquist and the quotient is noise over noise.
    srf 2 is the one case it survived (4.0e-10): a 4 px FWHM aliases enough
    that the transform never gets that small. The closed-form ratio is exact
    at every srf, leaving only that aliasing floor (8.1e-10 at srf 2, 6e-16
    above it).

    The sky is windowed away from the frame edge and only the interior is
    compared: the two-step path pads and unpads twice, and the flux that spills
    into the pad is dropped at the first unpad. That edge artefact is real but
    is not what this test is about, and it swamps everything at ~1e-1.
    """
    npix = 256
    rng = np.random.default_rng(42)
    # band limited to |k| <= 1 / (2 * srf) cycles/pixel, then tapered
    k = np.fft.fftfreq(npix)
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    hat = np.fft.fft2(rng.standard_normal((npix, npix)))
    hat[kr > 1.0 / (2 * srf)] = 0.0
    sky = np.real(np.fft.ifft2(hat))
    coord = -(npix // 2) + np.arange(npix)
    radius = np.sqrt(coord[:, None] ** 2 + coord[None, :] ** 2)
    sky = sky * np.exp(-0.5 * (radius / (npix / 8.0)) ** 2)
    sky = (sky / np.abs(sky).max())[None]

    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    fwhm = 2 * srf
    gausspari = np.array([fwhm, 0.7 * fwhm, 0.0])
    gaussparf = np.array([2 * fwhm, 1.4 * fwhm, 0.0])

    im_i = convolve2gaussres(sky, xx, yy, gausspari, nthreads=1)
    two_step = convolve2gaussres(im_i, xx, yy, gaussparf, nthreads=1, gausspari=gausspari)
    direct = convolve2gaussres(sky, xx, yy, gaussparf, nthreads=1)

    interior = slice(npix // 4, 3 * npix // 4)
    err = np.abs(two_step[:, interior, interior] - direct[:, interior, interior]).max()
    assert err < 1e-8 * np.abs(direct).max()


@pmp(
    "gausspari, gaussparf",
    [
        ((6.0, 4.0, 0.0), (5.0, 3.0, 0.0)),  # sharper on both axes
        ((6.0, 4.0, 0.0), (7.0, 3.5, 0.0)),  # minor axis shrinks
        ((10.0, 5.0, 0.0), (10.0, 5.0, 0.3)),  # same axes, rotated
    ],
)
def test_convolve2gaussres_refuses_to_sharpen(gausspari, gaussparf):
    """Sharpening is a deconvolution; say so instead of amplifying noise.

    The rotated case is the one that catches callers by surprise, and it is
    what forced restoration.lowest_resolution to become a Loewner envelope
    (D32): its old max-of-each-axis, mean-of-the-position-angles target landed
    here routinely, with both axes grown.
    """
    npix = 64
    coord = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    image = np.zeros((1, npix, npix))
    image[0, npix // 2, npix // 2] = 1.0

    with pytest.raises(ValueError, match="deconvolution"):
        convolve2gaussres(image, xx, yy, np.array(gaussparf), nthreads=1, gausspari=np.array(gausspari))


def test_convolve2gaussres_rejects_xy_ordered_grids():
    """The closed-form ratio reads the pixel size off xx/yy, so order matters.

    np.meshgrid's *default* indexing is "xy", which transposes both grids and
    makes the inferred spacings zero -- every frequency infinite and the whole
    image NaN. Fail instead of returning that.
    """
    npix = 64
    coord = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(coord, coord)  # note: no indexing="ij"
    image = np.zeros((1, npix, npix))
    image[0, npix // 2, npix // 2] = 1.0

    with pytest.raises(ValueError, match="indexing='ij'"):
        convolve2gaussres(image, xx, yy, np.array([6.3, 4.2, 0.0]), nthreads=1, gausspari=np.array([6.0, 4.0, 0.0]))


def test_convolve2gaussres_allows_an_unchanged_resolution():
    """gaussparf == gausspari must survive the positive semi-definite check."""
    npix = 64
    coord = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    image = np.zeros((1, npix, npix))
    image[0, npix // 2 + 3, npix // 2 - 2] = 1.0
    gausspar = np.array([6.0, 4.0, 0.7])

    out = convolve2gaussres(image, xx, yy, gausspar, nthreads=1, gausspari=gausspar)

    np.testing.assert_allclose(out, image, rtol=0, atol=1e-10)
