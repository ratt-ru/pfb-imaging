"""Pin the image and beam orientation conventions.

These tests are the durable form of the synthetic experiments that measured
the conventions documented in docs/wiki/image-and-beam-orientation.md:

1. the wgridder dirty image is (X, Y)-ordered with ix <-> l/RA descending and
   iy <-> m/Dec ascending under pfb's fixed flip conventions and the effective
   +2pi*i visibility phase convention of real MS data;
2. reproject_and_interp_scat_beam maps a (Y, X)-ordered beam around the
   pointing centre onto the (Y, X)-ordered output image grid exactly (up to
   interpolation error), including under rephasing and for non-square images.

An elliptical test beam is used because every historical alignment bug
(transpose/flip/rotation) is invisible for a circularly symmetric beam.
"""

import numpy as np
import pytest
from astropy.wcs import WCS
from ducc0.wgridder import vis2dirty
from scipy.constants import c as lightspeed

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.utils.beam import reproject_and_interp_scat_beam
from pfb_imaging.utils.misc import fitcleanbeam

pmp = pytest.mark.parametrize


def test_wgridder_image_orientation():
    """A point source at (l, m) = (east, north) must land at
    (ix, iy) = (nx//2 - l/cell, ny//2 + m/cell), i.e. ix <-> RA descending and
    iy <-> Dec ascending, matching the hci coordinate assignment
    (out_ras descending, out_decs ascending) and the save_fits transpose."""
    rng = np.random.default_rng(42)
    nx = ny = 128
    cell_rad = np.deg2rad(1.0 / 3600.0)
    freq = np.array([1.4e9])

    lpix, mpix = 8, 20  # east, north
    l_s = lpix * cell_rad
    m_s = mpix * cell_rad
    n_s = np.sqrt(1.0 - l_s**2 - m_s**2)

    nrow = 10000
    uvw = np.zeros((nrow, 3))
    uvw[:, 0] = rng.uniform(-3e3, 3e3, nrow)
    uvw[:, 1] = rng.uniform(-3e3, 3e3, nrow)
    uvw[:, 2] = rng.uniform(-50, 50, nrow)

    # the effective phase convention of real MS data through this pipeline
    # (the -2pi*i convention produces a 180 degree rotated image)
    phase = 2j * np.pi * freq[None, :] / lightspeed * (uvw[:, 0:1] * l_s + uvw[:, 1:2] * m_s + uvw[:, 2:] * (n_s - 1.0))
    vis = np.exp(phase).astype(np.complex128)
    wgt = np.ones((nrow, 1))

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    dirty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=wgt,
        npix_x=nx,
        npix_y=ny,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        center_x=x0,
        center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=4,
    )

    ix, iy = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert (ix, iy) == (nx // 2 - lpix, ny // 2 + mpix)


def _elliptical_beam(ll, mm):
    """Gaussian with axial ratio 0.6 at PA 30 deg — no rotation or reflection
    symmetry, so any axis mix-up shows up as a large amplitude error."""
    pa = np.deg2rad(30.0)
    a, b = 0.25, 0.15  # deg
    u = ll * np.cos(pa) + mm * np.sin(pa)
    v = -ll * np.sin(pa) + mm * np.cos(pa)
    return np.exp(-(u**2 / (2 * a**2) + v**2 / (2 * b**2)))


@pmp("offset_lm", [(0.0, 0.0), (0.12, 0.0), (0.0, 0.08), (-0.1, 0.06)])
def test_reproject_scat_beam_matches_analytic(offset_lm):
    """reproject_and_interp_scat_beam must reproduce the analytically
    evaluated beam on the output grid: (Y, X) in and out, signed cdelt from
    the beam coords, target WCS identical to the hci output header, correct
    handling of the pointing -> phase-centre offset, non-square output."""
    ra0, dec0 = 45.0, -30.0
    radec0 = np.deg2rad([ra0, dec0])
    # rephased output centre offset by (dl, dm) on the sky
    dl_off, dm_off = offset_lm
    raf = ra0 + dl_off / np.cos(np.deg2rad(dec0))
    decf = dec0 + dm_off
    radecf = np.deg2rad([raf, decf])

    # coarse, wide beam grid as in the MdV bds (degrees, centred, ascending)
    nb = 301
    l_beam = (np.arange(nb) - nb // 2) * 0.02
    m_beam = (np.arange(nb) - nb // 2) * 0.02
    ll, mm = np.meshgrid(l_beam, m_beam)  # default indexing="xy" -> (Y, X)
    beam = _elliptical_beam(ll, mm)[None, :, :]

    # fine, non-square output grid
    nxo, nyo = 128, 96
    cell_out = 0.01

    pbeam = reproject_and_interp_scat_beam(
        beam,
        l_beam,
        m_beam,
        radec0,
        radecf,
        cell_out,
        nxo,
        nyo,
        "I",
    )
    assert pbeam.shape == (1, nyo, nxo)
    # the beam grid fully covers the output fov, so no pixel may be masked
    assert np.all(pbeam[0] > 0)

    # analytic reference: output pixel -> world -> beam-frame (l, m) -> beam
    wcs_target = WCS(naxis=2)
    wcs_target.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_target.wcs.cdelt = np.array((-cell_out, cell_out))
    wcs_target.wcs.cunit = ["deg", "deg"]
    wcs_target.wcs.crval = np.array((raf, decf))
    wcs_target.wcs.crpix = [1 + nxo // 2, 1 + nyo // 2]

    wcs_beam = WCS(naxis=2)
    wcs_beam.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_beam.wcs.cdelt = np.array((1.0, 1.0))
    wcs_beam.wcs.cunit = ["deg", "deg"]
    wcs_beam.wcs.crval = np.array((ra0, dec0))
    wcs_beam.wcs.crpix = [1.0, 1.0]  # 0-based pixel == (l, m) in degrees

    xpix, ypix = np.meshgrid(np.arange(nxo), np.arange(nyo))  # (nyo, nxo) grids
    world = wcs_target.pixel_to_world(xpix, ypix)
    l_out, m_out = wcs_beam.world_to_pixel(world)
    ref = _elliptical_beam(l_out, m_out)

    # bilinear interpolation of the 0.02-deg-sampled Gaussian: ~1e-3
    assert np.max(np.abs(pbeam[0] - ref)) < 5e-3


def test_fitcleanbeam_yx_order_invariant():
    """fitcleanbeam(yx_order=True) must return bitwise the same (emaj, emin,
    pa) for a transposed PSF, so the hci path can hand it cube (Y, X)-ordered
    arrays without changing the PA convention breifast consumes."""
    nx, ny = 128, 96
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # elliptical PSF main lobe, PA distinct from 0/45/90 deg
    psf = _elliptical_beam(xx * 0.01, yy * 0.01)[None, :, :]  # (1, nx, ny)

    ref = fitcleanbeam(psf, level=0.5, pixsize=1.0)
    yx = fitcleanbeam(psf.transpose(0, 2, 1), level=0.5, pixsize=1.0, yx_order=True)
    np.testing.assert_array_equal(ref, yx)
    # sanity: the fit resolved an elongated beam at a nontrivial angle
    emaj, emin, pa = ref[0]
    assert emaj > emin
    assert 0 < pa < np.pi


@pmp("emaj_true,ar,pa_deg", [(2.0, 0.8, 45), (2.5, 0.6, 30), (12.0, 0.5, 20), (3.0, 1.0, 0)])
def test_fitcleanbeam_axis_ordering_and_quiet(emaj_true, ar, pa_deg, capsys):
    """The optimiser may converge with the axes reversed (routine for a
    marginally-resolved super-resolution beam, whose level 0.5 main lobe spans
    ~1 px). fitcleanbeam must un-swap to emaj >= emin every time and must NOT
    print a warning for this expected, correctly-handled outcome."""
    from pfb_imaging.utils.misc import gaussian2d

    npix = 129
    x = -(npix // 2) + np.arange(npix)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    psf = gaussian2d(xx, yy, (emaj_true, emaj_true * ar, np.deg2rad(pa_deg)), normalise=False)[None]

    emaj, emin, pa = fitcleanbeam(psf, level=0.5, pixsize=1.0)[0]
    assert emaj >= emin
    assert 0 <= pa
    assert "flipped" not in capsys.readouterr().out.lower()
