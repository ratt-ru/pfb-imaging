"""Numerical sanity checks on the ducc0 gridder itself.

These do not test pfb-imaging. They test that the ducc0 *build* we are running
on computes the right numbers, which is not something the rest of the suite can
distinguish from a pfb bug: a gridder that silently returns zeros looks exactly
like an imager that read no data.

Why this is worth a test file of its own:

* ducc0 publishes no manylinux aarch64 wheel, so on arm it is compiled from
  sdist against whatever ``DUCC0_ARCH_FLAGS`` we pass, with SIMD paths
  (NEON/SVE) selected at compile time by ``src/ducc0/infra/simd.h``. A
  miscompiled or mis-selected kernel is a silent-wrong-answer failure, not a
  crash.
* ducc0 0.41.0 (2026-03-26) predates arm64 entering ducc's own CI matrix
  (mreineck/ducc#63, 2026-09-23) and several correctness fixes that are still
  unreleased, including mreineck/ducc#77, which is about axis ordering when an
  array has unusual strides.

That last point is why `test_transposed_output_matches_contiguous` exists
rather than a single vanilla call: `operators/gridder.grid_partition` passes
``dirty=dirty[c].T``, i.e. ducc writes into a **non-contiguous, F-ordered
view** (wiki D19/D20 -- our arrays are (Y, X) and ducc's world is x-major).
A check that only ever grids into a C-contiguous buffer would pass while the
code path we actually use returns garbage.

Kept deliberately tiny (64x64, a handful of rows) so this runs in well under a
second and can gate a build before the real suite.
"""

import numpy as np
import pytest
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from pfb_imaging.operators.gridder import wgridder_conventions

CELL_RAD = 1.0e-5
NPIX = 64

# ducc0 picks its gridding kernel from (epsilon, precision): asking for 1e-7 in
# single precision raises "No appropriate kernel found". Both precisions are
# worth covering -- they are different compiled kernels, and the aarch64 imager
# failure that prompted this file showed up in both.
PRECISIONS = {
    "single": (np.float32, np.complex64, 1e-5),
    "double": (np.float64, np.complex128, 1e-10),
}
PRECISION_IDS = list(PRECISIONS)


def _conventions():
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    return dict(flip_u=flip_u, flip_v=flip_v, flip_w=flip_w, center_x=x0, center_y=y0)


def _uvw_freq(nrow=64, nchan=4, seed=42):
    rng = np.random.default_rng(seed)
    uvw = rng.normal(scale=2.0e3, size=(nrow, 3))
    uvw[:, 2] *= 0.05  # keep w modest; this is not a w-term test
    freq = np.linspace(1.0e9, 1.1e9, nchan)
    return uvw, freq


@pytest.mark.parametrize("precision", PRECISION_IDS)
def test_zero_baseline_grids_to_a_flat_nonzero_image(precision):
    """A single visibility at uvw = 0 must produce a flat, nonzero image.

    exp(-2*pi*i*(ul + vm + w(n-1))) is 1 everywhere when uvw is 0, so the dirty
    image is the weighted visibility at every pixel. Exact, analytic, and the
    cheapest possible detector for "the gridder returned zeros".
    """
    real_t, complex_t, eps = PRECISIONS[precision]
    uvw = np.zeros((1, 3))
    freq = np.array([1.0e9])
    vis = np.ones((1, 1), dtype=complex_t)
    wgt = np.ones((1, 1), dtype=real_t)

    dirty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=wgt,
        npix_x=NPIX,
        npix_y=NPIX,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **_conventions(),
    )

    assert np.isfinite(dirty).all(), "gridder produced non-finite values"
    assert np.any(dirty != 0), "gridder returned an all-zero image for uvw=0"
    np.testing.assert_allclose(dirty, 1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("precision", PRECISION_IDS)
def test_a_point_source_lands_on_the_centre_pixel(precision):
    """Round-trip a centred point source: dirty2vis then vis2dirty.

    A source at the phase centre has l = m = 0, so every model visibility is
    the source flux; gridding those back must peak at the centre pixel. Catches
    a gridder that produces *something* but puts it in the wrong place -- which
    an all-zero check alone would miss.
    """
    real_t, complex_t, eps = PRECISIONS[precision]
    uvw, freq = _uvw_freq()
    conv = _conventions()

    model = np.zeros((NPIX, NPIX), dtype=real_t)
    model[NPIX // 2, NPIX // 2] = 1.0

    vis = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=model,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **conv,
    )
    # l = m = 0 means every model visibility is the source flux itself
    np.testing.assert_allclose(vis.real, 1.0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vis.imag, 0.0, rtol=0, atol=1e-4)

    dirty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=np.ones(vis.shape, dtype=real_t),
        npix_x=NPIX,
        npix_y=NPIX,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **conv,
    )

    assert np.any(dirty != 0), "gridder returned an all-zero image"
    peak = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert peak == (NPIX // 2, NPIX // 2), f"PSF peaks at {peak}, expected the centre"


@pytest.mark.parametrize("precision", PRECISION_IDS)
def test_transposed_output_matches_contiguous(precision):
    """Gridding into a `.T` view must equal gridding into a fresh array.

    `operators/gridder.grid_partition` passes `dirty=dirty[c].T` -- ducc writes
    into an F-ordered, non-contiguous view, because our image arrays are (Y, X)
    and ducc is x-major (wiki D19/D20). Axis ordering by stride is exactly what
    mreineck/ducc#77 corrected, so a build can get the contiguous case right and
    this one wrong. That would show up as a wrong or empty image from every pfb
    command while a naive gridder check passed.
    """
    real_t, complex_t, eps = PRECISIONS[precision]
    uvw, freq = _uvw_freq()
    conv = _conventions()
    vis = np.ones((uvw.shape[0], freq.size), dtype=complex_t)
    wgt = np.ones(vis.shape, dtype=real_t)

    kwargs = dict(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=wgt,
        npix_x=NPIX,
        npix_y=NPIX,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **conv,
    )

    returned = vis2dirty(**kwargs)

    # the pfb call shape: a (Y, X) array handed over as its x-major transpose
    out = np.zeros((NPIX, NPIX), dtype=real_t)
    vis2dirty(dirty=out.T, **kwargs)

    assert np.any(out != 0), "gridding into a transposed view produced zeros"
    np.testing.assert_allclose(out.T, returned, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("precision", PRECISION_IDS)
def test_gridder_and_degridder_are_adjoint(precision):
    """<G x, y> == <x, G^T y>, the standard adjointness check.

    Degridding and gridding are a transpose pair, so this one scalar identity
    exercises both kernels over the whole array at once. It is far more
    sensitive to a bad SIMD tail or a mis-ordered axis than a peak-position
    assertion: a handful of wrong elements anywhere shifts the inner product.
    """
    real_t, complex_t, eps = PRECISIONS[precision]
    uvw, freq = _uvw_freq(nrow=128, nchan=4, seed=7)
    conv = _conventions()
    rng = np.random.default_rng(3)

    x = rng.normal(size=(NPIX, NPIX)).astype(real_t)
    y = (rng.normal(size=(uvw.shape[0], freq.size)) + 1j * rng.normal(size=(uvw.shape[0], freq.size))).astype(complex_t)

    gx = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=x,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **conv,
    )
    gty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=y,
        wgt=np.ones(y.shape, dtype=real_t),
        npix_x=NPIX,
        npix_y=NPIX,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        nthreads=1,
        **conv,
    )

    lhs = np.vdot(gx, y).real
    rhs = np.vdot(x, gty).real
    assert abs(lhs) > 0, "adjointness test is vacuous: <Gx, y> is zero"
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4)


@pytest.mark.parametrize("precision", PRECISION_IDS)
@pytest.mark.parametrize("nthreads", [2, 4])
def test_threaded_gridding_matches_single_threaded(nthreads, precision):
    """Thread count must not change the answer beyond the requested accuracy.

    ducc0 splits its loops across threads, so the accumulation order changes
    with `nthreads` and the output is NOT bit-identical -- asserting that would
    be wrong, and in single precision it fails by ~1e-3 relative on pixels where
    1024 unit-magnitude contributions cancel. What must hold is ducc's own
    contract: agreement to `epsilon` as a fraction of the image peak. A bad SIMD
    tail or a race breaks that by orders of magnitude, not by a few ulp.

    Covers the `nthreads > 1` path every real pfb run takes.
    """
    real_t, complex_t, eps = PRECISIONS[precision]
    uvw, freq = _uvw_freq(nrow=256, nchan=4, seed=11)
    conv = _conventions()
    vis = np.ones((uvw.shape[0], freq.size), dtype=complex_t)
    wgt = np.ones(vis.shape, dtype=real_t)

    kwargs = dict(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=wgt,
        npix_x=NPIX,
        npix_y=NPIX,
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        epsilon=eps,
        do_wgridding=False,
        divide_by_n=False,
        **conv,
    )

    reference = vis2dirty(nthreads=1, **kwargs)
    assert np.any(reference != 0), "gridder returned an all-zero image"

    threaded = vis2dirty(nthreads=nthreads, **kwargs)
    peak = np.abs(reference).max()
    err = np.abs(threaded - reference).max() / peak
    assert err < 10 * eps, f"nthreads={nthreads} changed the image by {err:.2e} of peak (epsilon={eps:.0e})"
