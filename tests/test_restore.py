"""Tests for ``pfb restore`` on the imager ``.dt`` DataTree (issue #303).

Unit tests exercise the pure array functions in ``utils/restoration.py``.
Driver tests build a synthetic ``.dt`` in-process (``_write_restore_dt``, no
MS or imager needed), in the style of ``tests/test_fits_tree.py``. One
integration test runs ``imager -> deconv -> restore`` against the
``sky_truth`` fixture.
"""

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits as afits


def test_restore_products_algebra():
    """The three products differ only in where the beam is applied.

    With a spatially constant beam the closed forms collapse to scalar
    relations, so the test pins the definitions rather than the numerics of
    the convolution.
    """
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 64
    rng = np.random.default_rng(0)
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 2.0
    residual = rng.standard_normal((1, ny, nx)) * 0.01
    beam = np.full((1, ny, nx), 0.5)
    gpar = np.array([[4.0, 4.0, 0.0]])

    out = restore_products(model, residual, beam, gpar, products=("a", "i", "k"), pb_min=0.1)

    mconv = out["k"] - residual  # m (x) G, recovered from the mixed product
    np.testing.assert_allclose(out["i"], mconv + residual / 0.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["a"], 0.5 * mconv + residual, rtol=0, atol=1e-10)
    # the relation the mosaic case depends on
    np.testing.assert_allclose(out["a"] / 0.5, out["i"], rtol=0, atol=1e-10)


def test_restore_products_pb_min_zeroes_low_beam():
    """The intrinsic image is zeroed below pb_min, following utils/spi.py:31."""
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 32
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 1.0
    residual = np.full((1, ny, nx), 0.1)
    beam = np.ones((1, ny, nx))
    beam[0, :, : nx // 2] = 0.05  # below the floor
    gpar = np.array([[3.0, 3.0, 0.0]])

    out = restore_products(model, residual, beam, gpar, products=("i",), pb_min=0.1)

    assert np.all(out["i"][0, :, : nx // 2] == 0.0)
    assert np.any(out["i"][0, :, nx // 2 :] != 0.0)


@pytest.mark.parametrize("pa_deg", [20.0, 70.0])
def test_restore_products_preserves_position_angle(pa_deg):
    """Restoring a delta with an elliptical beam reproduces that beam on the
    (y, x) raster. Dropping convolve2gaussres(yx_order=True) mirrors the PA
    and this fails (wiki D19/D20).
    """
    from pfb_imaging.utils.misc import fitcleanbeam
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 128
    gpar = np.array([[10.0, 4.0, np.deg2rad(pa_deg)]])
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 1.0
    residual = np.zeros((1, ny, nx))
    beam = np.ones((1, ny, nx))

    out = restore_products(model, residual, beam, gpar, products=("k",))

    emaj, emin, pa = fitcleanbeam(out["k"] / out["k"].max(), yx_order=True)[0]
    np.testing.assert_allclose(emaj, 10.0, rtol=0.05)
    np.testing.assert_allclose(emin, 4.0, rtol=0.05)
    np.testing.assert_allclose(pa, np.deg2rad(pa_deg), atol=np.deg2rad(3.0))


def test_restore_apparent_applies_the_beam_before_convolving():
    """``a`` is ``(B*m) (x) G``, not ``B*(m (x) G)``.

    Convolution does not commute with multiplication by a spatially varying
    beam, so the two differ. Every other product test uses a constant beam,
    under which they coincide -- this is the only guard against a future
    "simplification" to ``beam * mconv + rconv``.
    """
    from pfb_imaging.utils.restoration import restore_products

    nx = ny = 64
    model = np.zeros((1, ny, nx))
    model[0, ny // 2, nx // 2] = 1.0
    residual = np.zeros((1, ny, nx))
    # a beam that varies strongly across the restoring kernel's footprint
    ramp = np.linspace(0.2, 1.0, nx)[None, None, :]
    beam = np.broadcast_to(ramp, (1, ny, nx)).copy()
    gpar = np.array([[8.0, 8.0, 0.0]])

    out = restore_products(model, residual, beam, gpar, products=("a", "k"), pb_min=0.0)

    shortcut = beam * out["k"]  # the wrong-but-plausible B*(m (x) G)
    assert not np.allclose(out["a"], shortcut, rtol=1e-3, atol=1e-8)
    # the correct form attenuates by the beam at the source, then spreads
    peak = float(beam[0, ny // 2, nx // 2])
    np.testing.assert_allclose(out["a"].max(), peak * out["k"].max(), rtol=0.05)


def test_clean_beam_is_weighted_and_not_the_mean_of_per_band_fits():
    """G_mfs is the fit to the wsum-weighted average PSF, so a band carrying
    almost no weight barely moves it. The mean of the per-band fitted
    Gaussians -- the legacy MFS beam -- is weight-blind and lands far away.
    """
    from pfb_imaging.utils.misc import gaussian2d
    from pfb_imaging.utils.restoration import clean_beam

    n = 128
    coord = -(n // 2) + np.arange(n)
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    shapes = [(6.0, 6.0, 0.0), (18.0, 18.0, 0.0)]
    wsums = np.array([[100.0], [1.0]])
    # stored PSFs are un-normalised (shape * wsum), as core/imager.py writes them
    psfs = np.stack([w[0] * gaussian2d(xx, yy, g, normalise=False).T[None] for g, w in zip(shapes, wsums)])

    per_band = np.stack([clean_beam(p, w) for p, w in zip(psfs, wsums)])
    mfs = clean_beam(psfs.sum(axis=0), wsums.sum(axis=0))

    np.testing.assert_allclose(per_band[0, 0, 0], 6.0, rtol=0.05)
    np.testing.assert_allclose(per_band[1, 0, 0], 18.0, rtol=0.05)
    # the heavily-weighted narrow band dominates the summed PSF
    assert mfs[0, 0] < 8.0
    # ... and that is nowhere near the weight-blind mean of the two fits (12.0)
    assert abs(mfs[0, 0] - per_band[:, 0, 0].mean()) > 3.0


def test_clean_beam_zero_wsum_is_nan_not_a_crash():
    """A fully flagged band has wsum 0; the fit must degrade to NaN so the
    caller's nanmax/nanmean skip it rather than dividing by zero.
    """
    from pfb_imaging.utils.restoration import clean_beam

    psf = np.zeros((1, 32, 32))
    out = clean_beam(psf, np.array([0.0]))
    assert np.isnan(out).all()


def test_lowest_resolution_takes_max_axes_and_mean_pa():
    from pfb_imaging.utils.restoration import lowest_resolution

    gp = np.array([[[6.0, 3.0, 0.2]], [[4.0, 5.0, 0.6]]])
    np.testing.assert_allclose(lowest_resolution(gp), np.array([[6.0, 5.0, 0.4]]))


# ---------------------------------------------------------------------------
# driver tests (Task 4): synthetic .dt, no MS or imager needed
# ---------------------------------------------------------------------------


def _write_restore_dt(
    store,
    nband=2,
    nx=64,
    ny=64,
    beam_vals=(1.0, 0.5),
    gpars=((6.0, 4.0, 0.3), (10.0, 6.0, 0.3)),
    wsums=(10.0, 30.0),
    model_flux=2.0,
    with_psf=True,
    with_model=True,
):
    """Synthetic single-time .dt carrying exactly the band variables restore reads.

    PSF is stored un-normalised (shape x wsum) so PSF/WSUM is a peak-1 Gaussian
    and fitcleanbeam recovers ``gpars[b]``, matching what core/imager.py writes.
    RESIDUAL is likewise stored un-normalised.
    """
    from pfb_imaging.utils.misc import gaussian2d

    nx_psf, ny_psf = 2 * nx, 2 * ny
    xp = -(nx_psf // 2) + np.arange(nx_psf)
    yp = -(ny_psf // 2) + np.arange(ny_psf)
    xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

    for b in range(nband):
        model = np.zeros((1, ny, nx))
        model[0, ny // 2, nx // 2] = model_flux
        data_vars = {
            "DIRTY": (("corr", "y", "x"), np.zeros((1, ny, nx))),
            "RESIDUAL": (("corr", "y", "x"), np.full((1, ny, nx), 0.01) * wsums[b]),
            "BEAM": (("corr", "y", "x"), np.full((1, ny, nx), beam_vals[b])),
            "WSUM": (("corr",), np.array([wsums[b]])),
        }
        if with_model:
            data_vars["MODEL"] = (("corr", "y", "x"), model)
        coords = {"corr": ["I"]}
        if with_psf:
            psf = gaussian2d(xxp, yyp, gpars[b], normalise=False).T[None]
            data_vars["PSF"] = (("corr", "y_psf", "x_psf"), psf * wsums[b])
            data_vars["PSFPARSN"] = (("corr", "bpar"), np.array([list(gpars[b])]))
            coords["bpar"] = ["BMAJ", "BMIN", "BPA"]
        xr.Dataset(
            data_vars,
            coords=coords,
            attrs={
                "bandid": b,
                "timeid": 0,
                "freq_out": 1.0e9 + b * 1.0e8,
                "freq_nominal": 1.0e9 + b * 1.0e8,
                "time_out": 1.7e9,
                "ra": 0.0,
                "dec": 0.0,
                "l0": 0.0,
                "m0": 0.0,
                "cell_rad": 1.0e-6,
                "niters": 1,
            },
        ).to_zarr(store, group=f"band{b:04d}_time0000", mode="a")


def _run_restore(tmp_path, name="rt", **kwargs):
    """Call the driver with the fixed plumbing arguments these tests share."""
    from pfb_imaging.core.restore import restore as restore_core

    restore_core(
        str(tmp_path / name),
        fits_output_folder=str(tmp_path / "fits"),
        log_directory=str(tmp_path / "logs"),
        nthreads=1,
        **kwargs,
    )


def test_restore_writes_products_and_psfparsf(tmp_path):
    """All three products land in the band nodes at native resolution, and
    BIMAGE / BEAM == IMAGE -- the mosaic relation.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="aik")

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    for b in range(2):
        ds = dt[f"band{b:04d}_time0000"].ds
        for v in ("IMAGE", "BIMAGE", "KIMAGE", "PSFPARSF"):
            assert v in ds, f"{v} missing from band {b}"
        # no --gausspar: the final resolution is the native one
        np.testing.assert_allclose(ds.PSFPARSF.values, ds.PSFPARSN.values, rtol=1e-6)
        np.testing.assert_allclose(ds.BIMAGE.values / ds.BEAM.values, ds.IMAGE.values, rtol=1e-5, atol=1e-10)


def test_restore_preserves_band_attrs(tmp_path):
    """to_zarr(mode='a') replaces a group's attrs wholesale, so the driver must
    re-stamp them (core/deconv.py:409-424). Losing bandid breaks every later read.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kK")

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    ds = dt["band0001_time0000"].ds
    assert ds.attrs["bandid"] == 1
    assert ds.attrs["timeid"] == 0
    np.testing.assert_allclose(ds.attrs["freq_out"], 1.1e9)
    np.testing.assert_allclose(ds.attrs["cell_rad"], 1.0e-6)
    assert len(ds.attrs["psfparsf_mfs"]) == 3


def test_restore_only_computes_requested_products(tmp_path):
    """--outputs gates the compute and the storage, not just the FITS."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kK")

    ds = xr.open_datatree(store, engine="zarr", chunks=None)["band0000_time0000"].ds
    assert "KIMAGE" in ds
    assert "IMAGE" not in ds
    assert "BIMAGE" not in ds


def test_restore_gausspar_homogenises_to_specified_resolution(tmp_path):
    """--gausspar is in degrees; PSFPARSF must come back in pixel units."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)
    cell_deg = np.rad2deg(1.0e-6)

    _run_restore(tmp_path, outputs="kK", gausspar=(12.0 * cell_deg, 8.0 * cell_deg, 45.0))

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    for b in range(2):
        pf = dt[f"band{b:04d}_time0000"].ds.PSFPARSF.values[0]
        np.testing.assert_allclose(pf[0], 12.0, rtol=1e-4)
        np.testing.assert_allclose(pf[1], 8.0, rtol=1e-4)
        np.testing.assert_allclose(pf[2], np.deg2rad(45.0), rtol=1e-6)


def test_restore_zero_gausspar_selects_lowest_resolution(tmp_path):
    """--gausspar 0 0 0 homogenises to the lowest-resolution band."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kK", gausspar=(0.0, 0.0, 0.0))

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    for b in range(2):
        pf = dt[f"band{b:04d}_time0000"].ds.PSFPARSF.values[0]
        np.testing.assert_allclose(pf[0], 10.0, rtol=1e-6)  # max emaj over bands
        np.testing.assert_allclose(pf[1], 6.0, rtol=1e-6)  # max emin over bands
        np.testing.assert_allclose(pf[2], 0.3, rtol=1e-6)  # mean pa


def test_restore_skips_zero_wsum_bands(tmp_path):
    """A fully flagged band (WSUM == 0) must be skipped, not divided by.

    Without the guard, `RESIDUAL / WSUM` yields inf/NaN which lands in the
    stored products AND in the MFS accumulators, poisoning every other band's
    MFS image. core/imager.py already emits such bands (freq_eff falls back to
    freq_nominal when wsum_tot == 0).
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, wsums=(10.0, 0.0))

    _run_restore(tmp_path, outputs="kK")

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    # the live band is restored and finite
    live = dt["band0000_time0000"].ds
    assert "KIMAGE" in live
    assert np.isfinite(live.KIMAGE.values).all()
    # the dead band is skipped entirely rather than written with NaNs
    assert "KIMAGE" not in dt["band0001_time0000"].ds
    # the live band's attrs still record the MFS beam, so the dead band did not
    # poison the reduction (the MFS *image* is asserted finite in Task 5)
    assert np.isfinite(np.asarray(live.attrs["psfparsf_mfs"], dtype=float)).all()


def test_restore_errors_without_psf(tmp_path):
    """A --no-psf imager tree cannot define a restoring beam."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, with_psf=False)

    with pytest.raises(ValueError, match="--psf"):
        _run_restore(tmp_path, outputs="kK")


def test_restore_errors_without_model(tmp_path):
    """Nothing to restore before pfb deconv has run."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, with_model=False)

    with pytest.raises(ValueError, match="MODEL"):
        _run_restore(tmp_path, outputs="kK")


# ---------------------------------------------------------------------------
# FITS dispatch (Task 5)
# ---------------------------------------------------------------------------


def test_restore_writes_mfs_and_cube_fits(tmp_path):
    """Every requested letter produces its file, with the case controlling
    whether it is the MFS image or the cube.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="aAiIkKrR")

    base = tmp_path / "fits" / "rt_I_main"
    for var in ("bimage", "image", "kimage", "residual"):
        assert (tmp_path / "fits" / f"rt_I_main_{var}_time0_mfs.fits").exists(), var
        assert (tmp_path / "fits" / f"rt_I_main_{var}_time0.fits").exists(), var
    cube = afits.getdata(str(base) + "_kimage_time0.fits")
    assert cube.shape[1] == 2  # FREQ axis carries both bands
    mfs = afits.getdata(str(base) + "_kimage_time0_mfs.fits")
    assert mfs.shape[1] == 1  # MFS collapses the FREQ axis


def test_restore_mfs_beam_is_fit_to_the_mfs_psf(tmp_path):
    """BMAJ on the MFS image comes from the summed PSF, not from averaging the
    per-band beams -- the correction issue #303 exists for.
    """
    from pfb_imaging.utils.restoration import clean_beam

    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, gpars=((6.0, 6.0, 0.0), (18.0, 18.0, 0.0)))

    _run_restore(tmp_path, outputs="kK")

    hdr = afits.getheader(str(tmp_path / "fits" / "rt_I_main_kimage_time0_mfs.fits"))
    cell_deg = np.rad2deg(1.0e-6)

    dt = xr.open_datatree(store, engine="zarr", chunks=None)
    psf_sum = sum(dt[f"band{b:04d}_time0000"].ds.PSF.values for b in range(2))
    wsum = sum(dt[f"band{b:04d}_time0000"].ds.WSUM.values for b in range(2))
    want = clean_beam(psf_sum, wsum)[0]

    np.testing.assert_allclose(hdr["BMAJ"], want[0] * cell_deg, rtol=1e-4)


def test_restore_cube_beams_come_from_psfparsf(tmp_path):
    """Cube BMAJ{i} cards carry the final resolution, not the native one."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)
    cell_deg = np.rad2deg(1.0e-6)

    _run_restore(tmp_path, outputs="kK", gausspar=(20.0 * cell_deg, 20.0 * cell_deg, 0.0))

    hdr = afits.getheader(str(tmp_path / "fits" / "rt_I_main_kimage_time0.fits"))
    np.testing.assert_allclose(hdr["BMAJ1"], 20.0 * cell_deg, rtol=1e-4)
    np.testing.assert_allclose(hdr["BMAJ2"], 20.0 * cell_deg, rtol=1e-4)


def test_restore_mfs_residual_floor_is_the_weighted_mean(tmp_path):
    """MFS restored = r_mfs + m_mfs (x) G_mfs, both wsum-weighted over kept
    bands. Away from the source the restored image is just r_mfs, which for a
    flat 0.01 residual in both bands must be flat 0.01.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, beam_vals=(1.0, 1.0))

    _run_restore(tmp_path, outputs="kK")

    got = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_kimage_time0_mfs.fits")))
    assert abs(float(got[0, 0]) - 0.01) < 1e-5
    # the source is still there at the centre
    assert float(got[32, 32]) > 0.01


def test_restore_intrinsic_mfs_is_apparent_over_beam(tmp_path):
    """The mosaic relation must survive the MFS reduction too: with a uniform
    beam, IMAGE == BIMAGE / beam pixel for pixel.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, beam_vals=(0.5, 0.5))

    _run_restore(tmp_path, outputs="ai")

    app = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_bimage_time0_mfs.fits")))
    intr = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_image_time0_mfs.fits")))
    np.testing.assert_allclose(app / 0.5, intr, rtol=1e-4, atol=1e-8)
    hdr = afits.getheader(str(tmp_path / "fits" / "rt_I_main_image_time0_mfs.fits"))
    assert hdr["PBMIN"] == 0.1


def test_restore_drop_bands_excludes_from_mfs_beam(tmp_path):
    """A dropped band must leave the G_mfs PSF fit, not just the image sum."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, gpars=((6.0, 6.0, 0.0), (18.0, 18.0, 0.0)))
    cell_deg = np.rad2deg(1.0e-6)

    _run_restore(tmp_path, outputs="kK", drop_bands=[1])

    hdr = afits.getheader(str(tmp_path / "fits" / "rt_I_main_kimage_time0_mfs.fits"))
    # band 1 (the 18 px beam) is gone, so G_mfs is band 0's 6 px beam
    np.testing.assert_allclose(hdr["BMAJ"], 6.0 * cell_deg, rtol=0.05)
    cube = afits.getdata(str(tmp_path / "fits" / "rt_I_main_kimage_time0.fits"))
    assert cube.shape[1] == 1  # dropped, not zeroed


def test_restore_zero_wsum_band_does_not_poison_mfs(tmp_path):
    """The Task 4 guard, asserted at the FITS level: a fully flagged band must
    leave the MFS image finite.
    """
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store, wsums=(10.0, 0.0))

    _run_restore(tmp_path, outputs="kK")

    mfs = afits.getdata(str(tmp_path / "fits" / "rt_I_main_kimage_time0_mfs.fits"))
    assert np.isfinite(mfs).all()
    cube = afits.getdata(str(tmp_path / "fits" / "rt_I_main_kimage_time0.fits"))
    assert cube.shape[1] == 1  # only the live band


# ---------------------------------------------------------------------------
# clean-beam and FFT-residual products (Task 6)
# ---------------------------------------------------------------------------


def test_restore_clean_beam_images(tmp_path):
    """c/C render the restoring Gaussian itself, peak 1 at the image centre."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kKcC")

    mfs = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_cpsf_time0_mfs.fits")))
    assert mfs.shape == (64, 64)
    np.testing.assert_allclose(mfs.max(), 1.0, rtol=1e-5)
    assert np.unravel_index(int(np.argmax(mfs)), mfs.shape) == (32, 32)

    cube = afits.getdata(str(tmp_path / "fits" / "rt_I_main_cpsf_time0.fits"))
    assert cube.shape == (1, 2, 64, 64)
    # band 1 has the wider native beam, so its Gaussian integrates to more
    assert cube[0, 1].sum() > cube[0, 0].sum()


def test_restore_fft_residual_products(tmp_path):
    """f/F write magnitude and phase of the FFT of the residual."""
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kKfF")

    for stem in ("abs_fft_residual", "phase_fft_residual"):
        for tail in ("_time0_mfs.fits", "_time0.fits"):
            assert (tmp_path / "fits" / f"rt_I_main_{stem}{tail}").exists(), stem + tail

    # a flat residual transforms to a single central spike
    mag = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_abs_fft_residual_time0_mfs.fits")))
    assert np.unravel_index(int(np.argmax(mag)), mag.shape) == (32, 32)
    assert mag.max() > 100.0 * np.median(mag)

    phase = np.squeeze(afits.getdata(str(tmp_path / "fits" / "rt_I_main_phase_fft_residual_time0_mfs.fits")))
    assert np.all(np.abs(phase) <= np.pi + 1e-5)


def test_restore_clean_beam_not_written_when_not_requested(tmp_path):
    store = str(tmp_path / "rt_I.dt")
    _write_restore_dt(store)

    _run_restore(tmp_path, outputs="kK")

    assert not (tmp_path / "fits" / "rt_I_main_cpsf_time0_mfs.fits").exists()
    assert not (tmp_path / "fits" / "rt_I_main_abs_fft_residual_time0_mfs.fits").exists()
