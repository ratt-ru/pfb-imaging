"""imager --transfer-model-from (#309): upfront refusals and the pass-2 residual."""

from pathlib import Path

import numpy as np
import pytest

from pfb_imaging.core.imager import imager as imager_core


def _ms_radec(ms_name):
    import arcae

    with arcae.table(f"{ms_name}::FIELD") as tab:
        return tuple(float(v) for v in np.asarray(tab.getcol("PHASE_DIR")).squeeze())


def _write_mds(
    path, *, radec, cell_rad, stokes=("I",), nx=32, ny=32, freqs=(1.0e9, 1.1e9), flux_scale=None, freq_bounds=True
):
    from pfb_model_spec.utils.io import build_mds_dataset

    freqs = np.asarray(freqs)
    ns = len(stokes)
    coeffs = np.ones((ns, 1, 1))  # one flat-spectrum component per plane
    ds = build_mds_dataset(
        coeffs,
        np.array([ny // 2]),
        np.array([nx // 2]),
        "t0",
        ["t0"],
        "t",
        "f",
        np.array([1.62393461e9]),
        freqs,
        cell_rad,
        nx,
        ny,
        0.0,
        0.0,
        False,
        True,
        False,
        radec,
        list(stokes),
        "test",
        freq_bounds=np.stack([freqs - 5e7, freqs + 5e7], axis=1) if freq_bounds else None,
        flux_scale=flux_scale,
    )
    ds.to_zarr(str(path), mode="w")
    return str(path)


def _no_ray(monkeypatch):
    import pfb_imaging.core.imager as imager_mod

    def _boom(*a, **kw):
        raise AssertionError("init_ray was called: the guard fired too late")

    monkeypatch.setattr(imager_mod, "init_ray", _boom)


def test_transfer_refuses_a_missing_product_before_ray(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    mds = _write_mds(tmp_path / "q.mds", radec=_ms_radec(ms_name), cell_rad=1e-5, stokes=("Q",))
    with pytest.raises(ValueError, match="missing"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            product="I",
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_transfer_refuses_a_missing_path_before_ray(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    with pytest.raises(ValueError, match="No .mds"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            transfer_model_from=str(tmp_path / "nope.mds"),
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_transfer_refuses_an_apparent_model_before_ray(ms_name, tmp_path, monkeypatch):
    """Pass 2 applies BEAM to the model, so an apparent one would be attenuated twice."""
    _no_ray(monkeypatch)
    mds = _write_mds(tmp_path / "app.mds", radec=_ms_radec(ms_name), cell_rad=1e-5, flux_scale="apparent")
    with pytest.raises(ValueError, match="intrinsic-flux model"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            product="I",
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


@pytest.mark.parametrize("flux_scale", ["intrinsic", None])
def test_transfer_accepts_an_intrinsic_or_legacy_model(ms_name, tmp_path, monkeypatch, flux_scale):
    """Reaching init_ray means every pre-Ray transfer check passed, the geometry included."""
    _no_ray(monkeypatch)
    # a model cell much finer than the run's: coarsening is no longer refused
    mds = _write_mds(tmp_path / "ok.mds", radec=_ms_radec(ms_name), cell_rad=1e-9, flux_scale=flux_scale)
    with pytest.raises(AssertionError, match="init_ray was called"):
        imager_core(
            [Path(ms_name)],
            str(tmp_path / "out"),
            product="I",
            field_of_view=1.0,
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_check_transfer_geometry(tmp_path):
    from pfb_model_spec.utils.spec import open_mds

    from pfb_imaging.core.imager import check_transfer_geometry

    radec = (1.0, -0.5)
    ds = open_mds(_write_mds(tmp_path / "m.mds", radec=radec, cell_rad=1e-5))

    # the cell no longer enters: the render remaps flux-conservingly (pfb-model-spec#32)
    check_transfer_geometry(ds, grid_radec=radec)
    # RA either side of zero is a tiny separation, not ~2 pi
    ds0 = open_mds(_write_mds(tmp_path / "z.mds", radec=(1e-12, -0.5), cell_rad=1e-5))
    check_transfer_geometry(ds0, grid_radec=(2 * np.pi - 1e-12, -0.5))

    with pytest.raises(ValueError, match="[Tt]angent"):
        check_transfer_geometry(ds, grid_radec=(1.0, -0.4))


def test_transfer_coverage_warning_uses_the_rendered_frequencies(tmp_path):
    from pfb_model_spec.utils.spec import open_mds

    from pfb_imaging.core.imager import transfer_coverage_warning, transfer_freq_coverage

    radec = (1.0, -0.5)
    # freq_bounds = fitted freqs -/+ 50 MHz
    cov = transfer_freq_coverage(open_mds(_write_mds(tmp_path / "m.mds", radec=radec, cell_rad=1e-5)))
    assert cov == (0.95e9, 1.15e9, True)
    assert transfer_coverage_warning([0.96e9, 1.14e9], cov) is None
    msg = transfer_coverage_warning([1.0e9, 1.3e9], cov)
    assert "extrapolated" in msg and "1300.000 MHz" in msg and "freq_bounds)" in msg

    # a legacy .mds: coverage is the fitted (effective) frequencies, so a run with
    # identical settings renders exactly at them and must not warn -- the nominal
    # band centres, which lie outside them, never enter the check
    legacy = open_mds(_write_mds(tmp_path / "l.mds", radec=radec, cell_rad=1e-5, freq_bounds=False))
    cov = transfer_freq_coverage(legacy)
    assert cov == (1.0e9, 1.1e9, False)
    assert transfer_coverage_warning(legacy.freqs.values * (1 + 1e-12), cov) is None
    msg = transfer_coverage_warning([0.9e9, 1.05e9], cov)
    assert "900.000 MHz" in msg and "1050" not in msg and "records no freq_bounds" in msg


def test_transfer_refuses_a_tangent_point_mismatch_before_ray(ms_name, tmp_path, monkeypatch):
    """Geometry is only known after the MS metadata read, which still precedes init_ray."""
    _no_ray(monkeypatch)
    ra, dec = _ms_radec(ms_name)
    mds = _write_mds(tmp_path / "off.mds", radec=(ra, dec + 1e-3), cell_rad=1e-5)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="[Tt]angent point mismatch"):
        imager_core(
            [Path(ms_name)],
            str(out),
            channels_per_image=2,
            product="I",
            field_of_view=1.0,
            transfer_model_from=mds,
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )
    assert not list(Path(f"{out}_I.scratch").glob("band*"))


def test_transfer_path_is_resolved_to_absolute(ms_name, tmp_path, monkeypatch):
    _no_ray(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _write_mds(tmp_path / "rel.mds", radec=_ms_radec(ms_name), cell_rad=1e-5, stokes=("Q",))
    # relative path that only resolves from tmp_path; the product refusal proves it was opened
    with pytest.raises(ValueError, match="missing"):
        imager_core(
            [Path(ms_name)],
            "out",
            product="I",
            transfer_model_from="rel.mds",
            overwrite=True,
            keep_ray_alive=True,
            log_directory=str(tmp_path / "logs"),
        )


def test_transfer_planes_follow_the_band_corr():
    from pfb_imaging.core.imager import _transfer_planes

    assert _transfer_planes(["I", "Q", "U", "V"], ["I"]) == [0]
    assert _transfer_planes(["I", "V"], ["V", "I"]) == [1, 0]


def test_iter_parts_keeps_the_phase_centre_attrs(tmp_path):
    """A --target run's residual needs each part's l0/m0 (residual_from_partitions reads them)."""
    import xarray as xr

    from pfb_imaging.core.imager import _iter_parts

    store = str(tmp_path / "x.dt")
    part = xr.Dataset(
        {
            "UVW": (("row", "three"), np.zeros((2, 3))),
            "WEIGHT": (("corr", "row", "chan"), np.ones((1, 2, 1))),
            "MASK": (("row", "chan"), np.ones((2, 1), np.uint8)),
            "FREQ": (("chan",), np.array([1e9])),
            "BEAM": (("corr", "y", "x"), np.ones((1, 4, 4))),
            "VIS": (("corr", "row", "chan"), np.zeros((1, 2, 1), np.complex64)),
        },
        attrs={"l0": 1.5e-4, "m0": -2.5e-4},
    )
    part.to_zarr(store, group="band0000_time0000/part0000", mode="w", consolidated=False)
    (got,) = list(_iter_parts(store, "band0000_time0000", 1))
    assert got.attrs["l0"] == 1.5e-4 and got.attrs["m0"] == -2.5e-4
    assert "VIS" not in got and got.WEIGHT.shape == (1, 2, 1)


def test_residual_sweep_holds_one_streamed_partition():
    """Partition N is freed before the iterator loads N+1, so pass 2 holds one at a time."""
    import weakref

    import xarray as xr

    from pfb_imaging.operators.gridder import residual_from_partitions

    held = []  # whether the previous partition's arrays were still alive at each advance

    def parts(k):
        prev = None
        for _ in range(k):
            held.append(prev is not None and prev() is not None)
            sub = xr.Dataset(
                {
                    "UVW": (("row", "three"), np.array([[10.0, 20.0, 0.0], [30.0, -5.0, 0.0]])),
                    "WEIGHT": (("corr", "row", "chan"), np.ones((1, 2, 1), np.float32)),
                    "MASK": (("row", "chan"), np.ones((2, 1), np.uint8)),
                    "FREQ": (("chan",), np.array([1e9])),
                    "BEAM": (("corr", "y", "x"), np.ones((1, 16, 16), np.float32)),
                }
            )
            prev = weakref.ref(sub["WEIGHT"].values)
            yield sub
            del sub

    dirty = np.zeros((1, 16, 16), np.float32)
    model = np.zeros_like(dirty)
    model[0, 8, 8] = 1
    residual_from_partitions(dirty, parts(3), model, 1e-5, epsilon=1e-4, do_wgridding=False)
    assert held == [False, False, False]


@pytest.mark.slow
def test_transferred_truth_nulls_the_residual(sky_truth, ms_meta, ms_name, tmp_path):
    """Transfer the injected sky; pass 2's RESIDUAL must be ~0 and deconv must resume.

    Same construction as test_imager_degrid_nulls_the_residual, but the
    subtraction happens in the imager's pass 2 instead of via degrid.
    """
    import xarray as xr
    from pfb_model_spec.utils.io import model_to_ds

    from pfb_imaging.operators.gridder import wgridder_conventions

    common = dict(
        channels_per_image=4,
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        psf=True,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    first = str(tmp_path / "first")
    imager_core([Path(ms_name)], first, **common)
    dt = xr.open_datatree(first + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        freq_out = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])
        time_out = np.array([float(dt[nodes[0]].ds.attrs["time_out"])])
        radec = (float(dt[nodes[0]].ds.attrs["ra"]), float(dt[nodes[0]].ds.attrs["dec"]))
        wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
        dirty_peak = float(np.abs(sum(dt[n].ds.DIRTY[0] for n in nodes).values).max() / wsum)
        bdirty_peak = float(np.abs(sum(dt[n].ds.BDIRTY[0] for n in nodes).values).max() / wsum)
    finally:
        dt.close()

    nx, ny = sky_truth.nx, sky_truth.ny
    model = np.zeros((freq_out.size, 1, ny, nx))
    for b, f in enumerate(freq_out):
        for s in range(sky_truth.lpix.size):
            ix = nx // 2 - int(sky_truth.lpix[s])
            iy = ny // 2 + int(sky_truth.mpix[s])
            model[b, 0, iy, ix] = sky_truth.ref_flux[s] * (f / ms_meta.freq0) ** sky_truth.alpha[s]
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    mds_path = str(tmp_path / "truth.mds")
    model_to_ds(
        time=time_out,
        freq=freq_out,
        fsel=np.ones(freq_out.size, dtype=bool),
        model=model,
        wgt=np.ones((freq_out.size, 1)),
        mds_name=mds_path,
        cell_rad=sky_truth.cell_rad,
        nx=nx,
        ny=ny,
        x0=x0,
        y0=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        radec=radec,
        stokes=["I"],
        writer_version="test",
    )

    second = str(tmp_path / "second")
    fits_dir = tmp_path / "fits"
    imager_core(
        [Path(ms_name)],
        second,
        transfer_model_from=mds_path,
        **{**common, "fits_mfs": True},
        fits_output_folder=str(fits_dir),
    )
    # the transferred residual reaches FITS beside the dirty image
    assert list(fits_dir.glob("*_residual_*mfs.fits")) and list(fits_dir.glob("*_dirty_*mfs.fits"))
    dt = xr.open_datatree(second + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        for n in nodes:
            band = dt[n].ds
            assert {"MODEL", "RESIDUAL", "BRESIDUAL"} <= set(band.data_vars)
            assert band.attrs["transferred_from"] == mds_path
            assert band.MODEL.dtype == band.DIRTY.dtype
        residual = sum(dt[n].ds.RESIDUAL[0] for n in nodes).values
        peak = float(np.abs(residual).max() / wsum)
        bresidual = sum(dt[n].ds.BRESIDUAL[0] for n in nodes).values
        bpeak = float(np.abs(bresidual).max() / wsum)
    finally:
        dt.close()
    assert peak < 0.05 * dirty_peak, f"residual peak {peak:.3e} vs dirty peak {dirty_peak:.3e}"
    # BRESIDUAL = BDIRTY - sum_p B_p G_p^T W_p G_p (B_p model) is the beam-attenuated
    # gradient; deconv normalises it by the same total wsum as RESIDUAL. Its model-free
    # term is BDIRTY, not DIRTY, so BDIRTY's peak is the like-for-like reference.
    assert bpeak < 0.05 * bdirty_peak, f"bresidual peak {bpeak:.3e} vs bdirty peak {bdirty_peak:.3e}"

    # onto a x2 coarser grid over the same field: the flux-conserving remap
    # (landmanbester/pfb-model-spec#32) keeps every band's model flux
    coarse = str(tmp_path / "coarse")
    imager_core(
        [Path(ms_name)],
        coarse,
        transfer_model_from=mds_path,
        **{**common, "nx": nx // 2, "ny": ny // 2, "cell_size": 2 * sky_truth.cell_size, "psf": False},
    )
    dt = xr.open_datatree(coarse + "_I.dt", engine="zarr", chunks=None)
    try:
        cnodes = sorted(n for n in dt.children if n.startswith("band"))
        cflux = np.array([float(dt[n].ds.MODEL[0].sum()) for n in cnodes])
        cwsum = sum(float(dt[n].ds.WSUM.values[0]) for n in cnodes)
        cdirty_peak = float(np.abs(sum(dt[n].ds.DIRTY[0] for n in cnodes).values).max() / cwsum)
        cpeak = float(np.abs(sum(dt[n].ds.RESIDUAL[0] for n in cnodes).values).max() / cwsum)
    finally:
        dt.close()
    # the old bilinear-times-area-ratio render inflated a point's flux x4 here
    np.testing.assert_allclose(cflux, model.sum(axis=(1, 2, 3)), rtol=1e-5)
    # not ~0 like the same-grid run: each source lands on its coarse pixel's centre,
    # off its true position by up to half a coarse cell (measured 8.7%)
    assert cpeak < 0.15 * cdirty_peak, f"coarse residual peak {cpeak:.3e} vs dirty peak {cdirty_peak:.3e}"

    # deconv resumes from the transferred tree without the "MODEL but no BRESIDUAL" refusal
    from pfb_imaging.core.deconv import deconv as deconv_core

    deconv_core(
        second,
        product="I",
        niter=1,
        nthreads=1,
        fits_mfs=False,
        fits_cubes=False,
        log_directory=str(tmp_path / "logs"),
    )
