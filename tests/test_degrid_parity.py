"""`degrid` vs `degrid-msv4`, and the end-to-end null (issue #278).

The two legacy-comparing tests are deletable with the old command -- that is
the point of them. They exist to make the swap boring.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def parity_ms(ms_name, tmp_path):
    """A private MS copy, zeroed MODEL_DATA, safe to degrid into."""
    from casacore.tables import table as pctable

    dest = tmp_path / "parity.ms"
    shutil.copytree(ms_name, dest)
    with pctable(str(dest), readonly=False, ack=False) as tab:
        tab.putcol("MODEL_DATA", np.zeros((tab.nrows(), 8, 4), np.complex64))
    return str(dest)


def test_new_kernel_matches_the_legacy_inline_evaluation(parity_ms, simple_mds):
    """The pfb-model-spec kernel must reproduce `_comps2vis_impl` exactly.

    `core/degrid.py` never used pfb-model-spec: `operators/gridder` evaluated
    the model inline as `image[x_index, y_index] = modelf(tout, fout, *comps)`.
    The kernel's `eval_coeffs_to_slice` does the same thing and then
    short-circuits its resampling branch when the output grid is the model's
    own -- which is always, here -- so the two are bit-identical, not merely
    close.

    Cheap by design: no dask cluster and no distributed scheduler, so this
    guard runs in the fast loop while the full command comparison does not.
    """
    import sympy as sm
    import xarray as xr
    from africanus.model.coherency import convert
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.utilities.lambdify import lambdify

    from pfb_imaging.operators.gridder import _comps2vis_impl
    from pfb_imaging.utils.degrid_msv4 import build_region_masks, degrid_region
    from pfb_imaging.utils.msv4 import get_engine, select_vis_nodes

    _, mds_ds = simple_mds

    dt = xr.open_datatree(parity_ms, **get_engine(parity_ms))
    try:
        node = select_vis_nodes(dt)[0]
        node_ds = dt[node.path].ds
        region = {"time": slice(0, 60), "frequency": slice(0, 8)}
        sub = node_ds.isel(**region)
        uvw = sub.UVW.values.reshape(-1, 3)
        # the legacy path has no visibility mask, so it only agrees where
        # there is no NaN padding. Assert that rather than assume it.
        assert np.isfinite(uvw).all(), "test MS unexpectedly carries padded rows"
        utime = sub.time.values
        freq = sub.frequency.values
        nrow = uvw.shape[0]

        got = degrid_region(
            node_ds,
            region=region,
            model_ds=mds_ds,
            masks=build_region_masks(mds_ds, None),
            columns=["MODEL_DATA"],
            corr_types=node.corr_types,
        ).MODEL_DATA.values.reshape(nrow, freq.size, 4)

        # the legacy path, driven directly: one time bin, one frequency bin
        params = sm.symbols(("t", "f")) + sm.symbols(tuple(mds_ds.params.values))
        modelf = lambdify(params, parse_expr(mds_ds.parametrisation))
        tfunc = lambdify(params[0], parse_expr(mds_ds.texpr))
        ffunc = lambdify(params[1], parse_expr(mds_ds.fexpr))
        legacy = _comps2vis_impl(
            uvw,
            utime,
            freq,
            np.array([0]),  # rbin_idx
            np.array([nrow]),  # rbin_cnts
            np.array([0]),  # tbin_idx
            np.array([utime.size]),  # tbin_cnts
            np.array([0]),  # fbin_idx
            np.array([freq.size]),  # fbin_cnts
            np.ones((int(mds_ds.npix_x), int(mds_ds.npix_y))),
            mds_ds,
            modelf,
            tfunc,
            ffunc,
            epsilon=1e-7,
            nthreads=1,
            do_wgridding=True,
            divide_by_n=False,
            product="I",
        )
        legacy = convert(legacy.astype(np.complex64), ["I"], ["XX", "XY", "YX", "YY"], implicit_stokes=True)
        np.testing.assert_array_equal(got, legacy.astype(np.complex64))
    finally:
        dt.close()


@pytest.mark.slow
def test_degrid_and_degrid_msv4_agree_on_the_same_ms(ms_name, simple_mds, tmp_path):
    """The two commands must fill MODEL_DATA identically.

    The highest-value guard for the port and nearly free while both commands
    exist. Deleted along with `degrid` itself.

    Chunking is matched deliberately: the model is re-rendered once per chunk,
    so `channels_per_image=8` (one band over the whole 8-channel SPW) has to
    pair with `channels_per_chunk=8`, or the two commands would evaluate the
    model at different frequencies and legitimately disagree.
    """
    from casacore.tables import table as pctable

    from pfb_imaging.core.degrid import degrid as degrid_core
    from pfb_imaging.core.degrid_msv4 import degrid_msv4

    mds_path, _ = simple_mds

    old_ms = tmp_path / "old.ms"
    new_ms = tmp_path / "new.ms"
    for dest in (old_ms, new_ms):
        shutil.copytree(ms_name, dest)
        with pctable(str(dest), readonly=False, ack=False) as tab:
            tab.putcol("MODEL_DATA", np.zeros((tab.nrows(), 8, 4), np.complex64))

    # the legacy core annotates `ms: list[Path]` but does `ms_name.rstrip("/")`,
    # so it actually requires strings. A latent bug in the command being retired.
    degrid_core(
        [str(old_ms)],
        str(tmp_path / "old_out"),
        mds=mds_path,
        product="I",
        integrations_per_image=-1,
        channels_per_image=8,
        nworkers=1,
        nthreads=1,
        log_directory=str(tmp_path / "old_logs"),
    )
    degrid_msv4(
        [Path(new_ms)],
        str(tmp_path / "new_out"),
        8,
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "new_logs"),
    )

    with pctable(str(old_ms), ack=False) as tab:
        old = np.asarray(tab.getcol("MODEL_DATA"))
    with pctable(str(new_ms), ack=False) as tab:
        new = np.asarray(tab.getcol("MODEL_DATA"))

    assert np.any(old != 0), "the legacy command wrote nothing"
    # the gridder is called with identical arguments on both paths, so this is
    # equality at gridder epsilon, not a tolerance negotiation
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_imager_degrid_msv4_nulls_the_residual(sky_truth, ms_meta, ms_name, tmp_path):
    """The full loop must subtract the sky it degridded.

    `imager --psf` -> real `.mds` -> `degrid-msv4` -> `imager` on
    `DATA-MODEL_DATA`. The model here is the injected truth rather than a
    deconvolved estimate, written through pfb-model-spec's own `model_to_ds`,
    so the `.mds` schema, the component fit and the re-render are all the
    production ones and the null is strict: no beam anywhere on this path, and
    a perfect model, so what is left is gridder error.

    **Why `deconv` is not in this chain.** It cannot be, in this test session.
    A single-band run makes `fit_image_cube` raise `UnboundLocalError` (`xfit`
    is only assigned inside its multi-band branches) and `deconv` swallows
    that in a bare `except Exception`, so no `.mds` is written at all --
    silently. More than one band makes `deconv` stand up one long-lived Ray
    actor per band, which starves on the session cluster's `RAY_NUM_CPUS=2`
    (the same constraint `test_deconv_groundtruth` documents when it pins
    itself to `channels_per_image=-1`). So there is no band count at which
    `deconv -> degrid` runs here. Using the truth as the model tests strictly
    more of degrid than a deconvolved model would, and the `.mds` is still
    produced by the real writer.
    """
    import xarray as xr
    from pfb_model_spec.utils.io import model_to_ds

    from pfb_imaging.core.degrid_msv4 import degrid_msv4
    from pfb_imaging.core.imager import imager as imager_core
    from pfb_imaging.operators.gridder import wgridder_conventions

    work_ms = tmp_path / "e2e.ms"
    shutil.copytree(ms_name, work_ms)

    outname = str(tmp_path / "e2e")
    imager_core(
        [Path(work_ms)],
        outname,
        channels_per_image=4,  # 2 bands: fit_image_cube needs more than one
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

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        assert len(nodes) == 2, f"expected 2 bands, got {nodes}"
        freq_out = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])
        time_out = np.array([float(dt[nodes[0]].ds.attrs["time_out"])])
        radec = (float(dt[nodes[0]].ds.attrs["ra"]), float(dt[nodes[0]].ds.attrs["dec"]))
        dirty = sum(dt[n].ds.DIRTY[0] for n in nodes).values
        wsum_total = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
        dirty_peak = float(np.abs(dirty).max() / wsum_total)
    finally:
        dt.close()

    # the injected truth on the imager's own grid, x-major (nband, nx, ny)
    nx, ny = sky_truth.nx, sky_truth.ny
    model = np.zeros((freq_out.size, nx, ny))
    for b, f in enumerate(freq_out):
        for s in range(sky_truth.lpix.size):
            ix = nx // 2 - int(sky_truth.lpix[s])
            iy = ny // 2 + int(sky_truth.mpix[s])
            model[b, ix, iy] = sky_truth.ref_flux[s] * (f / ms_meta.freq0) ** sky_truth.alpha[s]

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    mds_path = str(tmp_path / "truth.mds")
    model_to_ds(
        time_out,
        freq_out,
        np.ones(freq_out.size, dtype=bool),  # fsel: fit every band
        model,  # (nband, nx, ny); model_to_ds adds the time axis itself
        np.ones(freq_out.size),  # per-band fit weight
        mds_path,
        sky_truth.cell_rad,
        nx,
        ny,
        x0,
        y0,
        flip_u,
        flip_v,
        flip_w,
        radec,
        "I",
        "test",
    )
    assert Path(mds_path).exists()

    degrid_msv4(
        [Path(work_ms)],
        outname,
        4,
        mds=mds_path,
        product="I",
        integrations_per_chunk=-1,
        nworkers=1,
        nthreads=1,
        progressbar=False,
        log_directory=str(tmp_path / "logs"),
    )

    resname = str(tmp_path / "e2e_res")
    imager_core(
        [Path(work_ms)],
        resname,
        data_column="DATA-MODEL_DATA",
        channels_per_image=4,
        integrations_per_image=-1,
        product="I",
        nx=nx,
        ny=ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    dt = xr.open_datatree(resname + "_I.dt", engine="zarr", chunks=None)
    try:
        nodes = sorted(n for n in dt.children if n.startswith("band"))
        residual = sum(dt[n].ds.DIRTY[0] for n in nodes).values
        wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)
        peak = float(np.abs(residual).max() / wsum)
    finally:
        dt.close()

    # Stated against the pre-subtraction dirty peak, so an all-zero
    # MODEL_DATA (peak == dirty_peak) cannot pass and the threshold does not
    # silently encode a model-quality assumption.
    assert peak < 0.05 * dirty_peak, (
        f"residual peak {peak:.3e} vs dirty peak {dirty_peak:.3e} -- the degridded model did not subtract"
    )
