"""Tests for the ``.dt``-native ``pfb deconv`` driver.

``test_deconv_groundtruth`` runs ``imager``+``deconv`` on simulated
visibilities predicted from an injected ``sky_truth`` sky and checks recovery
against that ground truth directly (no legacy oracle). The remaining tests
build a synthetic ``.dt`` store in-process (``_write_synthetic_dt``, no
MS/imager needed) for fast smoke coverage: ``test_deconv_two_band_smoke``
guards the nband>1 Ray actor-pool deadlock, and
``test_band_workers_load_matches_driver_side`` checks that band-worker-side
loading of vis-scale inputs from the store reproduces driver-side reads
exactly.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.mark.slow
def test_deconv_groundtruth(sky_truth, ms_name, tmp_path):
    """deconv on the noiseless predicted sky recovers the injected fluxes.

    Replaces test_deconv_matches_legacy_sara: the reference is the injected
    truth itself rather than the legacy sara implementation. The wavelet
    model legitimately spreads a point source over neighbouring pixels, so
    flux is asserted as a +/-4-pixel box sum (measured recovery ~1.08-1.12x
    after 5 cycles at eta=0.001; single-pixel values plateau near ~40%),
    position as the box argmax, and convergence as the normalised residual
    peak dropping well below the faintest source.

    Single band (channels_per_image=-1) and nthreads=1 on purpose: for
    nband==1 the Hessian/Psi pools use their local in-process path, keeping
    the long-lived Ray-actor CPU claims within the session cluster's
    num_cpus=1 (see tests/conftest.py) -- multi-band actor distribution is
    covered by test_hess_tree_ray.py/test_psi_operator.py.
    """
    from pfb_imaging.core.deconv import deconv as deconv_core
    from pfb_imaging.core.imager import imager as imager_core

    outname = str(tmp_path / "gtdeconv")
    imager_core(
        [Path(ms_name)],
        outname,
        channels_per_image=-1,
        integrations_per_image=-1,
        product="I",
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        robustness=0.0,
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )
    deconv_core(
        outname,
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=5,
        gamma=1.0,
        eta=0.001,
        rmsfactor=1.0,
        init_factor=1.0,
        l1_reweight_from=100,  # disabled within these few major cycles
        bases=["self", "db1"],
        nlevels=2,
        positivity=1,
        pd_tol=1e-6,
        pd_maxit=5000,
        cg_tol=1e-6,
        cg_maxit=3000,
        pm_tol=1e-4,
        pm_maxit=200,
        nthreads=1,
        do_wgridding=True,
        epsilon=1e-7,
        fits_mfs=False,
        fits_cubes=False,
        fits_per_partition=True,
        verbosity=0,
    )

    dt = xr.open_datatree(outname + "_I.dt", engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    for n in nodes:
        assert "MODEL" in dt[n].ds and "UPDATE" in dt[n].ds
    model_mean = sum(dt[n].ds.MODEL[0] for n in nodes) / len(nodes)
    residual = sum(dt[n].ds.RESIDUAL[0] for n in nodes).values
    wsum = sum(float(dt[n].ds.WSUM.values[0]) for n in nodes)

    # converged: normalised residual peak well below the faintest source
    assert np.abs(residual).max() / wsum < 0.1 * sky_truth.ref_flux.min()

    half = 4  # box half-width for the flux sums
    for s in range(sky_truth.lpix.size):
        ixm = sky_truth.nx // 2 - int(sky_truth.lpix[s])
        iym = sky_truth.ny // 2 + int(sky_truth.mpix[s])
        box = model_mean.isel(x=slice(ixm - half, ixm + half + 1), y=slice(iym - half, iym + half + 1))
        # the n-term is folded into the stored BEAM (D22), so the model is in
        # intrinsic flux units -- no legacy *n correction
        got = float(box.values.sum())
        want = sky_truth.ref_flux[s]
        assert abs(got - want) < 0.2 * want, f"source {s}: box flux {got} vs {want}"
        # the box is centred on the source in both axes, so the argmax must
        # sit at its centre regardless of the ('x','y')/('y','x') dim order
        i0, i1 = np.unravel_index(int(np.argmax(box.values)), box.shape)
        assert (i0, i1) == (half, half), f"source {s}: model peak off-centre ({i0},{i1})"

    # mopped products (#311). --mop is on by default, so this run wrote them.
    # The claim is that model + M^-1 r very nearly cancels the residual:
    # A(m + M^-1 r) = A m + A M^-1 r ~= A m + r = data wherever M ~= A. Needs
    # consistent data to mean anything, which is why it is asserted here and
    # not on the synthetic tree.
    for n in nodes:
        assert "MODEL_MOPPED" in dt[n].ds and "RESIDUAL_MOPPED" in dt[n].ds
    residual_mopped = sum(dt[n].ds.RESIDUAL_MOPPED[0] for n in nodes).values
    peak, peak_mopped = np.abs(residual).max(), np.abs(residual_mopped).max()
    assert peak_mopped < 0.5 * peak, f"mopped peak {peak_mopped:.3e} vs deconvolved {peak:.3e}"
    # and it must be a different model, not a copy of MODEL
    assert not np.allclose(dt[nodes[0]].ds.MODEL.values, dt[nodes[0]].ds.MODEL_MOPPED.values)

    # per-partition debug FITS: with a single partition per band, the
    # re-gridded partition residual must reproduce the stored band RESIDUAL
    import glob

    from astropy.io import fits as afits

    pdir = str(tmp_path / "fits" / "gtdeconv_I_main_partitions")
    hits = sorted(glob.glob(f"{pdir}/residual_band*_part0000_*.fits"))
    assert len(hits) == len(nodes), f"expected {len(nodes)} partition residual FITS"
    n0 = nodes[0]
    wsum_p = float(np.asarray(dt[n0][sorted(dt[n0].children)[0]].ds.attrs["wsum"]).ravel()[0])
    with afits.open(hits[0]) as hdul:
        img = np.squeeze(hdul[0].data).astype(np.float64)
    ref = dt[n0].ds.RESIDUAL.values[0] / wsum_p
    np.testing.assert_allclose(img, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def _write_synthetic_dt(store, nx, ny, nrow, nchan, rng, parts_per_band=(1, 1)):
    """Build a minimal synthetic 2-band .dt store matching what core/deconv.py reads.

    No MS / imager pipeline involved -- just the native DataTree groups the
    driver's ``deconv()`` opens directly (see architecture.md §8 tree layout).
    ``parts_per_band`` sets the number of ``part####`` children per band --
    bands legitimately carry different partition counts when a field's chunk
    is fully flagged (stokes_vis writes no scratch piece for it).
    """
    nx_psf, ny_psf = 2 * nx, 2 * ny
    xo2 = nx + 1
    freqs = [1e9, 1.1e9]

    for b, freq in enumerate(freqs):
        bandname = f"band{b:04d}_time0000"
        dirty = rng.standard_normal((1, ny, nx))
        band_ds = xr.Dataset(
            data_vars={
                "DIRTY": (("corr", "y", "x"), dirty),
                # unit partition beams -> BDIRTY == DIRTY (D23)
                "BDIRTY": (("corr", "y", "x"), dirty.copy()),
                "RESIDUAL": (("corr", "y", "x"), rng.standard_normal((1, ny, nx))),
                "PSF": (("corr", "y_psf", "x_psf"), np.ones((1, ny_psf, nx_psf))),
                "WSUM": (("corr",), np.array([1.0])),
            },
            coords={"corr": ["I"]},
            attrs={
                "bandid": b,
                "timeid": 0,
                "freq_out": freq,
                "time_out": 1.7e9,
                "ra": 0.0,
                "dec": 0.0,
                "cell_rad": 2.5e-6,
                "niters": 0,
            },
        )
        band_ds.to_zarr(store, group=bandname, mode="a")

        for pid in range(parts_per_band[b]):
            uvw = rng.uniform(-50.0, 50.0, size=(nrow, 3))
            part_ds = _make_part(uvw, nrow, nchan, nx, ny, ny_psf, xo2, freq, parts_per_band[b])
            part_ds.to_zarr(store, group=f"{bandname}/part{pid:04d}", mode="a")


def _make_part(uvw, nrow, nchan, nx, ny, ny_psf, xo2, freq, nparts):
    return xr.Dataset(
        data_vars={
            # delta-function PSF -> Fourier-domain magnitude is all ones
            # (scaled by the per-part wsum share so the band's Hessian stays
            # the identity), matching the abs()'d PSFHAT convention
            # core/deconv.py expects.
            "PSFHAT": (("corr", "y_psf", "xo2"), np.full((1, ny_psf, xo2), 1.0 / nparts)),
            "BEAM": (("corr", "y", "x"), np.ones((1, ny, nx))),
            "VIS": (("corr", "row", "chan"), np.ones((1, nrow, nchan), dtype=np.complex128)),
            "UVW": (("row", "three"), uvw),
            "WEIGHT": (("corr", "row", "chan"), np.ones((1, nrow, nchan))),
            "MASK": (("row", "chan"), np.ones((nrow, nchan), dtype=np.uint8)),
            "FREQ": (("chan",), np.array([freq])),
        },
        attrs={
            "wsum": [1.0 / nparts],
            "l0": 0.0,
            "m0": 0.0,
            "msid": 0,
            "field_name": "f0",
            "spw_name": "s0",
            "baseline_group": "all",
        },
    )


@pytest.mark.timeout(120)
@pytest.mark.slow
def test_deconv_two_band_smoke(tmp_path):
    """Multi-band driver smoke test: regression guard for the nband>1 Ray deadlock.

    Builds a synthetic 2-band .dt store directly (no MS/imager needed) and
    runs the deconv driver with default-ish worker settings (nworkers=1):
    this MUST NOT hang (see the HessTreeRay/PsiNocopytRay actor-pool CPU
    claim fix -- previously the aggregate actor CPU claim could exceed the
    driver's Ray cluster capacity for nband > 1).
    """
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(42)
    nx = ny = 32
    nrow, nchan = 64, 1

    output_filename = str(tmp_path / "synth")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng)

    deconv_core(
        output_filename,
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=1.0,
        pd_maxit=20,
        cg_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        fits_per_partition=True,
        debug=True,
        verbosity=0,
    )

    # --debug: chi2 trajectories (baseline snapshot + 1 iteration) and
    # baseline-binned residual profiles in a machine-readable JSON
    import json

    with open(tmp_path / "fits" / "synth_I_main_debug.json") as f:
        rec = json.load(f)
    assert len(rec["iterations"]) == 2  # iter0 baseline + niter=1
    for entry in rec["iterations"]:
        assert len(entry["bands"]) == 2
        for bnd in entry["bands"]:
            for p in bnd["partitions"]:
                assert np.isfinite(p["chi2"][0]) and p["ndata"] > 0
    assert set(rec["uv_profiles"]) == {e["band"] for e in rec["iterations"][0]["bands"]}
    prof = next(iter(rec["uv_profiles"].values()))[0]
    assert len(prof["uvdist_edges_lambda"]) == len(prof["resid_power"]) + 1
    assert np.isfinite(prof["resid_power"]).all() and sum(prof["count"]) > 0

    # per-partition debug FITS written at the end of the run
    import glob

    from astropy.io import fits as afits

    pdir = str(tmp_path / "fits" / "synth_I_main_partitions")
    for var in ("dirty", "residual", "model_apparent"):
        hits = glob.glob(f"{pdir}/{var}_band*_part*.fits")
        assert len(hits) == 2, f"{var}: expected 2 partition FITS, got {len(hits)}"
    with afits.open(sorted(glob.glob(f"{pdir}/residual_band*.fits"))[0]) as hdul:
        assert np.isfinite(hdul[0].data).all()
        for card in ("WSUMP", "CHI2", "NDATA", "RCHI2", "FIELDNAM"):
            assert card in hdul[0].header

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    assert len(nodes) == 2
    for n in nodes:
        ds = dt[n].ds
        assert "MODEL" in ds and "UPDATE" in ds
        assert "BRESIDUAL" in ds  # gradient residual written back (D23)
        assert ds.attrs["niters"] == 1
        assert np.isfinite(ds.MODEL.values).all()


@pytest.mark.timeout(120)
@pytest.mark.slow
def test_band_workers_load_matches_driver_side(tmp_path):
    """Worker-side load_bands reproduces driver-side reads exactly.

    The band workers read their own vis-scale inputs (PSFHAT/BEAM/wsum for
    the Hessian, UVW/WEIGHT/MASK/FREQ/BEAM/DIRTY for the exact residual)
    straight from the .dt store; this checks the resulting operators against
    the same data loaded in the test process.
    """
    from numpy.testing import assert_allclose

    from pfb_imaging.operators.band_worker import BandWorkerPool
    from pfb_imaging.operators.gridder import residual_from_partitions
    from pfb_imaging.operators.hessian import HessianTree, HessTreeRay

    rng = np.random.default_rng(99)
    nx = ny = 16
    nrow, nchan = 32, 1
    dt_name = str(tmp_path / "synth_I.dt")
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng)

    # randomise the PSF magnitudes so the Hessian check is non-trivial
    import zarr

    root = zarr.open_group(dt_name, mode="a")
    for _, grp in root.groups():
        # non-trivial BDIRTY so the gradient-residual check is meaningful
        grp["BDIRTY"][:] = rng.standard_normal(grp["BDIRTY"].shape)
        for _, child in grp.groups():
            child["PSFHAT"][:] = rng.uniform(0.5, 2.0, size=child["PSFHAT"].shape)
            # distinct non-unit per-partition beams: the exact gradient is the
            # per-partition beam-weighted sum, not a band-average (D23)
            child["BEAM"][:] = rng.uniform(0.3, 1.0, size=child["BEAM"].shape)
    zarr.consolidate_metadata(dt_name)

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    nband = len(nodes)
    cell_rad = dt[nodes[0]].ds.attrs["cell_rad"]

    pool = BandWorkerPool(nband, nthreads=1)
    pool.load_bands(dt_name, nodes)
    hess = HessTreeRay(None, nx, ny, 2 * nx, 2 * ny, etas=0.1, wsums=1.0, workers=pool)

    x = rng.standard_normal((nband, nx, ny))
    out_pool = hess.dot(x)
    model = rng.standard_normal((nband, 1, nx, ny))
    res_pool, bres_pool = pool.residual(model, cell_rad)

    for b, n in enumerate(nodes):
        band = dt[n]
        parts, hess_parts = [], []
        for cname in sorted(band.children):
            child = band[cname].ds
            pds = child[["UVW", "WEIGHT", "MASK", "FREQ", "BEAM"]].load()
            pds.attrs.update(child.attrs)
            hess_parts.append(
                {
                    "psfhat": np.abs(child.PSFHAT.values),
                    "beam": pds.BEAM.values,
                    "wsum": np.asarray(child.attrs["wsum"]),
                }
            )
            parts.append(pds)
        ref_hess = HessianTree(hess_parts, nx, ny, 2 * nx, 2 * ny, eta=0.1, wsum=1.0)
        assert_allclose(out_pool[b], ref_hess.dot(x[b])[0], rtol=1e-12, atol=1e-12)
        ref_res, ref_bres = residual_from_partitions(
            band.ds.DIRTY.values, parts, model[b], cell_rad, bdirty=band.ds.BDIRTY.values
        )
        assert_allclose(res_pool[b], ref_res, rtol=1e-12, atol=1e-12)
        assert_allclose(bres_pool[b], ref_bres, rtol=1e-12, atol=1e-12)


@pytest.mark.timeout(120)
@pytest.mark.slow
def test_deconv_unequal_partition_counts(tmp_path):
    """Bands legitimately carry different partition counts (a fully flagged
    field chunk writes no scratch piece, so its band node has fewer part####
    children -- e.g. a mosaic field flagged out of one band only). The deconv
    driver and band workers must be partition-count-agnostic per band.
    """
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(7)
    nx = ny = 32
    nrow, nchan = 64, 1

    output_filename = str(tmp_path / "unequal")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, nx, ny, nrow, nchan, rng, parts_per_band=(1, 3))

    deconv_core(
        output_filename,
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=1.0,
        pd_maxit=20,
        cg_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
    )

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    nodes = sorted(n for n in dt.children if n.startswith("band"))
    assert len(nodes) == 2
    assert len(dt[nodes[0]].children) == 1 and len(dt[nodes[1]].children) == 3
    for n in nodes:
        ds = dt[n].ds
        assert "MODEL" in ds and "UPDATE" in ds
        assert "BRESIDUAL" in ds  # gradient residual written back (D23)
        assert ds.attrs["niters"] == 1
        assert np.isfinite(ds.MODEL.values).all()
        assert np.isfinite(ds.RESIDUAL.values).all()


@pytest.mark.timeout(120)
def test_deconv_requires_bdirty(tmp_path):
    """A .dt without BDIRTY (pre-D23 imager) is refused with a clear error."""
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(5)
    output_filename = str(tmp_path / "nobd")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, 16, 16, 32, 1, rng)

    import shutil

    import zarr

    root = zarr.open_group(dt_name, mode="a")
    for gname, _ in root.groups():
        shutil.rmtree(f"{dt_name}/{gname}/BDIRTY")
    zarr.consolidate_metadata(dt_name)

    with pytest.raises(ValueError, match="BDIRTY"):
        deconv_core(
            output_filename, product="I", nthreads=1, fits_mfs=False, fits_cubes=False, log_directory=str(tmp_path)
        )


@pytest.mark.slow
def test_deconv_driver_runs_with_the_frequency_prior(tmp_path):
    """End-to-end driver path with --gp-length-scale on (issue #307).

    The unit tests stop at presets; this is the only coverage of the whole
    chain -- core params -> geometry["freq_out"] -> _build_hess -> HessTreeRay
    -> the cube-level CG that replaces the band-parallel one -> the exact
    residual -> the hess_norm signature written back to the band attrs.
    """
    import xarray as xr

    from pfb_imaging.core.deconv import _m_signature
    from pfb_imaging.core.deconv import deconv as deconv_core

    rng = np.random.default_rng(43)
    nx = ny = 32
    output_filename = str(tmp_path / "synthgp")
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, nx, ny, 64, 1, rng)

    opts = dict(
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=None,  # exercise the power method on the coupled operator
        pd_maxit=20,
        cg_maxit=20,
        pm_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
        # the synthetic tree has a unit PSF, so at the default eta=1e-3 the data
        # term swamps the prior and it moves the update by only ~3e-4. Raise eta
        # so the test probes the regime the prior is for (eta a real share of M).
        eta=0.1,
    )
    deconv_core(output_filename, gp_length_scale=0.5, gp_cap=10.0, **opts)

    dt = xr.open_datatree(dt_name, engine="zarr", chunks=None)
    band = dt["band0000_time0000"].ds
    assert np.isfinite(band.MODEL.values).all()
    assert np.isfinite(band.UPDATE.values).all()
    assert band.attrs["niters"] == 1

    # the cached norm must be labelled with the prior settings, so a later run
    # without them re-estimates rather than reusing lambda_max of a different M
    sig = band.attrs["hess_norm_opts"]
    assert sig == _m_signature(dict(eta=0.1, eta_mode=None, eta_cap=100.0, gp_length_scale=0.5, gp_cap=10.0))
    assert sig != _m_signature(dict(eta=0.1, eta_mode=None, eta_cap=100.0, gp_length_scale=None, gp_cap=10.0))

    # the prior actually changed the update: same tree, same seed, prior off
    out2 = str(tmp_path / "synthref")
    _write_synthetic_dt(f"{out2}_I.dt", nx, ny, 64, 1, np.random.default_rng(43))
    deconv_core(out2, gp_length_scale=None, **opts)
    ref = xr.open_datatree(f"{out2}_I.dt", engine="zarr", chunks=None)["band0000_time0000"].ds
    rel = np.linalg.norm(band.UPDATE.values - ref.UPDATE.values) / np.linalg.norm(ref.UPDATE.values)
    assert rel > 1e-2, f"the prior left the update unchanged (rel={rel:.3e})"


# ---------------------------------------------------------------------------
# --eta-in-grad (issue #310): the prior enters the objective, not only M
# ---------------------------------------------------------------------------


def _eta_grad_opts(**over):
    """Driver opts for the --eta-in-grad tests.

    eta is 0.1, not the 1e-3 default: the synthetic tree has a unit PSF, so at
    the default the data term swamps the prior and the correction is lost in
    the CG tolerance (same reason as the frequency-prior driver test).
    """
    opts = dict(
        product="I",
        minor_cycle="sara",
        opt_backend="primal-dual",
        niter=1,
        hess_norm=1.0,
        pd_maxit=20,
        cg_maxit=20,
        bases=["self"],
        nlevels=1,
        l1_reweight_from=100,
        nthreads=2,
        nworkers=1,
        fits_mfs=False,
        fits_cubes=False,
        verbosity=0,
        eta=0.1,
    )
    opts.update(over)
    return opts


def _run_two_cycles(tmp_path, tag, eta_in_grad):
    """Two major cycles on a fresh synthetic tree; returns the final UPDATE."""
    import xarray as xr

    from pfb_imaging.core.deconv import deconv as deconv_core

    output_filename = str(tmp_path / tag)
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, 32, 32, 64, 1, np.random.default_rng(51))
    deconv_core(output_filename, eta_in_grad=eta_in_grad, **_eta_grad_opts(niter=2))
    ds = xr.open_datatree(dt_name, engine="zarr", chunks=None)["band0000_time0000"].ds
    return ds.UPDATE.values.copy()


@pytest.mark.slow
def test_eta_in_grad_changes_the_update_once_the_model_is_nonzero(tmp_path):
    """The prior term is -K^-1 m, so it bites from the second cycle on.

    Cycle 1 starts from a zero model and is identical either way; cycle 2 sees
    a gradient that differs by K^-1 m, so its update must differ too.
    """
    off = _run_two_cycles(tmp_path, "eig0", False)
    on = _run_two_cycles(tmp_path, "eig1", True)

    assert not np.allclose(off, on, rtol=1e-6, atol=1e-12)


@pytest.mark.slow
def test_eta_in_grad_is_a_no_op_on_the_first_step_from_a_zero_model(tmp_path):
    """K^-1 * 0 == 0: a fresh tree's first update cannot move, bit for bit.

    Also guards the scale. Applying the correction to the raw residual instead
    of the wsum-normalised one, or dropping the model factor, would perturb
    this even at a zero model.
    """
    import xarray as xr

    from pfb_imaging.core.deconv import deconv as deconv_core

    updates = {}
    for flag in (False, True):
        output_filename = str(tmp_path / f"eigz{int(flag)}")
        dt_name = f"{output_filename}_I.dt"
        _write_synthetic_dt(dt_name, 32, 32, 64, 1, np.random.default_rng(52))
        deconv_core(output_filename, eta_in_grad=flag, **_eta_grad_opts())
        ds = xr.open_datatree(dt_name, engine="zarr", chunks=None)["band0000_time0000"].ds
        updates[flag] = ds.UPDATE.values.copy()

    np.testing.assert_allclose(updates[False], updates[True], rtol=0, atol=0)


def test_grad_with_prior_subtracts_the_prior_term_on_the_normalised_scale():
    """The scale decision, isolated.

    ``residual``/``bresidual`` reaching the solver are already divided by the
    total wsum, and ``prior_dot`` is defined on that same normalised operator
    (--eta is a fraction of the total wsum), so the correction is subtracted
    with no further scaling. It must not touch the caller's arrays: the raw
    ones are what get stored, and storing an eta-inclusive gradient would
    double-count it on every resume.
    """
    from pfb_imaging.core.deconv import _grad_with_prior

    class _Prior:
        def __init__(self, eta):
            self.eta = eta

        def prior_dot(self, x):
            return self.eta * x

    rng = np.random.default_rng(54)
    nband, ny, nx = 2, 4, 4
    residual = rng.standard_normal((nband, ny, nx))
    bresidual = rng.standard_normal((nband, ny, nx))
    model = rng.standard_normal((nband, ny, nx))
    r0, b0 = residual.copy(), bresidual.copy()

    r, b = _grad_with_prior(residual, bresidual, model, _Prior(0.25))

    np.testing.assert_allclose(r, r0 - 0.25 * model, rtol=0, atol=1e-15)
    np.testing.assert_allclose(b, b0 - 0.25 * model, rtol=0, atol=1e-15)
    np.testing.assert_allclose(residual, r0, rtol=0, atol=0)
    np.testing.assert_allclose(bresidual, b0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# --mop (issue #311): the near-perfect-residual products
# ---------------------------------------------------------------------------


def _run_mop(tmp_path, tag, **over):
    """One major cycle on a fresh synthetic tree; returns the band 0 dataset."""
    import xarray as xr

    from pfb_imaging.core.deconv import deconv as deconv_core

    output_filename = str(tmp_path / tag)
    dt_name = f"{output_filename}_I.dt"
    _write_synthetic_dt(dt_name, 32, 32, 64, 1, np.random.default_rng(61))
    deconv_core(output_filename, **_eta_grad_opts(**over))
    return xr.open_datatree(dt_name, engine="zarr", chunks=None)["band0000_time0000"].ds


@pytest.mark.slow
def test_mop_writes_mopped_products_by_default(tmp_path):
    """--mop is on by default and lands both products in the band node."""
    ds = _run_mop(tmp_path, "mop_on")

    assert "MODEL_MOPPED" in ds
    assert "RESIDUAL_MOPPED" in ds
    assert ds.MODEL_MOPPED.dims == ("corr", "y", "x")
    assert np.isfinite(ds.MODEL_MOPPED.values).all()
    assert np.isfinite(ds.RESIDUAL_MOPPED.values).all()


@pytest.mark.slow
def test_no_mop_writes_neither_product(tmp_path):
    """It costs a forward solve and a gridding sweep, so it must be skippable."""
    ds = _run_mop(tmp_path, "mop_off", mop=False)

    assert "MODEL_MOPPED" not in ds
    assert "RESIDUAL_MOPPED" not in ds
    assert "MODEL" in ds  # the ordinary products are unaffected


@pytest.mark.slow
def test_mop_leaves_the_deconvolved_model_alone(tmp_path):
    """MODEL stays the regularised model; the mop is a separate product.

    MODEL_MOPPED is MODEL plus a least-squares update, so it is not sparse and
    not a component model -- it must not overwrite what deconv converged to.
    """
    ds = _run_mop(tmp_path, "mop_sep")

    assert not np.allclose(ds.MODEL.values, ds.MODEL_MOPPED.values, rtol=1e-6, atol=1e-12)


# Note: "mopping lowers the residual" is NOT tested on _write_synthetic_dt.
# That fixture's DIRTY and its partitions' VIS are independent random draws, so
# the operator and the right-hand side describe different data and M is nowhere
# near A -- M^-1 r is then a large least-squares answer to an inconsistent
# system and the exact residual against it is worse, not better (measured:
# 5.2e8 against 9.5e3). The claim needs consistent data, so it lives in
# test_deconv_groundtruth, which images a real MS with a known injected sky.


@pytest.mark.slow
def test_mop_preserves_the_run_attrs(tmp_path):
    """The mop write must not wipe niters/rms/hess_norm off the band node.

    to_zarr(mode="a") merges variables but REPLACES attrs wholesale, so a
    second write built from the band's original attrs silently drops
    everything the major cycle recorded -- and a resumed run would then restart
    from iteration 0 and re-estimate the norm.
    """
    ds = _run_mop(tmp_path, "mop_attrs")

    assert ds.attrs["niters"] == 1
    assert "hess_norm" in ds.attrs
    assert "hess_norm_opts" in ds.attrs
    assert "rms" in ds.attrs and "rmax" in ds.attrs
    assert ds.attrs["bandid"] == 0  # the original attrs survive too


def _seed_model(store, rng, nband=2, nx=32, ny=32):
    """Put a nonzero MODEL and its matching BRESIDUAL into a synthetic tree.

    Lets a run start from a model without spending a major cycle to build one,
    which is what makes a niter=0 comparison possible.
    """
    import xarray as xr

    for b in range(nband):
        n = f"band{b:04d}_time0000"
        ds = xr.open_datatree(store, engine="zarr", chunks=None)[n].ds
        model = rng.standard_normal((1, ny, nx))
        xr.Dataset(
            {
                "MODEL": (("corr", "y", "x"), model),
                "BRESIDUAL": (("corr", "y", "x"), ds.BDIRTY.values.copy()),
            },
            attrs=dict(ds.attrs),
        ).to_zarr(store, group=n, mode="a")


@pytest.mark.slow
def test_mop_uses_the_data_gradient_not_the_eta_corrected_one(tmp_path):
    """The mop must solve against r_data, whatever --eta-in-grad is doing.

    "Near perfect residual" means the *data* residual is near zero, so the mop
    direction is M^-1 r_data. Solving against the eta-corrected gradient
    r_data - K^-1 m instead makes the mop collapse to nothing exactly where it
    is wanted: at a regularised fixed point that gradient is ~0, so
    MODEL_MOPPED -> MODEL and the residual is not mopped at all.

    Pinned by holding the model fixed (niter=0, seeded MODEL) so the only thing
    the flag can change is the mop right-hand side. The two must agree exactly.
    """
    import xarray as xr

    from pfb_imaging.core.deconv import deconv as deconv_core

    mopped = {}
    for flag in (False, True):
        output_filename = str(tmp_path / f"moprhs{int(flag)}")
        dt_name = f"{output_filename}_I.dt"
        _write_synthetic_dt(dt_name, 32, 32, 64, 1, np.random.default_rng(71))
        _seed_model(dt_name, np.random.default_rng(72))
        deconv_core(output_filename, eta_in_grad=flag, **_eta_grad_opts(niter=0))
        ds = xr.open_datatree(dt_name, engine="zarr", chunks=None)["band0000_time0000"].ds
        mopped[flag] = ds.MODEL_MOPPED.values.copy()
        assert not np.allclose(ds.MODEL_MOPPED.values, ds.MODEL.values), "mop was a no-op"

    np.testing.assert_allclose(mopped[False], mopped[True], rtol=0, atol=0)
