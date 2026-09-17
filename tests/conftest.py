import os

# Threading and memory caps for the test session. These MUST be set before
# any scientific stack import: numpy/scipy resolve OpenBLAS/MKL/OMP thread
# counts at load time, numexpr at init, jax at first use. setdefault lets
# CI or a developer override individual values without editing this file.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
os.environ.setdefault("RAY_NUM_CPUS", "2")

# Disable ray's uv_runtime_env_hook BEFORE importing ray. When the driver runs
# under `uv run`, the hook overrides py_executable so workers are launched via
# `uv run --frozen python …`, which rebuilds a fresh venv per worker from the
# driver's cmdline. That venv only contains the default project deps (not the
# [full] extra where ray itself lives), so workers crash on `import ray` and
# ray.wait() blocks forever. The constant is read at import time, so this must
# come before `import ray`.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import shutil  # noqa: E402
import tarfile  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import ray  # noqa: E402
import requests  # noqa: E402

# ── Numba cache location ────────────────────────────────────────────
# Pin Numba's file cache to <repo>/.numba_cache so it survives across
# pytest sessions (by default Numba writes .nbi/.nbc into __pycache__
# alongside the .py files, which mixes build artefacts with source state).
#
# Numba's cache is keyed per-function by source hash, but functions
# decorated with `inline="always"` get compiled into their callers and
# the cross-function dependency is NOT tracked.  If an inlined function
# changes while the caller's source stays the same, the caller's cached
# machine code is stale and loading it can segfault.  Set
# PFB_FRESH_NUMBA_CACHE=1 to force a clean rebuild when iterating on
# inline-decorated functions.
_repo_root = Path(__file__).resolve().parent.parent
_numba_cache = _repo_root / ".numba_cache"
_numba_cache.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))

if os.environ.get("PFB_FRESH_NUMBA_CACHE"):
    shutil.rmtree(_numba_cache, ignore_errors=True)
    _numba_cache.mkdir(exist_ok=True)

from pfb_imaging import set_envs, setup_ray_worker  # noqa: E402

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")
test_data_path.mkdir(parents=True, exist_ok=True)

_data_tar_name = "test_ascii_1h60.0s.MS.tar.gz"
_ms_name = "test_ascii_1h60.0s.MS"

data_tar_path = Path(test_data_path, _data_tar_name)
ms_path = Path(test_data_path, _ms_name)

# https://drive.google.com/file/d/1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT/view?usp=sharing

gdrive_id = "1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT"

url = "https://drive.google.com/uc?id={id}".format(id=gdrive_id)


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        download = requests.get(url)  # , params={"dl": 1}
        with open(data_tar_path, "wb") as f:
            f.write(download.content)
        with tarfile.open(data_tar_path, "r:gz") as tar:
            tar.extractall(path=test_data_path)
        data_tar_path.unlink()
        print("Test data successfully downloaded.")


@pytest.fixture(scope="session")
def ms_name():
    return str(ms_path)


@pytest.fixture(scope="session")
def ms_meta(ms_name):
    """MS-derived state shared across MS-based integration tests.

    Reading the MS and extracting uvw/freq/times once per session avoids
    re-doing the same I/O and reductions in every test.
    """
    from daskms import xds_from_ms, xds_from_table

    xds = xds_from_ms(ms_name, chunks={"row": -1, "chan": -1})[0]
    spw = xds_from_table(f"{ms_name}::SPECTRAL_WINDOW")[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    uvw = xds.UVW.values
    ant1 = xds.ANTENNA1.values
    ant2 = xds.ANTENNA2.values
    time = xds.TIME.values

    return SimpleNamespace(
        xds=xds,
        spw=spw,
        utime=utime,
        freq=freq,
        freq0=float(np.mean(freq)),
        ntime=utime.size,
        nchan=freq.size,
        nant=int(np.maximum(ant1.max(), ant2.max()) + 1),
        ncorr=xds.corr.size,
        uvw=uvw,
        nrow=uvw.shape[0],
        max_blength=float(np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2).max()),
        max_freq=float(freq.max()),
        time=time,
        ant1=ant1,
        ant2=ant2,
    )


@pytest.fixture(scope="session")
def image_geometry(ms_meta):
    """Standard image geometry (fov=1.0, srf=2.0) used by kclean/sara/polproducts/model2comps."""
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size

    cell_n = 1.0 / (2 * ms_meta.max_blength * ms_meta.max_freq / lightspeed)
    srf = 2.0
    cell_rad = cell_n / srf
    cell_deg = cell_rad * 180 / np.pi

    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    return SimpleNamespace(
        fov=fov,
        srf=srf,
        cell_rad=cell_rad,
        cell_deg=cell_deg,
        cell_size=cell_deg * 3600,
        nx=npix,
        ny=npix,
    )


@pytest.fixture(scope="session")
def gain_cholesky(ms_meta):
    """Cholesky factors of the gain covariance used for corrupted-vis simulation."""
    from africanus.gps.utils import abs_diff

    t = (ms_meta.utime - ms_meta.utime.min()) / (ms_meta.utime.max() - ms_meta.utime.min())
    nu = 2.5 * (ms_meta.freq / ms_meta.freq0 - 1.0)

    tt = abs_diff(t, t)
    cov_t = 0.1 * np.exp(-(tt**2) / (2 * 0.25**2))
    chol_t = np.linalg.cholesky(cov_t + 1e-10 * np.eye(ms_meta.ntime))

    vv = abs_diff(nu, nu)
    cov_nu = 0.1 * np.exp(-(vv**2) / (2 * 0.1**2))
    chol_nu = np.linalg.cholesky(cov_nu + 1e-10 * np.eye(ms_meta.nchan))

    return SimpleNamespace(chol_t=chol_t, chol_nu=chol_nu, nu=nu)


@pytest.fixture(scope="session")
def time_chunks(ms_meta):
    """Row-to-time-bin mapping used when applying gain corruptions via corrupt_vis."""
    from pfb_imaging.utils.misc import chunkify_rows

    row_chunks, tbin_idx, tbin_counts = chunkify_rows(ms_meta.time, ms_meta.ntime)
    return SimpleNamespace(row_chunks=row_chunks, tbin_idx=tbin_idx, tbin_counts=tbin_counts)


@pytest.fixture(scope="session", autouse=True)
def manage_ray():
    def get_excludes():
        if os.path.exists(".rayignore"):
            return [line.strip() for line in open(".rayignore") if line.strip() and not line.startswith("#")]

    # Define the environment once
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    os.environ["RAY_PROCESS_SPAWN"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

    env_vars = set_envs(2, 1)
    env_vars["JAX_LOGGING_LEVEL"] = "ERROR"
    env_vars["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    env_vars["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    env_vars["RAY_RUNTIME_ENV_WORKING_DIR_MAX_SIZE_MB"] = "2048"
    env_vars["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

    runtime_env = {
        "env_vars": env_vars,
        "excludes": get_excludes(),
        "worker_process_setup_hook": setup_ray_worker,
    }

    ray.init(num_cpus=1, runtime_env=runtime_env, ignore_reinit_error=True, include_dashboard=False)

    yield

    # Shutdown after all tests in the session are done
    ray.shutdown()


@pytest.fixture(scope="module")
def sky_truth(ms_name, ms_meta, image_geometry):
    """Deterministic point-source sky + flag pattern injected into the test MS.

    Writes DATA (predicted vis, Stokes I into XX/YY), FLAG (~10% random
    samples plus one fully flagged channel) and FLAG_ROW into the shared MS.
    Module-scoped: the injection is seeded and idempotent (same DATA/FLAG
    every time), so re-injecting once per consuming module is cheap and safe
    now that the mid-session DATA-overwriting legacy tests are gone.

    The truth WCS is built with plain astropy (not pfb's set_wcs) so the
    coordinate truth is independent of the code under test.
    """
    import dask
    import dask.array as da
    from astropy.wcs import WCS
    from daskms import xds_from_table, xds_to_table
    from ducc0.wgridder import dirty2vis

    from pfb_imaging.operators.gridder import wgridder_conventions

    rng = np.random.default_rng(1234)
    xds = ms_meta.xds
    freq = ms_meta.freq
    freq0 = ms_meta.freq0
    nchan = ms_meta.nchan
    ncorr = ms_meta.ncorr
    uvw = ms_meta.uvw
    nrow = ms_meta.nrow

    nx = ny = 256
    cell_rad = image_geometry.cell_rad
    cell_deg = image_geometry.cell_deg
    cell_size = image_geometry.cell_size  # arcsec

    field = xds_from_table(f"{ms_name}::FIELD")[0]
    radec = field.PHASE_DIR.values.squeeze()  # (ra, dec) rad
    assert radec.shape == (2,), f"unexpected PHASE_DIR shape {radec.shape}"

    # sources at exact pixel centres: (lpix, mpix) = pixels east / north of
    # centre. Asymmetric on purpose (any transpose/flip moves at least one).
    lpix = np.array([3, 0, 40])
    mpix = np.array([-2, 55, 12])
    ref_flux = np.array([1.0, 2.5, 1.7])
    alpha = np.array([-0.7, -0.4, -1.0])

    l_s = lpix * cell_rad
    m_s = mpix * cell_rad
    nvals = np.sqrt(1.0 - l_s**2 - m_s**2)

    # x-major model raster per the pinned wgridder convention
    # (test_beam_orientation.py): source (l, m) -> raster [nx//2 - lpix, ny//2 + mpix]
    epsilon = 1e-7
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(0.0, 0.0)
    model_vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    for c in range(nchan):
        model = np.zeros((nx, ny))
        for s in range(lpix.size):
            model[nx // 2 - lpix[s], ny // 2 + mpix[s]] = ref_flux[s] * (freq[c] / freq0) ** alpha[s]
        model_vis[:, c : c + 1, 0] = dirty2vis(
            uvw=uvw,
            freq=freq[c : c + 1],
            dirty=model,
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            do_wgridding=True,
            nthreads=2,
        )
        model_vis[:, c, -1] = model_vis[:, c, 0]

    # deterministic flags: ~10% of (row, chan) samples plus one fully
    # flagged channel; FLAG_ROW consistent with fully flagged rows
    flagged_chan = 3
    flag_rc = rng.random((nrow, nchan)) < 0.1
    flag_rc[:, flagged_chan] = True
    flag = np.broadcast_to(flag_rc[:, :, None], (nrow, nchan, ncorr)).copy()
    flag_row = flag.all(axis=(1, 2))

    xds_w = xds.assign(
        DATA=(("row", "chan", "corr"), da.from_array(model_vis, chunks=(-1, -1, -1))),
        FLAG=(("row", "chan", "corr"), da.from_array(flag, chunks=(-1, -1, -1))),
        FLAG_ROW=(("row",), da.from_array(flag_row, chunks=-1)),
    )
    dask.compute(xds_to_table(xds_w, ms_name, columns=["DATA", "FLAG", "FLAG_ROW"]))

    # truth WCS (plain astropy; 0-based pixel (ix, iy) with crpix 1-based)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cdelt = [-cell_deg, cell_deg]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [np.rad2deg(radec[0]), np.rad2deg(radec[1])]
    w.wcs.crpix = [1 + nx // 2, 1 + ny // 2]
    sky_coords = [w.pixel_to_world(nx // 2 - lpix[s], ny // 2 + mpix[s]) for s in range(lpix.size)]

    return SimpleNamespace(
        nx=nx,
        ny=ny,
        cell_rad=cell_rad,
        cell_size=cell_size,
        radec=radec,
        lpix=lpix,
        mpix=mpix,
        ref_flux=ref_flux,
        alpha=alpha,
        nvals=nvals,
        flag=flag,
        flagged_chan=flagged_chan,
        wcs=w,
        sky_coords=sky_coords,
    )


@pytest.fixture
def degrid_ms(ms_name, tmp_path):
    """A private copy of the shared test MS, safe to add columns to and write.

    The session MS is read by many other modules; adding columns to it would
    leak across tests, and writing to it would corrupt the `sky_truth` DATA.
    """
    import shutil

    dest = tmp_path / "degrid.ms"
    shutil.copytree(ms_name, dest)
    return str(dest)


def drop_column(ms_path, column):
    """Remove a column with python-casacore (arcae has no removecols)."""
    from casacore.tables import table as pctable

    with pctable(ms_path, readonly=False, ack=False) as tab:
        if column in tab.colnames():
            tab.removecols(column)


@pytest.fixture
def simple_mds(ms_name, tmp_path):
    """A minimal, deliberately non-square `.mds` with a spectral slope.

    64 x 32 in (nx, ny): a wrongly oriented mask or image is invisible on a
    square grid, so nothing here is square. One component at (x=40, y=9),
    1.0 Jy at 1.0 GHz and 2.0 Jy at 1.1 GHz, so a render at 1.05 GHz must give
    exactly 1.5 -- which is the frequency-upsampling assertion.

    The tangent point is read from the test MS's own FIELD.PHASE_DIR rather
    than invented: `degrid-msv4` refuses to degrid a model whose tangent point
    differs from the field's (wiki D21, mosaics are not supported in v1), so a
    made-up radec makes every driver test fail the guard rather than exercise
    the code under test.
    """
    from casacore.tables import table as pctable
    from pfb_model_spec.utils.io import build_mds_dataset
    from pfb_model_spec.utils.modelspec import fit_image_cube

    with pctable(f"{ms_name}::FIELD", ack=False) as tab:
        radec = np.asarray(tab.getcol("PHASE_DIR")).squeeze()
    assert radec.shape == (2,), f"unexpected PHASE_DIR shape {radec.shape}"

    nx, ny = 64, 32
    cell_rad = 1.0e-5
    image = np.zeros((1, 2, nx, ny))
    image[0, 0, 40, 9] = 1.0
    image[0, 1, 40, 9] = 2.0
    times = np.array([1.62393461e9])  # unix seconds, as the .dt uses
    freqs = np.array([1.0e9, 1.1e9])

    coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
        times, freqs, image, wgt=np.ones((1, 2)), method="Legendre"
    )
    ds = build_mds_dataset(
        coeffs,
        x_index,
        y_index,
        expr,
        params,
        texpr,
        fexpr,
        times,
        freqs,
        cell_rad,
        nx,
        ny,
        0.0,
        0.0,  # center_x, center_y
        False,
        True,
        False,  # flip_u, flip_v, flip_w
        (float(radec[0]), float(radec[1])),  # tangent point, from the MS
        "I",
        "test",
    )
    path = tmp_path / "simple.mds"
    ds.to_zarr(str(path), mode="w")
    return str(path), ds


def make_multi_spw_ms(src, dest, nchan2, freq_offset=2.0e8, name2="spw-upper"):
    """Copy an MS and graft on a second spectral window.

    Real multi-SPW data is not in `tests/data`, but the selection and
    column-creation paths both branch on having more than one, so the tests
    build one. The new SPW is a distinct DDID with its own NAME, so
    `--spw-names` can address it.

    Args:
        src: Measurement set to copy.
        dest: Destination path.
        nchan2: Channels in the second SPW. Must equal the first: the MAIN
            data columns copied here are fixed-shape, so a different count
            leaves the table internally inconsistent and xarray-ms refuses to
            open it. A genuinely ragged MS needs variably-shaped DATA columns
            and has to be built from scratch, which is why the heterogeneous
            branch of `ensure_model_columns` is tested against a synthetic
            DataTree instead.
        freq_offset: Hz to shift the second SPW's band by.
        name2: NAME for the second SPW.

    Returns:
        `dest` as a string.
    """
    import shutil

    from casacore.tables import table as pctable

    shutil.copytree(src, dest)
    dest = str(dest)

    with pctable(f"{dest}::SPECTRAL_WINDOW", readonly=False, ack=False) as spw:
        s0 = spw.nrows()
        chan_freq = np.asarray(spw.getcell("CHAN_FREQ", 0))
        chan_width = np.asarray(spw.getcell("CHAN_WIDTH", 0))
        spw.addrows(1)
        for col in spw.colnames():
            try:
                spw.putcell(col, s0, spw.getcell(col, 0))
            except Exception:  # noqa: S110 - unset optional columns
                pass
        step = float(chan_width.ravel()[0])
        spw.putcell("CHAN_FREQ", s0, chan_freq[0] + freq_offset + step * np.arange(nchan2))
        spw.putcell("CHAN_WIDTH", s0, np.full(nchan2, step))
        spw.putcell("EFFECTIVE_BW", s0, np.full(nchan2, step))
        spw.putcell("RESOLUTION", s0, np.full(nchan2, step))
        spw.putcell("NUM_CHAN", s0, nchan2)
        spw.putcell("NAME", s0, name2)

    with pctable(f"{dest}::DATA_DESCRIPTION", readonly=False, ack=False) as dd:
        d0 = dd.nrows()
        dd.addrows(1)
        for col in dd.colnames():
            dd.putcell(col, d0, dd.getcell(col, 0))
        dd.putcell("SPECTRAL_WINDOW_ID", d0, s0)

    with pctable(dest, readonly=False, ack=False) as tab:
        nrow = tab.nrows()
        ncorr = np.asarray(tab.getcell("DATA", 0)).shape[-1]
        tab.addrows(nrow)
        for col in tab.colnames():
            try:
                tab.putcol(col, tab.getcol(col, 0, nrow), nrow, nrow)
            except Exception:  # noqa: S110 - FLAG_CATEGORY and friends are unset
                pass
        tab.putcol("DATA_DESC_ID", np.full(nrow, d0, np.int32), nrow, nrow)
        # the new rows must carry the second SPW's channel count
        for col in ("DATA", "MODEL_DATA", "CORRECTED_DATA", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"):
            if col not in tab.colnames():
                continue
            try:
                sample = np.asarray(tab.getcell(col, 0))
            except Exception:
                continue
            fill = np.zeros((nrow, nchan2, ncorr), sample.dtype)
            try:
                tab.putcol(col, fill, nrow, nrow)
            except Exception:  # noqa: S110 - fixed-shape columns cannot be reshaped
                pass
    return dest


@pytest.fixture
def multi_spw_ms(ms_name, tmp_path):
    """Two spectral windows of equal width: the normal multi-SPW case."""
    return make_multi_spw_ms(ms_name, tmp_path / "mspw.ms", nchan2=8)
