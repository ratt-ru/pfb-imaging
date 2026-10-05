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

import functools  # noqa: E402
import importlib.util  # noqa: E402
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


def _have_daskms():
    return importlib.util.find_spec("daskms") is not None


@functools.lru_cache(maxsize=1)
def casacore_unusable_reason():
    """Why python-casacore cannot be used here, or None if it works.

    Importing `casacore.tables` is NOT enough to know casacore works. The
    python bindings are a thin layer over casacore's own `libcasa_python3`,
    whose NumPy ABI is fixed when *that* library was compiled -- so on a distro
    casacore built against NumPy 1.x the import succeeds and the first table
    open raises:

        RuntimeError: PycArray: failed to load the numpy API

    Ubuntu 24.04 is exactly that case (casacore 3.5.0), which is why the arm
    `--extra all` CI leg turned 25 tests red with the same message instead of
    skipping them. Probe a real table open once per session so those tests skip
    with one honest reason, and let
    `test_optional_extras.test_casacore_is_usable_when_installed` be the single
    place that reports the breakage. Details and a container reproducer:
    `scripts/casacore_issues/numpy2_abi_distro_casacore.sh` (#330).

    Returns:
        A reason string suitable for `pytest.skip`, or None when casacore works.
    """
    try:
        from casacore.tables import makescacoldesc, maketabdesc, table
    except ImportError as exc:
        return f"python-casacore is not installed ([casacore] extra): {exc}"

    import tempfile

    # a scratch table rather than the test MS: self-contained, and it exercises
    # the same converter layer that fails.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            desc = maketabdesc([makescacoldesc("X", 0)])
            with table(f"{tmp}/probe.tab", desc, nrow=1, ack=False) as tab:
                tab.putcol("X", np.zeros(1, dtype=np.int32))
    except Exception as exc:  # noqa: BLE001 - any failure here means unusable
        return f"python-casacore is installed but cannot open a table: {exc}"

    return None


def require_casacore():
    """Skip unless python-casacore is installed *and* actually works."""
    reason = casacore_unusable_reason()
    if reason:
        pytest.skip(reason)


def daskms_unusable_reason():
    """Why dask-ms cannot be used here, or None if it works.

    dask-ms reads and writes through python-casacore, so "dask-ms is
    importable" is not enough -- on a casacore whose NumPy ABI does not match
    (see `casacore_unusable_reason`) every dask-ms call fails at the first
    table open, exactly as a direct casacore call would.
    """
    if not _have_daskms():
        return "dask-ms is not installed ([casacore] extra)"
    return casacore_unusable_reason()


def require_daskms():
    """Skip unless dask-ms is installed and its casacore actually works."""
    reason = daskms_unusable_reason()
    if reason:
        pytest.skip(reason)


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    # Downloaded unconditionally: arcae reads this MS without dask-ms or
    # python-casacore, so the [casacore]-free legs need it too (they run the
    # imager and degrid tests against it).
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
    """Path to the shared MSv2 test set.

    This is just a path, and the MS is *downloaded*, not built -- `imager` and
    `degrid` read MSv2 tables through arcae, which vendors casacore. So this
    fixture deliberately does NOT require the [casacore] extra.

    It used to `importorskip("daskms")`, which cascaded a skip to every
    MSv2-backed test. That was convenient and wrong: it meant the aarch64
    `--extra full` leg -- the one we gate on -- skipped all 77 of them and
    never gridded a visibility, so "arm is green" said nothing about whether
    imaging worked there. Fixtures and helpers that genuinely need dask-ms or
    python-casacore now skip for themselves (`ms_meta`, `drop_column`,
    `simple_mds`, `make_multi_spw_ms`), which is narrower and honest.
    """
    return str(ms_path)


@pytest.fixture(scope="session")
def ms_meta(ms_name):
    """MS-derived state shared across MS-based integration tests.

    Reading the MS and extracting uvw/freq/times once per session avoids
    re-doing the same I/O and reductions in every test.

    Read with **arcae**, not dask-ms: this is plain column access, arcae ships
    aarch64 wheels and python-casacore does not, and every consumer of this
    fixture is a test we want running on the casacore-free legs. Exposing plain
    numpy rather than an xarray dataset also means `sky_truth` can write the MS
    back with arcae (#330).
    """
    import arcae

    with arcae.table(ms_name) as tab:
        time = np.asarray(tab.getcol("TIME"))
        uvw = np.asarray(tab.getcol("UVW"))
        ant1 = np.asarray(tab.getcol("ANTENNA1"))
        ant2 = np.asarray(tab.getcol("ANTENNA2"))
        # arcae indexes with a tuple of slices, not casacore's (start, nrow)
        ncorr = int(np.asarray(tab.getcol("DATA", (slice(0, 1),))).shape[-1])
    with arcae.table(f"{ms_name}::SPECTRAL_WINDOW") as spw:
        freq = np.asarray(spw.getcol("CHAN_FREQ")).squeeze()

    utime = np.unique(time)

    return SimpleNamespace(
        utime=utime,
        freq=freq,
        freq0=float(np.mean(freq)),
        ntime=utime.size,
        nchan=freq.size,
        nant=int(np.maximum(ant1.max(), ant2.max()) + 1),
        ncorr=ncorr,
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
    # Define the environment once
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    os.environ["RAY_PROCESS_SPAWN"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

    env_vars = set_envs(2, 1)
    env_vars["JAX_LOGGING_LEVEL"] = "ERROR"
    env_vars["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    env_vars["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    env_vars["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

    runtime_env = {
        "env_vars": env_vars,
        "worker_process_setup_hook": setup_ray_worker,
    }

    # num_cpus=2, not 1: the session-scoped `band_pool` actors each hold a nominal
    # 1e-2 CPU claim for the whole session, which would leave a num_cpus=1 cluster
    # unable to ever schedule a default 1-CPU task (pfb hci hung on exactly this).
    ray.init(num_cpus=2, runtime_env=runtime_env, ignore_reinit_error=True, include_dashboard=False)

    yield

    # Shutdown after all tests in the session are done
    ray.shutdown()


@pytest.fixture(scope="session")
def band_pool(manage_ray):
    """Session-scoped BandWorkerPool cache, keyed by (nband, nthreads).

    Ray actor startup, not arithmetic, is what makes the pool-backed tests
    expensive: test_hess_tree_ray.py alone paid ~45 s of it on 8x8 and 16x16
    arrays (29% of the fast loop) because every test built its own pool.
    `HessTreeRay(..., workers=pool)` and `make_sara(..., workers=pool)` are
    existing injection points, and `BandWorkerPool.init_hess` rebuilds every
    per-band HessianTree from scratch, so reuse is numerically invisible --
    pinned by tests/test_band_pool.py.

    The guarantee covers the Hessian role ONLY. `init_hess` rebuilds `_hess` but
    does not refresh `_psi` or `_hess_parts`, so a pooled test must not rely on
    `init_hess(None, ...)` or on `load_bands` state: both would silently see
    whatever the previous user of the pool left behind.

    Facades sharing a pool share its worker state: the workers hold a single
    operator, so the last `init_hess` wins. `HessTreeRay`s that coexist on one
    pool must therefore pass identical `init_hess` arguments (partitions, nx, ny,
    nx_psf, ny_psf, eta, wsum, eta_mode, eta_cap) or be used strictly
    sequentially. `freq_prec` is driver-side (`_dC`) and so may differ freely.

    `nband == 1` pools are cached too, but they cost nothing: that branch runs
    in-process and never imports ray.

    Depends on `manage_ray` explicitly, not just by autouse, so finalisation
    order is guaranteed: this fixture's teardown (ray.kill) must run BEFORE
    manage_ray's ray.shutdown(), and pytest finalises in reverse setup order.
    """
    # deferred: optional heavy runtime (ray) -- mirrors band_worker's own import
    from pfb_imaging.operators.band_worker import BandWorkerPool

    pools = {}

    def get(nband, nthreads=1):
        # a core driver that ran without keep_ray_alive=True tore the cluster
        # down and killed these actors; rebuild rather than hand back corpses
        if not ray.is_initialized():
            pools.clear()
        key = (nband, nthreads)
        if key not in pools:
            pools[key] = BandWorkerPool(nband, nthreads)
        return pools[key]

    yield get

    for pool in pools.values():
        pool.shutdown()


def copy_tree(src_base, dest_base):
    """Copy a `<base>_I.dt` (and `.scratch`, if present) to a new base prefix.

    Session-scoped pipeline products are read by several tests and written in
    place by deconv and restore, so a writer takes a copy. Copying a small
    test `.dt` costs ~0.2 s against a ~56 s imager+deconv rebuild.

    Args:
        src_base: Output prefix of the tree to copy.
        dest_base: Output prefix to copy it to.

    Returns:
        `dest_base` as a string.
    """
    for suffix in ("_I.dt", "_I.scratch"):
        src = Path(str(src_base) + suffix)
        if src.exists():
            shutil.copytree(src, Path(str(dest_base) + suffix))
    return str(dest_base)


# Ground-truth pipeline settings, shared so the imager/deconv/restore ground-truth
# tests provably image the SAME data. Three tests previously inlined these and
# happened to match byte for byte -- nothing enforced it.
GT_IMAGER_KW = dict(
    channels_per_image=-1,
    integrations_per_image=-1,
    product="I",
    robustness=0.0,
    fits_mfs=False,
    fits_cubes=False,
    overwrite=True,
    keep_ray_alive=True,
)

GT_DECONV_KW = dict(
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
    verbosity=0,
)


@pytest.fixture(scope="session")
def gt_dt(ms_name, sky_truth, tmp_path_factory):
    """One imager run on the injected ground-truth sky, shared session-wide.

    Three tests issued byte-identical imager_core calls on this sky:
    test_deconv_groundtruth, test_restore_groundtruth, and
    test_preconditioner_consistency's unregularised-descent test.

    Returns the output BASE prefix, so the tree is f"{gt_dt}_I.dt". Writers
    must take a copy via `copy_tree` -- deconv and restore both write in place.
    """
    # deferred: collection cost -- keeps ducc0/africanus out of every pytest collection
    from pfb_imaging.core.imager import imager as imager_core

    base = str(tmp_path_factory.mktemp("gt_imaged") / "gtimg")
    imager_core(
        [Path(ms_name)],
        base,
        nx=sky_truth.nx,
        ny=sky_truth.ny,
        cell_size=sky_truth.cell_size,
        **GT_IMAGER_KW,
    )
    return base


@pytest.fixture(scope="session")
def gt_deconv_dt(gt_dt, tmp_path_factory):
    """`gt_dt` deconvolved at the shared ground-truth settings.

    test_deconv_groundtruth and test_restore_groundtruth each ran this exact
    imager+deconv pair -- ~56 s of the 616 s slow suite apiece. Shared here,
    with `fits_per_partition=True` because test_deconv_groundtruth asserts on
    the per-partition residual FITS (they land in `<dirname>/fits/`, the
    default FITS folder, and do not touch the tree). test_deconv_groundtruth
    only reads this tree; restore takes a `copy_tree` copy because it writes.
    """
    # deferred: collection cost -- keeps ducc0/africanus out of every pytest collection
    from pfb_imaging.core.deconv import deconv as deconv_core

    base = copy_tree(gt_dt, tmp_path_factory.mktemp("gt_deconvolved") / "gtdec")
    deconv_core(base, fits_per_partition=True, **GT_DECONV_KW)
    return base


@pytest.fixture(scope="session")
def sky_truth(ms_name, ms_meta, image_geometry):
    """Deterministic point-source sky + flag pattern injected into the test MS.

    Writes DATA (predicted vis, Stokes I into XX/YY), FLAG (~10% random
    samples plus one fully flagged channel) and FLAG_ROW into the shared MS.
    Session-scoped: the injection is seeded and idempotent (same DATA/FLAG
    every time) and this is the only writer to the shared MS -- every
    test_degrid writer and both drop_column callers target a function-scoped
    copy. Session scope is required, not merely cheaper: the gt_dt pipeline
    fixtures are session-scoped and pytest refuses to let them reach a
    module-scoped fixture (ScopeMismatch). It buys no time on its own, since
    the 13.19 s first use is JIT warm-up that reattaches elsewhere.

    The truth WCS is built with plain astropy (not pfb's set_wcs) so the
    coordinate truth is independent of the code under test.
    """
    import arcae
    from astropy.wcs import WCS
    from ducc0.wgridder import dirty2vis

    from pfb_imaging.operators.gridder import wgridder_conventions

    rng = np.random.default_rng(1234)
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

    with arcae.table(f"{ms_name}::FIELD") as field:
        radec = np.asarray(field.getcol("PHASE_DIR")).squeeze()  # (ra, dec) rad
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

    with arcae.table(ms_name, readonly=False) as tab:
        dtype = np.asarray(tab.getcol("DATA", (slice(0, 1),))).dtype
        tab.putcol("DATA", model_vis.astype(dtype))
        tab.putcol("FLAG", flag)
        tab.putcol("FLAG_ROW", flag_row)

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


# Modules that import dask-ms at module scope. pytest reports a collection-time
# ImportError as an error rather than a skip, so these have to be excluded
# before collection when dask-ms cannot be used.
#
# "cannot be used" covers more than "not installed": dask-ms works through
# python-casacore, so a casacore that imports but cannot open a table (the
# distro NumPy-1 ABI case, #330) makes every one of these fail at runtime with
# the same message. Excluding them there keeps the failure reported once, by
# `test_optional_extras.test_casacore_is_usable_when_installed`, instead of
# once per test.
collect_ignore = [] if daskms_unusable_reason() is None else ["test_hci.py", "test_imager_pol.py"]


@pytest.fixture
def writable_ms(ms_name, tmp_path):
    """A private copy of the shared test MS, for tests that write to it.

    The session MS is shared and `sky_truth` injects its DATA/FLAG only once per
    session, so a test that overwrites DATA, FLAG or FLAG_ROW (including via
    dask-ms `xds_to_table`) would corrupt every later test. Work on a copy.
    """
    dest = tmp_path / "writable.ms"
    shutil.copytree(ms_name, dest)
    return str(dest)


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


@pytest.fixture
def needs_rephasing():
    """Skip unless rephasing can run.

    `--phase-dir`, any multi-field selection and `--target` all go through
    `utils/astrometry.synthesize_uvw` / `get_coordinates`, which wrap pyrap
    measures -- the one part of the *imaging* path that still needs
    python-casacore (#330). Everything else in `imager` reaches the MS through
    arcae.
    """
    pytest.importorskip("pyrap.measures", reason="rephasing needs the [casacore] extra")
    # pyrap goes through the same libcasa_python3 converters as casacore.tables
    require_casacore()


@pytest.fixture
def pctable():
    """python-casacore's ``table``, or skip the test.

    Several tests need casacore's *write* API (adding rows, putcell on
    subtables, removecols) which arcae does not expose. They are the only
    reason those tests cannot run on a [full]-only install, so they declare it
    here rather than each repeating the check.
    """
    require_casacore()

    from casacore.tables import table

    return table


def drop_column(ms_path, column):
    """Remove a column with python-casacore (arcae has no removecols).

    Skips rather than errors without the [casacore] extra -- raising Skipped
    from a helper works the same inside a fixture or a test body.
    """
    require_casacore()

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
    than invented: `degrid` refuses to degrid a model whose tangent point
    differs from the field's (wiki D21, mosaics are not supported in v1), so a
    made-up radec makes every driver test fail the guard rather than exercise
    the code under test.
    """
    import arcae
    from pfb_model_spec.utils.io import build_mds_dataset
    from pfb_model_spec.utils.modelspec import fit_image_cube

    # arcae, not python-casacore: this only reads one subtable cell, and arcae
    # ships aarch64 wheels while python-casacore does not. Keeping it casacore-
    # free is what lets the degrid tests run on the arm gating leg.
    with arcae.table(f"{ms_name}::FIELD") as tab:
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

    require_casacore()

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
