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


@pytest.fixture(scope="function")
def sky_truth(ms_name, ms_meta, image_geometry):
    """Deterministic point-source sky + flag pattern injected into the test MS.

    Writes DATA (predicted vis, Stokes I into XX/YY), FLAG (~10% random
    samples plus one fully flagged channel) and FLAG_ROW into the shared MS.
    Function-scoped and seeded (idempotent): legacy tests overwrite DATA
    mid-session until they are retired, so every consumer re-injects.

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
