import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import requests

from pfb_imaging import set_envs, setup_ray_worker

# ── Clear stale Numba caches ────────────────────────────────────────
# Numba's file cache (`cache=True`) is keyed per-function by source hash.
# Functions decorated with `inline="always"` get compiled into their callers,
# but Numba does NOT track this cross-function dependency.  If an inlined
# function changes while the caller's source stays the same, the caller's
# cached machine code is stale and loading it can segfault.
#
# Clearing __pycache__ dirs on every session start is cheap (~ms) and
# eliminates the problem entirely during development.
_src_root = Path(__file__).resolve().parent.parent / "src" / "pfb_imaging"
for _cache_dir in _src_root.rglob("__pycache__"):
    shutil.rmtree(_cache_dir, ignore_errors=True)

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

    runtime_env = {
        "env_vars": env_vars,
        "working_dir": None,
        "excludes": get_excludes(),
        "worker_process_setup_hook": setup_ray_worker,
    }

    # Start Ray
    ray.init(num_cpus=1, runtime_env=runtime_env, ignore_reinit_error=True, include_dashboard=False)

    yield

    # Shutdown after all tests in the session are done
    ray.shutdown()
