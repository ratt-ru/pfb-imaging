import ctypes
import importlib
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version

from pfb_imaging.utils import logging as pfb_logging

__version__ = "0.0.11"
pfb_version = version("pfb-imaging")
# x86_64 aliases across the platforms we care about. platform.machine() reports
# "x86_64" on Linux/macOS and "AMD64" on Windows.
_X86_MACHINES = ("x86_64", "AMD64", "amd64")


def _tbb_available():
    """True when the `tbb` distribution is installed in this environment."""
    try:
        importlib.metadata.distribution("tbb")
    except PackageNotFoundError:
        return False
    return True


def _default_threading_layer():
    """Numba threading layer to request for this environment.

    Numba treats an explicit NUMBA_THREADING_LAYER as a demand, not a
    preference: if the named layer cannot be loaded it raises

        ValueError: No threading layer could be loaded.

    at the first parallel dispatch, in whichever process hit it -- including
    Ray workers, where it surfaces as a dead actor. So the value here has to
    reflect what is actually installed, not what we would like.

    Intel TBB publishes manylinux_2_28_x86_64 and win_amd64 wheels only, and no
    sdist, for every version ever released. On aarch64 (DGX Spark / Grace,
    Graviton, arm64 Linux generally) it therefore cannot be installed at all,
    and since it now lives behind the optional [x86] extra it can be absent on
    x86_64 too.

    "default" is numba's own fallback chain -- tbb, then omp, then workqueue,
    first one that loads -- and never raises. Set PFB_NUMBA_THREADING_LAYER to
    pin a specific layer (benchmarking, or forcing omp where workqueue would
    silently cost you all your threads).
    """
    override = os.environ.get("PFB_NUMBA_THREADING_LAYER")
    if override:
        return override
    if platform.machine() in _X86_MACHINES and _tbb_available():
        return "tbb"
    return "default"


def _load_tbb(log=None):
    """ctypes-load the libtbb.so bundled in the `tbb` wheel, if there is one.

    numba needs libtbb.so resolvable before it initialises its threading layer,
    and the wheel does not put it anywhere the loader looks by default.

    Returns the loaded path, or None when TBB is not in use on this platform or
    is not installed. Never raises: absence of TBB is the expected state off
    x86_64, and callers that genuinely require it check the return value.
    """
    if _default_threading_layer() != "tbb":
        return None
    try:
        dist = importlib.metadata.distribution("tbb")
    except PackageNotFoundError:
        # Only reachable when PFB_NUMBA_THREADING_LAYER=tbb was forced.
        msg = "TBB was requested but the tbb package is not installed. Install pfb-imaging[x86]."
        (log.warning if log else logging.warning)(msg)
        return None
    for f in dist.files or ():
        if str(f).endswith("/libtbb.so"):
            tbb_path = str(dist.locate_file(f).resolve())
            ctypes.CDLL(tbb_path)
            return tbb_path
    return None


# This need to happen before importing numba
os.environ["NUMBA_THREADING_LAYER"] = _default_threading_layer()
# Also before importing numba: default the numba and meerkat-beams cache dirs
# to per-user directories so users on a shared host don't collide on ownership
# of a shared cache (#270). Hard-coded /tmp rather than gettempdir(): the
# defaults must match the static /tmp mount hints in the cab definitions, and
# apptainer leaks the host TMPDIR into the container where a TMPDIR-derived
# path may not be mounted. setdefault so an explicit env var (native env,
# stimela backend env) always wins.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    f"/tmp/numba-cache-{os.getuid()}",
)
os.environ.setdefault(
    "MBEAMS_CACHE_DIR",
    f"/tmp/mbeams-cache-{os.getuid()}",
)


def set_envs(nthreads, ncpu, log=None):
    # these seem to have more sensible defaults
    # Note - do not set NUMBA_NUM_THREADS here.
    # It should be initialised to the maximum and then we use numba.set_num_threads()
    # in the worker processes to set it to the number of threads per worker.
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NPY_NUM_THREADS"] = str(nthreads)
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["JAX_LOGGING_LEVEL"] = "INFO"  # for th emain process
    ne_threads = min(ncpu, nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)
    os.environ["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    threading_layer = _default_threading_layer()
    os.environ["NUMBA_THREADING_LAYER"] = threading_layer
    # gRPC EventEngine pool defaults to ~hw_threads per worker; cap it so we
    # don't blow ulimit -u with many workers. See ray-project/ray#54988.
    os.environ["RAY_worker_num_grpc_internal_threads"] = "1"
    # this is required for numba to use the tbb threading layer
    if threading_layer == "tbb" and _load_tbb(log) is None:
        if log:
            log.warning("Could not initialse TBB threading layer for numba.")
        else:
            logging.warning("Could not initialse TBB threading layer for numba.")

    # these get passed to child processes
    env_vars = {
        "OMP_NUM_THREADS": str(nthreads),
        "OPENBLAS_NUM_THREADS": str(nthreads),
        "MKL_NUM_THREADS": str(nthreads),
        "VECLIB_MAXIMUM_THREADS": str(nthreads),
        "NPY_NUM_THREADS": str(nthreads),
        "JAX_ENABLE_X64": "True",
        "JAX_LOGGING_LEVEL": "ERROR",  # for the workers
        "NUMEXPR_NUM_THREADS": str(ne_threads),
        "PYTHONWARNINGS": "ignore:.*CUDA-enabled jaxlib is not installed.*",
        "NUMBA_THREADING_LAYER": threading_layer,
        "NUMBA_CACHE_DIR": os.environ["NUMBA_CACHE_DIR"],
        "MBEAMS_CACHE_DIR": os.environ["MBEAMS_CACHE_DIR"],
        "RAY_worker_num_grpc_internal_threads": "1",
    }
    return env_vars


def init_ray(nworkers, ray_address="local", runtime_env=None, object_store_memory=None, log=None):
    """Initialise Ray for a pfb-imaging sub-command.

    With ray_address="local" (the default) a fresh private Ray instance is
    started even if another cluster is running on the node. Any other value
    is treated as the address of an existing cluster to connect to, in which
    case cluster properties (num_cpus, object_store_memory) must not be
    passed to ray.init and are therefore dropped.
    """
    # deferred: optional heavy runtime (ray)
    import ray

    logger = log or logging

    if ray.is_initialized():
        logger.warning("Ray is already initialised. Requested Ray settings will be ignored.")
        return

    if ray_address == "local":
        ray.init(
            address="local",
            num_cpus=nworkers,
            object_store_memory=object_store_memory,
            logging_level="INFO",
            runtime_env=runtime_env,
        )
    else:
        logger.info(f"Connecting to existing Ray cluster at {ray_address}")
        ray.init(
            address=ray_address,
            logging_level="INFO",
            runtime_env=runtime_env,
        )


def setup_ray_worker():
    logger = pfb_logging.get_logger("RAY_WORKER")
    logger.setLevel(logging.ERROR)
    # TBB is optional now that it lives behind the [x86] extra (and cannot be
    # installed at all off x86_64). Warn rather than raise: numba falls back to
    # its OpenMP layer on its own, and a worker that threads through libgomp is
    # far better than a worker that refuses to start.
    if _default_threading_layer() == "tbb" and _load_tbb() is None:
        logger.warning(
            "Could not initialise the TBB threading layer for numba in this worker process; "
            "numba will fall back to its default layer. Install pfb-imaging[x86] to restore TBB."
        )
