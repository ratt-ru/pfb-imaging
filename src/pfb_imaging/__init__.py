import ctypes
import importlib
import logging
import os
from importlib.metadata import version

from pfb_imaging.utils import logging as pfb_logging

__version__ = "0.0.9"
pfb_version = version("pfb-imaging")
# This need to happen before importing numba
os.environ["NUMBA_THREADING_LAYER"] = "tbb"


def set_envs(nthreads, ncpu, log=None):
    # disable Ray dashboard — it hangs on macOS without a display server
    os.environ["RAY_DISABLE_DASHBOARD"] = "1"
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    # these seem to have more sensible defaults
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
    # this is required for numba to use the tbb threaing layer
    tbb_path = None
    try:
        dist = importlib.metadata.distribution("tbb")
        for f in dist.files:
            if str(f).endswith("/libtbb.so"):
                tbb_path = str(dist.locate_file(f).resolve())
                ctypes.CDLL(tbb_path)
                break
    except importlib.metadata.PackageNotFoundError:
        # TBB installed via conda (not pip): search env lib dir for libtbb
        import sys
        import glob
        lib_dir = os.path.join(os.path.dirname(sys.executable), "..", "lib")
        for pattern in ("libtbb.dylib", "libtbb.*.dylib", "libtbb.so", "libtbb.so.*"):
            matches = glob.glob(os.path.join(lib_dir, pattern))
            if matches:
                try:
                    ctypes.CDLL(matches[0])
                    tbb_path = matches[0]
                except OSError:
                    pass
                break
        if tbb_path is None:
            # conda puts libtbb on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH; numba
            # will find it via the dynamic linker even without explicit pre-load.
            tbb_path = "conda-managed"
    if tbb_path is None:
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
        "NUMBA_THREADING_LAYER": "tbb",
        "RAY_DISABLE_DASHBOARD": "1",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    }
    return env_vars


def setup_ray_worker():
    logger = pfb_logging.get_logger("RAY_WORKER")
    logger.setLevel(logging.ERROR)
    tbb_path = None
    try:
        dist = importlib.metadata.distribution("tbb")
        for f in dist.files:
            if str(f).endswith("/libtbb.so"):
                tbb_path = str(dist.locate_file(f).resolve())
                ctypes.CDLL(tbb_path)
                break
    except importlib.metadata.PackageNotFoundError:
        import sys
        import glob
        lib_dir = os.path.join(os.path.dirname(sys.executable), "..", "lib")
        for pattern in ("libtbb.dylib", "libtbb.*.dylib", "libtbb.so", "libtbb.so.*"):
            matches = glob.glob(os.path.join(lib_dir, pattern))
            if matches:
                try:
                    ctypes.CDLL(matches[0])
                    tbb_path = matches[0]
                except OSError:
                    pass
                break
        if tbb_path is None:
            tbb_path = "conda-managed"
    if tbb_path is None:
        logger.error_and_raise("Could not initialse TBB threading layer for numba in worker process.", RuntimeError)


def set_client(nworkers, log, stack=None, host_address=None, direct_to_workers=False, client_log_level=None):
    import warnings

    warnings.filterwarnings("ignore", message="Port 8787 is already in use")
    if client_log_level == "error":
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("bokeh").setLevel(logging.ERROR)
        logging.getLogger("tornado").setLevel(logging.CRITICAL)
    elif client_log_level == "warning":
        logging.getLogger("distributed").setLevel(logging.WARNING)
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)
    elif client_log_level == "info":
        logging.getLogger("distributed").setLevel(logging.INFO)
        logging.getLogger("bokeh").setLevel(logging.INFO)
        logging.getLogger("tornado").setLevel(logging.INFO)
    elif client_log_level == "debug":
        logging.getLogger("distributed").setLevel(logging.DEBUG)
        logging.getLogger("bokeh").setLevel(logging.DEBUG)
        logging.getLogger("tornado").setLevel(logging.DEBUG)

    import dask

    # set up client
    host_address = host_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
    if host_address is not None:
        from distributed import Client

        log.info("Initialising distributed client.")
        if stack is not None:
            client = stack.enter_context(Client(host_address))
        else:
            client = Client(host_address)
    else:
        from dask.distributed import Client, LocalCluster

        log.info("Initialising client with LocalCluster.")
        dask.config.set({"distributed.comm.compression": {"on": True, "type": "lz4"}})
        cluster = LocalCluster(
            processes=True,
            n_workers=nworkers,
            threads_per_worker=1,
            memory_limit=0,  # str(mem_limit/nworkers)+'GB'
            asynchronous=False,
        )
        if stack is not None:
            cluster = stack.enter_context(cluster)
        client = Client(cluster, direct_to_workers=direct_to_workers)
        if stack is not None:
            client = stack.enter_context(client)

    client.wait_for_workers(nworkers)
    dashboard_url = client.dashboard_link
    log.info(f"Dask Dashboard URL at {dashboard_url}")

    return client
