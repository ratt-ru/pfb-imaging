import ctypes
import importlib
import logging
import os
from importlib.metadata import version

from pfb_imaging.utils import logging as pfb_logging

__version__ = "0.0.8"
pfb_version = version("pfb-imaging")
# This need to happen before importing numba
os.environ["NUMBA_THREADING_LAYER"] = "tbb"


def set_envs(nthreads, ncpu, log=None):
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
    dist = importlib.metadata.distribution("tbb")
    tbb_path = None
    for f in dist.files:
        if str(f).endswith("/libtbb.so"):
            tbb_path = str(dist.locate_file(f).resolve())
            ctypes.CDLL(tbb_path)
            break
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
    }
    return env_vars


def setup_ray_worker():
    logger = pfb_logging.get_logger("RAY_WORKER")
    logger.setLevel(logging.ERROR)
    dist = importlib.metadata.distribution("tbb")
    tbb_path = None
    for f in dist.files:
        if str(f).endswith("/libtbb.so"):
            tbb_path = str(dist.locate_file(f).resolve())
            ctypes.CDLL(tbb_path)
            break
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


def logo():
    print("""
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """)


def logo1():
    print(
        "\033[36m"
        + """
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """
        + "\033[0m"
    )


def logo2():
    logo_text = """
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """
    # Cyan for filled blocks, white for outline
    colored = logo_text.replace("█", "\033[36m█\033[0m").replace("░", "\033[37m░\033[0m")
    print(colored)


def logo3():
    print(
        "\033[100m\033[96m"
        + """
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """
        + "\033[0m"
    )
