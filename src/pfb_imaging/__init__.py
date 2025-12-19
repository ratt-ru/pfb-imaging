import os
import sys
import logging

def set_envs(nthreads, ncpu):
    # these seem to have more sensible defaults 
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NPY_NUM_THREADS"] = str(nthreads)
    os.environ["JAX_ENABLE_X64"] = 'True'
    os.environ["JAX_LOGGING_LEVEL"] = 'INFO'  # for th emain process
    ne_threads = min(ncpu, nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)
    # this may be required for numba parallelism
    # find python and set LD_LIBRARY_PATH
    paths = sys.path
    ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
    if len(ppath):
        ldpath = ppath[0].replace('bin', 'lib')
        ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ["LD_LIBRARY_PATH"] = ':'.join([ldpath, ldcurrent]).rstrip(':')
    else:
        raise RuntimeError("Could not set LD_LIBRARY_PATH for TBB")
    os.environ["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"

    # these get passed to child processes
    env_vars = {
        "OMP_NUM_THREADS": str(nthreads),
        "OPENBLAS_NUM_THREADS": str(nthreads),
        "MKL_NUM_THREADS": str(nthreads),
        "VECLIB_MAXIMUM_THREADS": str(nthreads),
        "NPY_NUM_THREADS": str(nthreads),
        "JAX_ENABLE_X64": 'True',
        "JAX_LOGGING_LEVEL": 'ERROR',  # for the workers
        "NUMEXPR_NUM_THREADS": str(ne_threads),
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "PYTHONWARNINGS": "ignore:.*CUDA-enabled jaxlib is not installed.*"
    }
    return env_vars


def set_client(nworkers, log, stack=None, host_address=None,
               direct_to_workers=False, client_log_level=None):

    import warnings
    warnings.filterwarnings("ignore", message="Port 8787 is already in use")
    if client_log_level == 'error':
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("bokeh").setLevel(logging.ERROR)
        logging.getLogger("tornado").setLevel(logging.CRITICAL)
    elif client_log_level == 'warning':
        logging.getLogger("distributed").setLevel(logging.WARNING)
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)
    elif client_log_level == 'info':
        logging.getLogger("distributed").setLevel(logging.INFO)
        logging.getLogger("bokeh").setLevel(logging.INFO)
        logging.getLogger("tornado").setLevel(logging.INFO)
    elif client_log_level == 'debug':
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
        dask.config.set({
                'distributed.comm.compression': {
                    'on': True,
                    'type': 'lz4'
                }
        })
        cluster = LocalCluster(processes=True,
                               n_workers=nworkers,
                               threads_per_worker=1,
                               memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                               asynchronous=False)
        if stack is not None:
            cluster = stack.enter_context(cluster)
        client = Client(cluster,
                        direct_to_workers=direct_to_workers)
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
