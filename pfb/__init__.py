import os
import sys
import logging
# import psutil
# import resource

# mem_total = psutil.virtual_memory().total
# _, hardlim = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (mem_total, hardlim))

__version__ = '0.0.4'

def set_envs(nthreads, ncpu):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    os.environ["JAX_PLATFORMS"] = 'cpu'
    os.environ["JAX_ENABLE_X64"] = 'True'
    # this may be required for numba parallelism
    # find python and set LD_LIBRARY_PATH
    paths = sys.path
    ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
    if len(ppath):
        ldpath = ppath[0].replace('bin', 'lib')
        ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ["LD_LIBRARY_PATH"] = f'{ldpath}:{ldcurrent}'
        # TODO - should we fall over in else?

    ne_threads = min(ncpu, nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)


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
        print("Initialising distributed client.", file=log)
        client = stack.enter_context(Client(host_address))
    else:
        from dask.distributed import Client, LocalCluster
        print("Initialising client with LocalCluster.", file=log)
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
    print(f"Dask Dashboard URL at {dashboard_url}", file=log)

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
