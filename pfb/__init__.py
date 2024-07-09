import os
import sys
# import psutil
# import resource

# mem_total = psutil.virtual_memory().total
# _, hardlim = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (mem_total, hardlim))

def set_envs(nthreads, ncpu):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
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


def set_client(nworkers, stack, log, host_address=None):
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
                    'type': 'blosc'
                }
        })
        cluster = LocalCluster(processes=True,
                                n_workers=nworkers,
                                threads_per_worker=1,
                                memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                                asynchronous=False)
        cluster = stack.enter_context(cluster)
        client = stack.enter_context(Client(cluster,
                                            direct_to_workers=False))

    client.wait_for_workers(nworkers)
    dashboard_url = client.dashboard_link
    print(f"Dask Dashboard URL at {dashboard_url}", file=log)

    return


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
