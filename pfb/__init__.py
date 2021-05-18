"""
Pre-conditioned Forward Backward Clean algorithm

author - Landman Bester
email  - lbester@ska.ac.za
date   - 31/03/2020
"""
__version__ = '0.0.1'


# TODO - there must be a better way to do this
# these environment variables need to be set before importing numpy
# but we need to parse our arguments to figure out how many threads
# to use in the first place

import os

def set_threads(nthreads: int, nbands: int, mem_limit: int):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    # TODO - does this result in thread over-subscription?
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
    import dask
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(nthreads))


def set_client(nthreads: int, mem_limit: int, nworkers: int,
               nthreads_per_worker: int, address, stack, log):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    # TODO - does this result in thread over-subscription?
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

    # set up client
    if address is not None:
        from distributed import Client
        print("Initialising distributed client.", file=log)
        client = stack.enter_context(Client(address))
        # if using distributed client the gridder can use all available threads
    else:
        from dask.distributed import Client, LocalCluster
        print("Initialising client with LocalCluster.", file=log)
        cluster = LocalCluster(processes=True, n_workers=nworkers,
                               threads_per_worker=nthreads_per_worker,
                               memory_limit=str(mem_limit/nworkers)+'GB')
        cluster = stack.enter_context(cluster)
        client = stack.enter_context(Client(cluster))

    from pfb.scheduling import install_plugin
    client.run_on_scheduler(install_plugin)
