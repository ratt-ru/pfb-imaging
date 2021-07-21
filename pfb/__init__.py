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


def set_client(args, stack, log):

    from omegaconf import open_dict
    # number of threads per worker
    if args.nthreads is None:
        if args.host_address is not None:
            raise ValueError("You have to specify nthreads when using a distributed scheduler")
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
        with open_dict(args):
            args.nthreads = nthreads
    else:
        nthreads = int(args.nthreads)

    # configure memory limit
    if args.mem_limit is None:
        if args.host_address is not None:
            raise ValueError("You have to specify mem-limit when using a distributed scheduler")
        import psutil
        mem_limit = int(psutil.virtual_memory()[1]/1e9)  # all available memory by default
        with open_dict(args):
            args.mem_limit = mem_limit
    else:
        mem_limit = int(args.mem_limit)

    if args.nworkers is None:
        raise ValueError("You have to specify the number of workers")
    else:
        nworkers = args.nworkers

    if args.nthreads_per_worker is None:
        nthreads_per_worker = 1
        with open_dict(args):
            args.nthreads_per_worker = nthreads_per_worker
    else:
        nthreads_per_worker = int(args.nthreads_per_worker)

    # the number of chunks being read in simultaneously is equal to
    # the number of dask threads
    nthreads_dask = nworkers * nthreads_per_worker

    if args.nvthreads is None:
        if args.host_address is not None:
            nvthreads = nthreads//nthreads_per_worker
        else:
            nvthreads = nthreads//nthreads_dask
        with open_dict(args):
            args.nvthreads = nvthreads

    os.environ["OMP_NUM_THREADS"] = str(args.nvthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.nvthreads)
    os.environ["MKL_NUM_THREADS"] = str(args.nvthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.nvthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(args.nvthreads)
    # TODO - does this result in thread over-subscription?
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.nvthreads)

    # set up client
    if args.host_address is not None:
        from distributed import Client
        print("Initialising distributed client.", file=log)
        client = stack.enter_context(Client(address))
    else:
        if nthreads_dask * args.nvthreads > args.nthreads:
            print("Warning - you are attempting to use more threads than available. "
                  "This may lead to suboptimal performance.", file=log)
        from dask.distributed import Client, LocalCluster
        print("Initialising client with LocalCluster.", file=log)
        cluster = LocalCluster(processes=True, n_workers=nworkers,
                               threads_per_worker=nthreads_per_worker,
                               memory_limit=str(mem_limit/nworkers)+'GB')
        cluster = stack.enter_context(cluster)
        client = stack.enter_context(Client(cluster))

    from pfb.scheduling import install_plugin
    client.run_on_scheduler(install_plugin)

    # return updated args
    return args


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