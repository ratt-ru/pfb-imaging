"""
Pre-conditioned Forward Backward Clean algorithm

author - Landman Bester
email  - lbester@ska.ac.za
date   - 31/03/2020
"""
__version__ = '0.0.1'

import os

def set_client(opts, stack, log, scheduler='distributed'):

    from omegaconf import open_dict
    # number of threads per worker
    if opts.nthreads is None:
        if opts.host_address is not None:
            raise ValueError("You have to specify nthreads when using a distributed scheduler")
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
        with open_dict(opts):
            opts.nthreads = nthreads
    else:
        nthreads = int(opts.nthreads)

    # configure memory limit
    if opts.mem_limit is None:
        if opts.host_address is not None:
            raise ValueError("You have to specify mem-limit when using a distributed scheduler")
        import psutil
        mem_limit = int(psutil.virtual_memory()[1]/1e9)  # all available memory by default
        with open_dict(opts):
            opts.mem_limit = mem_limit
    else:
        mem_limit = int(opts.mem_limit)

    if opts.nworkers is None:
        raise ValueError("You have to specify the number of workers")
    else:
        nworkers = opts.nworkers

    if opts.nthreads_per_worker is None:
        nthreads_per_worker = 1
        with open_dict(opts):
            opts.nthreads_per_worker = nthreads_per_worker
    else:
        nthreads_per_worker = int(opts.nthreads_per_worker)

    # the number of chunks being read in simultaneously is equal to
    # the number of dask threads
    if opts.scheduler == 'threads':
        # if using threads we use one dask thread per band
        nthreads_dask = max(opts.nthreads//opts.nband, 1)
    else:
        nthreads_dask = nworkers * nthreads_per_worker

    if opts.nvthreads is None:
        if opts.scheduler == 'single-threaded':
            nvthreads = nthreads
        elif opts.host_address is not None:
            nvthreads = max(nthreads//nthreads_per_worker, 1)
        else:
            nvthreads = max(nthreads//nthreads_dask, 1)
        with open_dict(opts):
            opts.nvthreads = nvthreads

    os.environ["OMP_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["MKL_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nvthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(opts.nvthreads)
    # avoids numexpr error, probably don't want more than 10 vthreads for ne anyway
    import numexpr as ne
    max_cores = ne.detect_number_of_cores()
    ne_threads = min(max_cores, opts.nvthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)

    if scheduler=='distributed':
        # set up client
        if opts.host_address is not None:
            from distributed import Client
            print("Initialising distributed client.", file=log)
            client = stack.enter_context(Client(address))
        else:
            if nthreads_dask * opts.nvthreads > opts.nthreads:
                print("Warning - you are attempting to use more threads than "
                      "available. This may lead to suboptimal performance.",
                      file=log)
            from dask.distributed import Client, LocalCluster
            print("Initialising client with LocalCluster.", file=log)
            cluster = LocalCluster(processes=True, n_workers=nworkers,
                                   threads_per_worker=nthreads_per_worker,
                                   memory_limit=0)  # str(mem_limit/nworkers)+'GB'
            cluster = stack.enter_context(cluster)
            client = stack.enter_context(Client(cluster))

        from pfb.scheduling import install_plugin
        client.run_on_scheduler(install_plugin)
    elif scheduler in ['sync', 'single-threaded']:
        import dask
        dask.config.set(scheduler=scheduler)
    elif scheduler=='threads':
        import dask
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(nthreads_dask))
    else:
        raise ValueError(f"Unknown scheduler option {opts.scheduler}")

    # return updated opts
    return opts


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
