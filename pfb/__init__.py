"""
pfb-imaging

author - Landman Bester
email  - lbester@ska.ac.za
date   - 31/03/2020

MIT License

Copyright (c) 2020 Rhodes University Centre for Radio Astronomy Techniques & Technologies (RATT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
__version__ = '0.0.4'

import os
import sys
import psutil
import resource

mem_total = psutil.virtual_memory().total
_, hardlim = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (mem_total, hardlim))

def set_client(opts, stack, log,
               scheduler='distributed',
               auto_restrict=True):

    from omegaconf import open_dict
    # attempt somewhat intelligent default setup
    import multiprocessing
    nthreads_max = max(multiprocessing.cpu_count(), 1)
    if opts.nvthreads is None:
        # we allocate half by default
        nthreads_tot = max(nthreads_max//2, 1)
        if opts.scheduler in ['single-threaded', 'sync']:
            with open_dict(opts):
                opts.nvthreads = nthreads_tot
        else:
            ndask_chunks = opts.nthreads_dask*opts.nworkers
            nvthreads = max(nthreads_tot//ndask_chunks, 1)
            with open_dict(opts):
                opts.nvthreads = nvthreads

    os.environ["OMP_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["MKL_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nvthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(opts.nvthreads)
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

    import numexpr as ne
    max_cores = ne.detect_number_of_cores()
    ne_threads = min(max_cores, opts.nvthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ne_threads)

    import dask
    if scheduler=='distributed':
        # we probably always want compression


        # set up client
        host_address = opts.host_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
        if host_address is not None:
            from distributed import Client
            print("Initialising distributed client.", file=log)
            client = stack.enter_context(Client(host_address))
        else:
            if opts.nthreads_dask * opts.nvthreads > nthreads_max:
                print("Warning - you are attempting to use more threads than "
                      "available. This may lead to suboptimal performance.",
                      file=log)
            from dask.distributed import Client, LocalCluster
            print("Initialising client with LocalCluster.", file=log)
            dask.config.set({
                    'distributed.comm.compression': {
                        'on': True,
                        'type': 'blosc'
                    }
            })
            cluster = LocalCluster(processes=opts.nworkers > 1,
                                    n_workers=opts.nworkers,
                                    threads_per_worker=opts.nthreads_dask,
                                    memory_limit=0,  # str(mem_limit/nworkers)+'GB'
                                    asynchronous=False)
            cluster = stack.enter_context(cluster)
            client = stack.enter_context(Client(cluster,
                                                direct_to_workers=True))

        if auto_restrict:
            from quartical.scheduling import install_plugin
            client.run_on_scheduler(install_plugin)
        client.wait_for_workers(opts.nworkers)
        dashboard_url = client.dashboard_link
        print(f"Dask Dashboard URL at {dashboard_url}", file=log)

    elif scheduler in ['sync', 'single-threaded']:
        dask.config.set(scheduler=scheduler)
        print(f"Initialising with synchronous scheduler",
              file=log)
    elif scheduler=='threads':
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(opts.nthreads_dask))
        print(f"Initialising ThreadPool with {opts.nthreads_dask} threads",
              file=log)
    elif scheduler=='processes':
        # TODO - why is the performance so terrible in this case?
        from multiprocessing.pool import Pool
        dask.config.set(pool=Pool(opts.nthreads_dask))
        print(f"Initialising Pool with {opts.nthreads_dask} processes",
              file=log)
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
