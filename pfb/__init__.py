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


def set_threads(nthreads: int):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    # TODO - does this result in thread over-subscription?
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(nthreads))
