"""Per-band Ray worker co-locating all of a band's deconvolution state.

Bands couple only through the prox, so everything else a deconv run needs —
the PSF-convolution Hessian (with its in-actor CG), the wavelet dictionary,
and the gridding inputs for the exact residual — is per-band state. One
``_BandWorkerImpl`` actor per band owns all three roles, so a run needs
exactly ``nband`` worker processes (instead of one process per role per
band): one numba JIT warm-up, one FFT-plan set and one thread pool per band,
and the partition data for the residual stays pinned in its actor for the
life of the run instead of being re-fetched every major cycle.

``BandWorkerPool`` spawns the actors and exposes cube-level role methods.
The ``HessTreeRay`` and ``PsiNocopytRay`` facades (and the driver's residual
step) share one pool; roles are initialised on demand, so e.g. an ista
solver never pays for the wavelet machinery. For ``nband == 1`` the pool
runs a single in-process worker (no Ray overhead), matching the facades'
historical local fallback.
"""

import numpy as np


class _BandWorkerImpl:
    """One band's co-located deconv state; roles initialised on demand.

    Role imports are deferred into the init methods: they execute in the
    actor process (keeping actor startup lazy) and avoid a module-level
    import cycle with ``operators.hessian``/``operators.psi``, whose Ray
    facades construct a ``BandWorkerPool``.
    """

    def __init__(self, nthreads):
        # Load TBB in this process — ctypes.CDLL in the driver process does
        # not carry over to forked/spawned Ray workers.
        # deferred: worker-side setup; keeps driver-side import of this module light
        import ctypes
        import importlib.metadata

        import numba

        dist = importlib.metadata.distribution("tbb")
        for f in dist.files:
            if str(f).endswith("/libtbb.so"):
                ctypes.CDLL(str(dist.locate_file(f).resolve()))
                break
        numba.set_num_threads(min(nthreads, numba.config.NUMBA_NUM_THREADS))

        # deferred: worker-side setup; keeps driver-side import light
        # resize ducc thread pool for worker
        from ducc0.misc import resize_thread_pool

        resize_thread_pool(nthreads)
        self._nthreads = nthreads
        self._hess = None
        self._psib = None
        self._parts = None
        self._dirty = None

    # --- band loading (worker-side reads; the driver never touches these arrays) ---

    def load_band(self, store_url, node_name):
        """Read this band's worker-side inputs straight from the ``.dt`` store.

        Loads the per-partition gridding inputs (``UVW``/``WEIGHT``/``MASK``/
        ``FREQ``/``BEAM``), the Hessian inputs (``PSFHAT``/``BEAM``/``wsum``)
        and the raw ``DIRTY`` in this process, so vis-scale data never enters
        the driver or the Ray object store. Selective loads only, then the
        Dataset handles are released and cycles collected (the ``stokes_vis``
        discipline; docs/wiki/memory-and-ray.md). Everything loaded is held
        for the life of the run — a deliberate RSS-for-work trade, visible in
        get_mem telemetry.
        """
        # deferred: worker-side loading; keeps driver-side import light
        import gc

        import xarray as xr

        try:
            band = xr.open_datatree(store_url, engine="zarr", chunks=None)[node_name]
            self._dirty = band.ds.DIRTY.values  # (corr, nx, ny)
            parts = []
            hess_parts = []
            for cname in sorted(band.children):
                child = band[cname].ds
                pds = child[["UVW", "WEIGHT", "MASK", "FREQ", "BEAM"]].load()
                pds.attrs.update(child.attrs)
                hess_parts.append(
                    {
                        # HessianTree expects the real, non-negative
                        # Fourier-domain PSF magnitude (the legacy sara()
                        # `abspsf` convention) -- the stored PSFHAT is the raw
                        # complex FFT of the PSF, so it must be abs()'d here,
                        # else the "Hessian" carries a phase and is no longer
                        # Hermitian-positive, breaking CG.
                        "psfhat": np.abs(child.PSFHAT.values),
                        "beam": pds.BEAM.values,
                        "wsum": np.asarray(child.attrs["wsum"]),
                    }
                )
                parts.append(pds)
            self._parts = parts
            self._hess_parts = hess_parts
        finally:
            # deserialised/lazy xarray objects sit in reference cycles that
            # refcounting cannot free
            gc.collect()

    # --- Hessian role ---

    def init_hess(self, partitions, nx, ny, nx_psf, ny_psf, eta, wsum):
        # deferred: import cycle with operators.hessian
        from pfb_imaging.operators.hessian import HessianTree

        if partitions is None:
            partitions = getattr(self, "_hess_parts", None)
            if partitions is None:
                raise RuntimeError("no partitions passed and none loaded; call load_band first")
        self._hess = HessianTree(partitions, nx, ny, nx_psf, ny_psf, eta=eta, nthreads=self._nthreads, wsum=wsum)
        self._hess.dot(np.zeros((nx, ny)))  # warm up the FFT plans

    def hess_dot(self, x):
        return self._hess.dot(x)

    def cg(self, rhs, x0, tol, maxit, minit, verbosity):
        # deferred: worker-side only; keeps driver-side import light
        from pfb_imaging.opt.pcg import pcg_numba

        if x0 is not None:
            # Ray deserialises task args as read-only zero-copy views;
            # pcg_numba updates x0 in place
            x0 = x0.copy()
        return pcg_numba(
            lambda z: self._hess.dot(z)[0],
            rhs,
            x0=x0,
            tol=tol,
            maxit=maxit,
            minit=minit,
            verbosity=verbosity,
        )

    # --- Psi (wavelet dictionary) role ---

    def init_psi(self, nx, ny, bases, nlevel):
        # deferred: import cycle with operators.psi
        from pfb_imaging.operators.psi import psi_band_maker_nocopyt

        self._psib = psi_band_maker_nocopyt(nx, ny, bases, nlevel)
        # pre-allocate output buffers (reused every call; the jitclass zeros
        # them internally) and trigger JIT compilation
        self._alphao = np.empty((self._psib.nbasis, self._psib.nxmax, self._psib.nymax))
        self._xo = np.empty((nx, ny))
        self._psib.dot(np.zeros((nx, ny)), self._alphao)
        self._psib.hdot(self._alphao, self._xo)
        return int(self._psib.nxmax), int(self._psib.nymax)

    def psi_dot(self, x):
        self._psib.dot(x, self._alphao)
        return self._alphao

    def psi_hdot(self, alpha):
        self._psib.hdot(alpha, self._xo)
        return self._xo

    # --- exact residual role ---

    def residual(self, model, cell_rad, epsilon, do_wgridding, double_accum):
        # deferred: worker-side only; keeps driver-side import light
        from pfb_imaging.operators.gridder import residual_from_partitions

        return residual_from_partitions(
            self._dirty,
            self._parts,
            model,
            cell_rad,
            nthreads=self._nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )

    # --- telemetry ---

    def get_mem(self):
        """Post-gc memory telemetry (docs/wiki/memory-and-ray.md)."""
        # deferred: telemetry-only, worker-side
        import gc
        import os
        import resource

        import psutil

        gc.collect()
        return {
            "pid": os.getpid(),
            "rss_gb": psutil.Process().memory_info().rss / 2**30,
            "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30,
        }

    def get_memory_mb(self):
        """RSS/USS in MB (rss includes shared pages; uss is private only)."""
        # deferred: telemetry-only, worker-side
        import psutil

        info = psutil.Process().memory_full_info()
        return {"rss": info.rss / 1e6, "uss": info.uss / 1e6}


class BandWorkerPool:
    """nband band workers plus cube-level dispatch for their role methods.

    Args:
        nband: Number of imaging bands (one worker each).
        nthreads: Total thread budget, divided across the workers.
    """

    def __init__(self, nband, nthreads=1):
        self.nband = nband
        self.nthreads_per_band = max(1, nthreads)
        if nband == 1:
            # single band: in-process worker, no Ray overhead
            self._local = _BandWorkerImpl(nthreads)
            self.actors = None
        else:
            # deferred: optional heavy runtime (ray); the single-band local path never needs it
            import ray

            # num_cpus is Ray's *scheduling* resource claim,
            # kept nominal because the workers are thread-pool-bound,
            # not Ray-scheduler-bound: compute concurrency comes from each
            # worker's own numba/FFT thread count, and any claim that scales
            # with nband can exceed the cluster's num_cpus and deadlock on
            # the init ray.get for large enough nband (e.g. the default
            # nworkers=1 driver cluster).
            self._local = None
            worker_cls = ray.remote(num_cpus=1e-2)(_BandWorkerImpl)
            self.actors = [worker_cls.remote(self.nthreads_per_band) for _ in range(nband)]

    def _map(self, method, per_band_args):
        """Run ``method(*args)`` on every band worker; return per-band results."""
        if self.actors is None:
            return [getattr(self._local, method)(*per_band_args[0])]
        # deferred: optional heavy runtime (ray)
        import ray

        return ray.get([getattr(a, method).remote(*args) for a, args in zip(self.actors, per_band_args)])

    # --- band loading ---

    def load_bands(self, store_url, node_names):
        """Each worker reads its own band node from the ``.dt`` store."""
        if len(node_names) != self.nband:
            raise ValueError(f"got {len(node_names)} band nodes for {self.nband} workers")
        self._map("load_band", [(store_url, node_names[b]) for b in range(self.nband)])

    # --- Hessian role ---

    def init_hess(self, partitions_per_band, nx, ny, nx_psf, ny_psf, etas, wsums):
        """Build per-band HessianTrees; ``partitions_per_band=None`` uses load_bands data."""
        self._map(
            "init_hess",
            [
                (
                    None if partitions_per_band is None else partitions_per_band[b],
                    nx,
                    ny,
                    nx_psf,
                    ny_psf,
                    etas[b],
                    wsums[b],
                )
                for b in range(self.nband)
            ],
        )

    def hess_dot(self, x):
        out = np.zeros_like(x)
        for b, res in enumerate(self._map("hess_dot", [(x[b],) for b in range(self.nband)])):
            out[b] = res[0]
        return out

    def hess_cg(self, rhs, x0, tol, maxit, minit, verbosity):
        out = np.zeros_like(rhs)
        args = [(rhs[b], None if x0 is None else x0[b], tol, maxit, minit, verbosity) for b in range(self.nband)]
        for b, res in enumerate(self._map("cg", args)):
            out[b] = res
        return out

    # --- Psi role ---

    def init_psi(self, nx, ny, bases, nlevel):
        shapes = self._map("init_psi", [(nx, ny, bases, nlevel)] * self.nband)
        return shapes[0]  # (nxmax, nymax), identical across bands

    def psi_dot(self, x, alphao):
        for b, res in enumerate(self._map("psi_dot", [(x[b],) for b in range(self.nband)])):
            alphao[b] = res

    def psi_hdot(self, alpha, xo):
        for b, res in enumerate(self._map("psi_hdot", [(alpha[b],) for b in range(self.nband)])):
            xo[b] = res

    # --- exact residual role ---

    def residual(self, model, cell_rad, epsilon=1e-7, do_wgridding=True, double_accum=True):
        """Exact per-band residual for a ``(nband, corr, nx, ny)`` model cube."""
        args = [(model[b], cell_rad, epsilon, do_wgridding, double_accum) for b in range(self.nband)]
        return np.stack(self._map("residual", args), axis=0)

    # --- telemetry ---

    def get_mem(self):
        """Per-worker post-gc memory telemetry (empty for the local path)."""
        if self.actors is None:
            return []
        # deferred: optional heavy runtime (ray)
        import ray

        return ray.get([a.get_mem.remote() for a in self.actors])
