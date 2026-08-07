from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from ducc0.fft import c2r, r2c
from ducc0.misc import empty_noncritical
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from pfb_imaging.opt.pcg import pcg_numba as pcg
from pfb_imaging.utils.misc import taperf


def hessian_slice(
    x,
    xout=None,
    uvw=None,
    weight=None,
    vis_mask=None,
    freq=None,
    beam=None,
    cell=None,
    x0=0.0,
    y0=0.0,
    flip_u=False,
    flip_v=True,
    flip_w=False,
    do_wgridding=True,
    epsilon=1e-7,
    double_accum=True,
    nthreads=1,
    eta=None,
    wsum=None,
):
    """
    Apply vis space Hessian approximation on a slice of an image.

    Important!
    x0, y0, flip_u, flip_v and flip_w must be consistent with the
    conventions defined in pfb.operators.gridder.wgridder_conventions

    These are inputs here to allow for testing but should generally be taken
    from the attrs produced by pfb.operators.gridder.wgridder_conventions /
    pfb.operators.gridder.grid_partition
    """
    if not x.any():
        return np.zeros_like(x)
    nx, ny = x.shape
    mvis = dirty2vis(
        uvw=uvw,
        freq=freq,
        mask=vis_mask,
        dirty=x if beam is None else x * beam,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=x0,
        center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=epsilon,
        nthreads=nthreads,
        do_wgridding=do_wgridding,
        divide_by_n=False,
    )

    convim = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=mvis,
        wgt=weight,
        mask=vis_mask,
        dirty=xout,  # return in case xout is None
        npix_x=nx,
        npix_y=ny,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=x0,
        center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=epsilon,
        nthreads=nthreads,
        do_wgridding=do_wgridding,
        double_precision_accumulation=double_accum,
        divide_by_n=False,
    )

    if wsum is not None:
        convim /= wsum

    if beam is not None:
        convim *= beam

    if eta is not None:
        convim += eta * x

    return convim


def hessian_psf_slice(
    x,  # input image, not overwritten
    xpad=None,  # preallocated array to store padded image
    xhat=None,  # preallocated array to store FTd image
    xout=None,  # preallocated array to store output image
    abspsf=None,
    beam=None,
    lastsize=None,
    nthreads=1,
    eta=None,
):
    """
    Tikhonov regularised Hessian approx
    """
    nx, ny = x.shape
    xpad.fill(0.0)
    if beam is None:
        np.copyto(xpad[0:nx, 0:ny], x)
    else:
        xpad[0:nx, 0:ny] = x * beam
    r2c(xpad, axes=(0, 1), nthreads=nthreads, forward=True, inorm=0, out=xhat)
    xhat *= abspsf
    c2r(
        xhat,
        axes=(0, 1),
        forward=False,
        out=xpad,
        lastsize=lastsize,
        inorm=2,
        nthreads=nthreads,
        allow_overwriting_input=True,
    )
    np.copyto(xout, xpad[0:nx, 0:ny])

    if beam is not None:
        xout *= beam

    if eta:
        xout += x * eta

    return xout


def hess_direct_slice(
    x,  # input image, not overwritten
    xpad=None,  # preallocated array to store padded image
    xhat=None,  # preallocated array to store FTd image
    xout=None,  # preallocated array to store output image
    abspsf=None,
    taperxy=None,
    lastsize=None,
    nthreads=1,
    eta=1,
    mode="forward",
):
    """
    Note eta must be relative to wsum (peak of PSF)
    """
    nx, ny = x.shape
    xpad.fill(0.0)
    xpad[0:nx, 0:ny] = x * taperxy
    r2c(xpad, out=xhat, axes=(0, 1), forward=True, inorm=0, nthreads=nthreads)
    if mode == "forward":
        xhat *= abspsf + eta
    else:
        xhat /= abspsf + eta
    c2r(
        xhat,
        axes=(0, 1),
        forward=False,
        out=xpad,
        lastsize=lastsize,
        inorm=2,
        nthreads=nthreads,
        allow_overwriting_input=True,
    )
    np.copyto(xout, xpad[0:nx, 0:ny])
    xout *= taperxy
    return xout


class HessPSF(object):
    def __init__(
        self,
        nx,
        ny,
        abspsf,
        beam=None,
        eta=1.0,
        nthreads=1,
        cgtol=1e-3,
        cgmaxit=300,
        cgverbose=2,
        cgrf=25,
        taper_width=32,
        min_beam=5e-3,
    ):
        self.nx = nx
        self.ny = ny
        self.abspsf = abspsf
        self.nband, self.nx_psf, self.nyo2 = abspsf.shape
        if beam is not None and not (beam == 1).all():
            assert self.nband == beam.shape[0]
            assert self.nx == beam.shape[1]
            assert self.ny == beam.shape[2]
            self.beam = beam
        else:
            # self.beam = None
            self.beam = (None,) * self.nband
        self.ny_psf = 2 * (self.nyo2 - 1)
        self.nx_pad = self.nx_psf - self.nx
        self.ny_pad = self.ny_psf - self.ny
        self.nthreads = nthreads
        if isinstance(eta, float):
            self.eta = np.tile(eta, self.nband)
        else:
            try:
                self.eta = np.array(eta)
                assert self.eta.size == self.nband
            except Exception as e:
                raise e

        # per band tmp arrays
        self.xhat = empty_noncritical((self.nx_psf, self.nyo2), dtype="c16")
        self.xpad = empty_noncritical((self.nx_psf, self.ny_psf), dtype="f8")
        self.xout = empty_noncritical((self.nband, self.nx, self.ny), dtype="f8")

        # conjugate gradient params
        self.cgtol = cgtol
        self.cgmaxit = cgmaxit
        self.cgverbose = cgverbose
        self.cgrf = cgrf

        # taper for direct mode
        self.taperxy = taperf((nx, ny), taper_width)

        # for beam application in direct mode
        self.min_beam = min_beam

    def set_beam(self, beam):
        assert beam.shape == (self.nband, self.nx, self.ny)
        self.beam = beam

    def dot(self, x):
        if len(x.shape) == 3:
            xtmp = x
        elif len(x.shape) == 2:
            xtmp = x[None, :, :]
        else:
            raise ValueError("Unsupported number of input dimensions")

        nband, nx, ny = xtmp.shape
        assert nband == self.nband
        assert nx == self.nx
        assert ny == self.ny

        for b in range(nband):
            self.xpad.fill(0.0)
            if self.beam[b] is None:
                np.copyto(self.xpad[0:nx, 0:ny], xtmp[b])
            else:
                self.xpad[0:nx, 0:ny] = xtmp[b] * self.beam[b]
            r2c(self.xpad, axes=(0, 1), nthreads=self.nthreads, forward=True, inorm=0, out=self.xhat)
            self.xhat *= self.abspsf[b]
            c2r(
                self.xhat,
                axes=(0, 1),
                forward=False,
                out=self.xpad,
                lastsize=self.ny_psf,
                inorm=2,
                nthreads=self.nthreads,
                allow_overwriting_input=True,
            )
            if self.beam[b] is None:
                np.copyto(self.xout[b], self.xpad[0:nx, 0:ny])
            else:
                self.xout[b] = self.xpad[0:nx, 0:ny] * self.beam[b]
        self.xout += xtmp * self.eta[:, None, None]
        return self.xout

    def hdot(self, x):
        # Hermitian operator
        return self.dot(x)

    def idot(self, x, mode="psf", x0=None, init_x0=True):
        if len(x.shape) == 3:
            xtmp = x
        elif len(x.shape) == 2:
            xtmp = x[None, :, :]
        else:
            raise ValueError("Unsupported number of dimensions")

        nband, nx, ny = xtmp.shape
        assert nband == self.nband
        assert nx == self.nx
        assert ny == self.ny

        if x0 is None and init_x0:
            # initialise with direct estimate
            x0 = np.zeros_like(xtmp)
            for b in range(self.nband):
                x0[b] = hess_direct_slice(
                    xtmp[b],
                    xpad=self.xpad,
                    xhat=self.xhat,
                    xout=self.xout[b],
                    abspsf=self.abspsf[b],
                    taperxy=self.taperxy,
                    lastsize=self.ny_psf,
                    nthreads=self.nthreads,
                    eta=self.eta[b] * np.sqrt(nx * ny),
                    mode="backward",
                )
                if self.beam[b] is not None:
                    mask = (self.xout[b] > 0) & (self.beam[b] > self.min_beam)
                    self.xout[b, mask] /= self.beam[b, mask] ** 2
        else:
            x0 = np.zeros_like(xtmp)

        if mode == "direct":
            for b in range(self.nband):
                self.xout[b] = hess_direct_slice(
                    xtmp[b],
                    xpad=self.xpad,
                    xhat=self.xhat,
                    xout=self.xout[b],
                    abspsf=self.abspsf[b],
                    taperxy=self.taperxy,
                    lastsize=self.ny_psf,
                    nthreads=self.nthreads,
                    eta=self.eta[b] * np.sqrt(nx * ny),
                    mode="backward",
                )
                if self.beam[b] is not None:
                    mask = (self.xout[b] > 0) & (self.beam[b] > self.min_beam)
                    self.xout[b, mask] /= self.beam[b, mask] ** 2

        elif mode == "psf":
            for b in range(self.nband):
                hess = partial(
                    hessian_psf_slice,
                    xpad=self.xpad,
                    xhat=self.xhat,
                    xout=self.xout[b],
                    abspsf=self.abspsf[b],
                    beam=self.beam[b],
                    lastsize=self.ny_psf,
                    nthreads=self.nthreads,
                    eta=self.eta[b],
                )
                self.xout[b] = pcg(
                    hess,
                    xtmp[b],
                    x0=x0[b],
                    tol=self.cgtol,
                    maxit=self.cgmaxit,
                    minit=3,
                    verbosity=self.cgverbose,
                    report_freq=self.cgrf,
                    backtrack=False,
                    return_resid=False,
                )
        else:
            raise ValueError(f"Unknown mode {mode}")

        return self.xout.copy()


ETA_MODES = ("invbeam", "invbeam2", "radial", "radial-invbeam")


def eta_profile(partitions, eta, mode, ny, nx, cap=1e2):
    """Spatially varying Tikhonov coefficient for the PSF-convolution preconditioner.

    ``lambda_max(M^-1 H_exact)`` is monotone decreasing in ``M`` in the PSD
    order, so raising ``eta`` where the PSF approximation to the Hessian is
    worst lowers it, and with it the smallest diverging ``gamma`` (issue #287).
    The profile enters the **preconditioner only** -- ``eta`` is absent from
    ``gridder.residual_from_partitions``, so the fixed point and the flux scale
    are untouched and a profile only damps each forward update.

    Every mode is normalised to ``eta`` where the operator is trustworthy (beam
    peak / tangent point) and has dynamic range exactly ``cap``, so ``--eta``
    keeps its meaning across modes, ``lambda_min(M)`` (hence the CG iteration
    count) cannot degrade, and the modes are directly comparable:

    * ``invbeam``  -- ``1/B_eff``, ``B_eff`` the effective mosaic beam (below);
    * ``invbeam2`` -- ``1/B_eff^2``, twice the log-slope of ``invbeam``;
    * ``radial``   -- ``1 + (cap-1) r^2``, ``r`` in field half-widths, aimed at
      the w-term mismatch (which grows with distance from the tangent point)
      rather than at the beam;
    * ``radial-invbeam`` -- the product, clipped to ``cap``.

    Note: ``r`` is measured from the image centre. With ``--target`` the tangent
    point is offset from it by ``l0/cell`` pixels, which this ignores.

    Args:
        partitions: the ``HessianTree`` partition dicts (``beam``, ``wsum``).
        eta: baseline coefficient -- the value used where the operator is trusted.
        mode: one of ``ETA_MODES``, or None for a uniform ``eta``.
        ny, nx: image dimensions ((Y, X) order, wiki D19).
        cap: dynamic range of the profile; also bounds the beam skirt, where
            ``1/B_eff`` would otherwise diverge.

    Returns:
        ``eta`` unchanged (float) when ``mode`` is None, else a
        ``(ncorr, ny, nx)`` array with minimum ``eta`` and maximum ``eta*cap``.

    Raises:
        ValueError: unknown mode, or ``cap < 1``.
    """
    if mode is None:
        return eta
    if mode not in ETA_MODES:
        raise ValueError(f"unknown eta_mode '{mode}'; choose from {ETA_MODES}")
    if cap < 1.0:
        raise ValueError(f"eta_cap must be >= 1 (got {cap}); the profile floor is eta itself")
    ncorr = partitions[0]["wsum"].size
    f = np.ones((ncorr, ny, nx))
    if "invbeam" in mode:
        # The mosaic response the operator actually carries is the wsum-weighted
        # mean of B_p^2: both M and H_exact apply the beam on BOTH sides (wiki
        # D23), so B^2 -- not the band node's first-power BEAM -- is what
        # multiplies the curvature.
        b2 = np.zeros((ncorr, ny, nx))
        wsum = np.zeros(ncorr)
        for p in partitions:
            b2 += p["wsum"][:, None, None] * p["beam"].astype(np.float64) ** 2
            wsum += p["wsum"]
        b2 /= wsum[:, None, None]
        b2 /= b2.max(axis=(1, 2), keepdims=True)  # peak 1 per correlation
        b = b2 if mode.endswith("2") else np.sqrt(b2)
        f *= np.clip(1.0 / np.maximum(b, 1.0 / cap), 1.0, cap)
    if mode.startswith("radial"):
        yy, xx = np.mgrid[0:ny, 0:nx]
        r2 = ((xx - nx / 2) / (nx / 2)) ** 2 + ((yy - ny / 2) / (ny / 2)) ** 2
        f *= 1.0 + (cap - 1.0) * r2[None]
    return eta * np.minimum(f, cap)


def freq_correlation(freq_out, length_scale):
    """Squared-exponential correlation matrix over the imaged band.

    Evaluated at the given frequencies, which need not be evenly spaced --
    ``freq_out`` is the wsum-weighted effective frequency and is data-dependent
    (wiki D28). The metric is linear in frequency with the length scale given as
    a fraction of the band span, so the knob reads directly ("correlated over
    half the band") and transfers between UHF and L-band runs. A log-frequency
    metric is the same thing to 4% over a full 2:1 band, so the extra
    parameterisation is not worth carrying.

    Args:
        freq_out: ``(nband,)`` band frequencies in Hz.
        length_scale: Correlation length as a fraction of the ``freq_out`` span.

    Returns:
        ``(nband, nband)`` unit-diagonal correlation matrix.
    """
    freq_out = np.asarray(freq_out, dtype=np.float64)
    span = float(freq_out.max() - freq_out.min())
    d = (freq_out[:, None] - freq_out[None, :]) / (length_scale * span)
    return np.exp(-0.5 * d**2)


def freq_precision(freq_out, length_scale, cap=10.0):
    """Normalised frequency precision for the preconditioner's GP prior (issue #307).

    Generalises ``--eta`` from a scalar to an ``nband x nband`` matrix: today's
    preconditioner carries ``K_nu^-1 = eta * I``, and this returns the ``C^-1_n``
    that replaces the identity. The prior enters ``M`` **only** -- it is absent
    from ``gridder.residual_from_partitions`` -- so the fixed point and the flux
    scale are untouched and it only reshapes each forward update (wiki D22/D26).

    The spectrum is normalised so ``eta`` is the precision on the **roughest
    frequency mode present** and smoother modes are damped up to ``cap`` times
    less. This is the "relax" convention: it makes the field-edge update *grow*
    along the frequency-smooth direction that borrows from bands where the beam
    is still open, which is the symptom issue #307 reports. Anchoring at the
    rough end (dividing by ``prec.max()``, not by ``cap``) is load-bearing:
    dividing by ``cap`` would make a white kernel return ``I/cap``, silently
    weakening ``eta`` everywhere instead of degrading to today's behaviour.

    Args:
        freq_out: ``(nband,)`` band frequencies in Hz.
        length_scale: Correlation length as a fraction of the band span, or None
            to disable the prior.
        cap: Ceiling on how far the smoothest mode may be relaxed. Bounds the
            drop in ``lambda_min(M)`` and hence the erosion of the stable gamma.

    Returns:
        ``(nband, nband)`` symmetric positive-definite matrix whose largest
        eigenvalue is exactly 1 and whose smallest is at least ``1/cap``, or
        None when the prior is disabled (no length scale, fewer than two bands,
        or every band at one frequency).

    Raises:
        ValueError: non-positive ``length_scale``, or ``cap < 1``.
    """
    if length_scale is None:
        return None
    if length_scale <= 0.0:
        raise ValueError(f"gp_length_scale must be > 0 (got {length_scale}); use None to disable the prior")
    if cap < 1.0:
        raise ValueError(f"gp_cap must be >= 1 (got {cap}); cap=1 is the uniform-eta limit")
    freq_out = np.asarray(freq_out, dtype=np.float64)
    if freq_out.size < 2 or freq_out.max() == freq_out.min():
        return None
    corr = freq_correlation(freq_out, length_scale)
    lam, evec = np.linalg.eigh(corr)
    # roundoff can push the smallest eigenvalues slightly negative; they are the
    # roughest modes and belong at the cap, so floor them rather than divide by them
    ratio = lam.max() / np.maximum(lam, lam.max() * 1e-12)
    prec = np.minimum(ratio, cap)
    prec /= prec.max()  # roughest mode present -> exactly 1 (see the docstring)
    return (evec * prec) @ evec.T


class HessianTree(object):
    """Sum-over-partitions PSF-convolution Hessian for the DataTree imager.

    Applies ``H x = (1/Σ_p wsum_p) Σ_p B_pᵀ (PSF_p ⊛ (B_p x)) + η x`` using the
    per-partition ``PSFHAT`` and ``BEAM`` precomputed in pass 2, so no gridding
    happens on the inner (minor-cycle) hot path. Generalises ``HessPSF`` to a
    sum over a band node's partition children, each with its own beam. The exact
    degrid/grid path lives in ``gridder.residual_from_partitions`` (the
    per-major-cycle gradient), not here.

    Args:
        partitions: list of per-partition dicts with ``psfhat`` ``(corr, ny_psf, nxo2)``,
            ``beam`` ``(corr, ny, nx)`` and ``wsum`` ``(corr,)``. Image-space
            arrays are (Y, X)-ordered (wiki D19); ``nx``/``ny`` keep meaning
            the X/Y pixel counts.
        nx, ny: image dimensions.
        nx_psf, ny_psf: PSF dimensions (``ny_psf`` is the real-FFT last size).
        eta: additive Tikhonov coefficient on the wsum-normalised operator.
            The deconv CLI's ``--eta`` (a fraction of the total wsum) passes
            through unscaled because the operator is normalised by the total
            wsum (``eta*wsum_tot`` in raw units; wiki D4).
        nthreads: FFT threads.
        wsum: optional normalisation override (defaults to the sum of
            per-partition ``wsum``). A ``HessTreeRay`` band actor passes the
            TOTAL wsum across all bands so the per-band operator matches the
            legacy total-normalised convention.
        eta_mode, eta_cap: optional spatially varying ``eta`` built from this
            band's own beams; see ``eta_profile``. ``eta`` is then the value at
            the beam peak / tangent point rather than everywhere.
    """

    def __init__(self, partitions, nx, ny, nx_psf, ny_psf, eta=0.0, nthreads=1, wsum=None, eta_mode=None, eta_cap=1e2):
        if not partitions:
            raise ValueError("HessianTree requires at least one partition")
        self.parts = partitions
        self.nx = nx
        self.ny = ny
        self.nx_psf = nx_psf
        self.ny_psf = ny_psf
        self.nthreads = nthreads
        self.ncorr = partitions[0]["wsum"].size
        # built here (not driver-side) so the profile follows this band's own
        # beams and never crosses Ray: the (ncorr, ny, nx) array stays in the
        # worker that owns the band
        self.eta = eta_profile(partitions, eta, eta_mode, ny, nx, cap=eta_cap)
        self.eta_mode = eta_mode
        if wsum is None:
            self.wsum = np.zeros(self.ncorr)
            for p in partitions:
                self.wsum += p["wsum"]
        else:
            # explicit normalisation (e.g. TOTAL wsum across all bands so the
            # per-band operator matches the legacy total-normalised convention)
            self.wsum = np.broadcast_to(np.asarray(wsum, dtype=float), (self.ncorr,)).copy()
        # preallocate FFT scratch so a (future) Ray actor reused across minor-cycle
        # iterations does not reallocate each dot(); safe because actor calls are
        # single-threaded (matches HessPSF)
        self.xpad = empty_noncritical((self.ny_psf, self.nx_psf), dtype="f8")
        self.xhat = empty_noncritical((self.ny_psf, self.nx_psf // 2 + 1), dtype="c16")

    def dot(self, x):
        # leading axis is the correlation axis (HessianTree acts per output image;
        # the band/time axis is distributed by Ray, not carried here)
        xtmp = x if x.ndim == 3 else x[None, :, :]
        ncorr, ny, nx = xtmp.shape
        assert ncorr == self.ncorr, f"expected {self.ncorr} correlations on axis 0, got {ncorr}"
        assert ny == self.ny and nx == self.nx
        out = np.zeros_like(xtmp)
        xpad = self.xpad
        xhat = self.xhat
        for p in self.parts:
            beam = p["beam"]
            psfhat = p["psfhat"]
            for c in range(self.ncorr):
                xpad.fill(0.0)
                xpad[0:ny, 0:nx] = xtmp[c] * beam[c]
                r2c(xpad, axes=(0, 1), nthreads=self.nthreads, forward=True, inorm=0, out=xhat)
                xhat *= psfhat[c]
                c2r(
                    xhat,
                    axes=(0, 1),
                    forward=False,
                    out=xpad,
                    # last (real-FFT) axis is x in (Y, X) order
                    lastsize=self.nx_psf,
                    inorm=2,
                    nthreads=self.nthreads,
                    allow_overwriting_input=True,
                )
                out[c] += beam[c] * xpad[0:ny, 0:nx]
        out /= self.wsum[:, None, None]
        out += self.eta * xtmp
        return out

    def hdot(self, x):
        # Hermitian operator
        return self.dot(x)


class HessTreeRay:
    """Cube-level Hessian over per-band HessianTrees held in band workers.

    Satisfies the ``LinearOperator`` Protocol on ``(nband, nx, ny)`` cubes.
    A thin facade over a ``BandWorkerPool`` (one worker per band; bands
    couple only through the prox, so Hessian applications and CG solves are
    embarrassingly parallel over bands). Pass ``workers`` to co-locate with
    the other per-band roles (Psi, exact residual) in the same worker
    processes; without it a private pool is created. ``cg`` is the
    distributed fast path sniffed by ``opt.pcg.PCG``: each band iterates its
    own CG to convergence inside its worker — one Ray dispatch per forward
    solve, not per CG iteration.

    For ``nband == 1`` the pool runs in-process (no Ray overhead).

    Args:
        partitions_per_band: list (over bands) of partition-dict lists, each
            dict with ``psfhat``/``beam``/``wsum`` as for ``HessianTree``; or
            None to build from data the workers loaded themselves via
            ``BandWorkerPool.load_bands`` (requires ``workers``).
        nx, ny, nx_psf, ny_psf: image/PSF geometry.
        etas: Tikhonov parameter, scalar or per-band sequence.
        eta_mode, eta_cap: optional spatially varying ``eta``; each worker builds
            the profile from its own band's beams (see ``eta_profile``).
        freq_prec: Optional ``(nband, nband)`` frequency precision from
            ``freq_precision``. When given, bands couple through the
            preconditioner and the forward CG runs at cube level on the driver
            instead of band-parallel inside the workers (issue #307).
        nthreads: total FFT threads (ignored when ``workers`` is passed; the
            pool's per-band thread budget applies).
        wsums: optional normalisation override, scalar or per-band sequence
            (pass the total wsum for the legacy convention).
        cg_tol, cg_maxit, cg_minit, cg_verbose: defaults for ``cg``.
        workers: optional shared ``BandWorkerPool``.
    """

    def __init__(
        self,
        partitions_per_band,
        nx,
        ny,
        nx_psf,
        ny_psf,
        etas=0.0,
        eta_mode=None,
        eta_cap=1e2,
        freq_prec=None,
        nthreads=1,
        wsums=None,
        cg_tol=1e-3,
        cg_maxit=150,
        cg_minit=1,
        cg_verbose=0,
        workers=None,
    ):
        # deferred to break the import cycle (band_worker imports HessianTree)
        # deferred: band_worker imports operators.hessian (import cycle)
        from pfb_imaging.operators.band_worker import BandWorkerPool

        if partitions_per_band is None:
            if workers is None:
                raise ValueError("partitions_per_band=None requires a workers pool with loaded bands")
            self.nband = workers.nband
        else:
            self.nband = len(partitions_per_band)
        self.nx = nx
        self.ny = ny
        self.cg_tol = cg_tol
        self.cg_maxit = cg_maxit
        self.cg_minit = cg_minit
        self.cg_verbose = cg_verbose
        etas = np.broadcast_to(np.asarray(etas, dtype=float), (self.nband,))
        if wsums is None:
            wsums = [None] * self.nband
        else:
            wsums = np.broadcast_to(np.asarray(wsums, dtype=float), (self.nband,))

        if workers is None:
            workers = BandWorkerPool(self.nband, nthreads)
        elif workers.nband != self.nband:
            raise ValueError(f"workers pool has {workers.nband} bands, expected {self.nband}")
        self._pool = workers
        self._pool.init_hess(partitions_per_band, nx, ny, nx_psf, ny_psf, etas, wsums, eta_mode, eta_cap)

        # GP prior over frequency (issue #307). Applied through the identity
        #   D^.5 Cinv D^.5 = D + D^.5 (Cinv - I) D^.5
        # so the workers keep applying their own eta/eta_mode term (the D) and
        # the driver adds only the remainder. That remainder is NEGATIVE
        # semi-definite (Cinv's eigenvalues are in (0, 1]); the total operator
        # is still symmetric positive definite, so CG applies -- but nothing may
        # assume this term alone is PSD.
        self._dC = None
        self._s = None
        if freq_prec is not None:
            freq_prec = np.asarray(freq_prec, dtype=np.float64)
            if freq_prec.shape != (self.nband, self.nband):
                raise ValueError(f"freq_prec has shape {freq_prec.shape}, expected {(self.nband, self.nband)}")
            self._dC = freq_prec - np.eye(self.nband)
            if eta_mode is None:
                # uniform eta: a per-band scalar, so no (nband, ny, nx) cube is
                # fetched or held (that is ~1 GB of f8 at 4096^2 x 8 bands)
                self._s = np.sqrt(etas)[:, None, None]
            else:
                self._s = np.sqrt(self._pool.get_eta(ny, nx))
            # reused every dot() on the CG hot path; allocating two
            # (nband, ny, nx) temporaries per application would dominate at
            # production image sizes (matches HessianTree's FFT scratch)
            self._buf = np.empty((self.nband, ny, nx))
            self._buf2 = np.empty((self.nband, ny, nx))

    def dot(self, x):
        out = self._pool.hess_dot(x)
        if self._dC is not None:
            # BLAS matmul over the band axis rather than utils.misc.freqmul,
            # which is njit(parallel=False); reshape of a C-contiguous buffer is
            # a view, so out= writes through
            np.multiply(x, self._s, out=self._buf)
            np.matmul(self._dC, self._buf.reshape(self.nband, -1), out=self._buf2.reshape(self.nband, -1))
            np.multiply(self._buf2, self._s, out=self._buf2)
            out += self._buf2
        return out

    def hdot(self, x):
        return self.dot(x)

    def cg(self, rhs, x0=None, tol=None, maxit=None, minit=None):
        """Solve ``hess @ update = rhs``.

        Without a frequency prior the solve is band-parallel: one Ray dispatch
        per band per call, each worker iterating its own CG to convergence in
        process. A frequency prior couples the bands, so the solve moves to a
        cube-level CG on the driver -- one dispatch per band per CG *iteration*.
        The FFT work is unchanged and still happens in the workers; only the
        round trips are added (the backward step already fans ``dot`` out this
        way, up to ``pd_maxit`` times per major cycle).

        Warning:
            On the coupled path ``x0`` is bound as the iterate and updated
            **in place** by ``pcg_numba``; the returned array IS ``x0``.
        """
        tol = self.cg_tol if tol is None else tol
        maxit = self.cg_maxit if maxit is None else maxit
        minit = self.cg_minit if minit is None else minit
        if self._dC is None:
            return self._pool.hess_cg(rhs, x0, tol, maxit, minit, self.cg_verbose)
        return pcg(
            self.dot,
            rhs,
            x0=x0,
            tol=tol,
            maxit=maxit,
            minit=minit,
            verbosity=self.cg_verbose,
        )

    def get_eta(self):
        """Tikhonov coefficient per band as an ``(nband, ny, nx)`` cube."""
        return self._pool.get_eta(self.ny, self.nx)

    def get_freq_prior_stats(self):
        """Spectrum of the frequency prior, or None when it is disabled.

        Returns:
            dict with ``prec_min``/``prec_max`` (the normalised precision
            spectrum, whose maximum is exactly 1 by construction) and
            ``eta_max``/``lam_min``/``lam_max`` (the same scaled by the largest
            eta, i.e. the prior's actual contribution to ``M``'s spectrum).
            ``lam_max == eta_max`` is the invariant that keeps ``lambda_max(M)``
            -- hence ``hess_norm`` and the primal-dual step sizes -- unchanged.
        """
        if self._dC is None:
            return None
        lam = np.linalg.eigvalsh(self._dC + np.eye(self.nband))
        eta_max = float(np.max(self._s)) ** 2
        return {
            "prec_min": float(lam.min()),
            "prec_max": float(lam.max()),
            "eta_max": eta_max,
            "lam_min": float(lam.min()) * eta_max,
            "lam_max": float(lam.max()) * eta_max,
        }

    def get_mem(self):
        """Per-worker post-gc memory telemetry (empty for the local path)."""
        return self._pool.get_mem()


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def hessian_slice_jax(nx, ny, nx_psf, ny_psf, eta, psfhat, x):
    psfh = jax.lax.stop_gradient(psfhat)
    xhat = jnp.fft.rfft2(x, s=(nx_psf, ny_psf), norm="backward")
    xout = jnp.fft.irfft2(xhat * psfh, s=(nx_psf, ny_psf), norm="backward")[0:nx, 0:ny]
    return xout + eta * x
