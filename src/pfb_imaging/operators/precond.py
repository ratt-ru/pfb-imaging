"""
Preconditioning operators for the PFB imaging problem.
"""

import numpy as np
from ducc0.fft import c2r, r2c
from ducc0.misc import empty_noncritical

from pfb_imaging.opt.pcg import pcg_numba as pcg


class HessPSF(object):
    """
    Preconditioning operator that approximates the Hessian as a convolution with the PSF.

    Methods:
    - dot: applies the preconditioner to a vector
    - hdot: applies the adjoint of the preconditioner to a vector (same as dot for Hermitian operators)
    - idot: applies the inverse of the preconditioner
    """

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
        memory_greedy=True,
    ):
        if not memory_greedy:
            raise NotImplementedError("Non-memory-greedy mode is not implemented yet")
        self.nx = nx
        self.ny = ny
        self.abspsf = abspsf
        self.nband, self.nx_psf, self.nyo2 = abspsf.shape
        if beam is not None:
            assert self.nband == beam.shape[0]
            assert self.nx == beam.shape[1]
            assert self.ny == beam.shape[2]
            self.beam = beam
        else:
            raise ValueError("Beam is required for HessPSF preconditioner")

        self.ny_psf = 2 * (self.nyo2 - 1)
        self.nx_pad = self.nx_psf - self.nx
        self.ny_pad = self.ny_psf - self.ny
        self.nthreads = nthreads
        if isinstance(eta, float):  # same everywhere
            self.eta = np.tile(eta, self.nband)[:, None, None]
        elif isinstance(eta, np.ndarray):
            if eta.size == self.nband:  # per band
                self.eta = eta[:, None, None]
            else:
                assert eta.shape == (self.nband, self.nx, self.ny)  # per pixel
        else:
            raise ValueError("Unsupported type for eta")

        # TODO - memory conservative version
        self.xhat = empty_noncritical((self.nband, self.nx_psf, self.nyo2), dtype="c16")
        self.xpad = empty_noncritical((self.nband, self.nx_psf, self.ny_psf), dtype="f8")
        self.xout = empty_noncritical((self.nband, self.nx, self.ny), dtype="f8")

        # conjugate gradient params
        self.cgtol = cgtol
        self.cgmaxit = cgmaxit
        self.cgverbose = cgverbose
        self.cgrf = cgrf
        self.memory_greedy = memory_greedy

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

        self.xpad.fill(0.0)
        self.xpad[:, 0:nx, 0:ny] = xtmp * self.beam
        r2c(
            self.xpad,
            axes=(-2, -1),
            nthreads=self.nthreads,
            forward=True,
            inorm=0,
            out=self.xhat,
        )
        self.xhat *= self.abspsf
        c2r(
            self.xhat,
            axes=(-2, -1),
            forward=False,
            out=self.xpad,
            lastsize=self.ny_psf,
            inorm=2,
            nthreads=self.nthreads,
            allow_overwriting_input=True,
        )
        np.copyto(self.xout, self.xpad[:, 0:nx, 0:ny])
        self.xout *= self.beam
        self.xout += xtmp * self.eta
        return self.xout

    def hdot(self, x):
        # Hermitian operator
        return self.dot(x)

    def idot(self, x, x0=None):
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

        if x0 is None:
            x0 = np.zeros_like(xtmp)

        self.xout[...] = pcg(
            self.dot,
            xtmp,
            x0=x0,
            tol=self.cgtol,
            maxit=self.cgmaxit,
            minit=2,
            verbosity=self.cgverbose,
            report_freq=self.cgrf,
            backtrack=False,
            return_resid=False,
        )

        return self.xout.copy()
