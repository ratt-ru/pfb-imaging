from functools import partial
import numpy as np
import numexpr as ne
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import r2c, c2r
from ducc0.misc import empty_noncritical
from pfb.operators.psf import (psf_convolve_slice,
                               psf_convolve_cube)
from pfb.opt.pcg import pcg
from pfb.utils.misc import taperf
from time import time

def _hessian_slice(x,
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
                   wsum=None):
    '''
    Apply vis space Hessian approximation on a slice of an image.

    Important!
    x0, y0, flip_u, flip_v and flip_w must be consistent with the
    conventions defined in pfb.operators.gridder.wgridder_conventions

    These are inputs here to allow for testing but should generally be taken
    from the attrs of the datasets produced by
    pfb.operators.gridder.image_data_products
    '''
    if not x.any():
        return np.zeros_like(x)
    nx, ny = x.shape
    mvis = dirty2vis(uvw=uvw,
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
                    divide_by_n=False)

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
                    divide_by_n=False)

    if wsum is not None:
        convim /= wsum

    if beam is not None:
        convim *= beam

    if eta is not None:
        convim += eta * x

    return convim


def _hessian_psf_slice(x,       # input image, not overwritten
                    xpad=None,  # preallocated array to store padded image
                    xhat=None,  # preallocated array to store FTd image
                    xout=None,  # preallocated array to store output image
                    abspsf=None,
                    beam=None,
                    lastsize=None,
                    nthreads=1,
                    eta=None):
    """
    Tikhonov regularised Hessian approx
    """
    nx, ny = x.shape
    xpad.fill(0.0)
    if beam is None:
        np.copyto(xpad[0:nx, 0:ny], x)
        # xpad[0:nx, 0:ny] = x
    else:
        xpad[0:nx, 0:ny] = x*beam
    r2c(xpad, axes=(0, 1), nthreads=nthreads,
        forward=True, inorm=0, out=xhat)
    # xhat *= abspsf
    ne.evaluate('xhat * abspsf',
                out=xhat,
                local_dict={
                    'xhat': xhat,
                    'abspsf': abspsf},
                casting='unsafe')
    c2r(xhat, axes=(0, 1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    np.copyto(xout, xpad[0:nx, 0:ny])

    if beam is not None:
        xout *= beam

    ne.evaluate('xout + x * eta',
                out=xout,
                local_dict={
                    'xout': xout,
                    'x': x,
                    'eta': eta
                },
                casting='unsafe')
    # xout += x * eta

    return xout


def hessian_psf_cube(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    beam,
                    abspsf,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    eta=1,
                    mode='forward'):
    """
    Tikhonov regularised Hessian approx
    """
    if mode=='forward':
        if beam is not None:
            psf_convolve_cube(xpad, xhat, xout, abspsf, lastsize, x*beam,
                            nthreads=nthreads)
        else:
            psf_convolve_cube(xpad, xhat, xout, abspsf, lastsize, x,
                            nthreads=nthreads)

        if beam is not None:
            xout *= beam

        if eta:
            xout += x * eta

        return xout
    else:
        raise NotImplementedError


def hess_direct(x,     # input image, not overwritten
                xpad=None,  # preallocated array to store padded image
                xhat=None,  # preallocated array to store FTd image
                xout=None,  # preallocated array to store output image
                abspsf=None,
                taperxy=None,
                lastsize=None,
                nthreads=1,
                eta=1,
                mode='forward'):
    nband, nx, ny = x.shape
    xpad.fill(0.0)
    xpad[:, 0:nx, 0:ny] = x * taperxy[None]
    r2c(xpad, out=xhat, axes=(1,2),
        forward=True, inorm=0, nthreads=nthreads)
    if mode=='forward':
        xhat *= (abspsf + eta)
    else:
        xhat /= (abspsf + eta)
    c2r(xhat, axes=(1, 2), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    np.copyto(xout, xpad[:, 0:nx, 0:ny])
    xout *= taperxy[None]
    return xout


def hess_direct_slice(x,     # input image, not overwritten
                xpad=None,  # preallocated array to store padded image
                xhat=None,  # preallocated array to store FTd image
                xout=None,  # preallocated array to store output image
                abspsf=None,
                taperxy=None,
                lastsize=None,
                nthreads=1,
                eta=1,
                mode='forward'):
    '''
    Note eta must be relative to wsum (peak of PSF)
    '''
    nx, ny = x.shape
    xpad.fill(0.0)
    xpad[0:nx, 0:ny] = x * taperxy
    r2c(xpad, out=xhat, axes=(0,1),
        forward=True, inorm=0, nthreads=nthreads)
    if mode=='forward':
        xhat *= (abspsf + eta)
    else:
        xhat /= (abspsf + eta)
    c2r(xhat, axes=(0, 1), forward=False, out=xpad,
        lastsize=lastsize, inorm=2, nthreads=nthreads,
        allow_overwriting_input=True)
    np.copyto(xout, xpad[0:nx, 0:ny])
    xout *= taperxy
    return xout


class hess_psf(object):
    def __init__(self, nx, ny, abspsf,
                 beam=None,
                 eta=1.0,
                 nthreads=1,
                 cgtol=1e-3,
                 cgmaxit=300,
                 cgverbose=2,
                 cgrf=25,
                 taper_width=32,
                 min_beam=5e-3):
        self.nx = nx
        self.ny = ny
        self.abspsf = abspsf
        self.nband, self.nx_psf, self.nyo2 = abspsf.shape
        if beam is not None and not (beam==1).all():
            assert self.nband == beam.shape[0]
            assert self.nx == beam.shape[1]
            assert self.ny == beam.shape[2]
            self.beam = beam
        else:
            # self.beam = None
            self.beam = (None,)*self.nband
        self.ny_psf = 2*(self.nyo2-1)
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
        self.xhat = empty_noncritical((self.nx_psf, self.nyo2),
                                      dtype='c16')
        self.xpad = empty_noncritical((self.nx_psf, self.ny_psf),
                                      dtype='f8')
        self.xout = empty_noncritical((self.nband, self.nx, self.ny),
                                      dtype='f8')

        # conjugate gradient params
        self.cgtol=cgtol
        self.cgmaxit=cgmaxit
        self.cgverbose=cgverbose
        self.cgrf=cgrf

        # taper for direct mode
        self.taperxy = taperf((nx, ny), taper_width)

        # for beam application in direct mode
        self.min_beam = min_beam

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

        # tii = time()
        for b in range(nband):
            self.xpad.fill(0.0)
            if self.beam[b] is None:
                np.copyto(self.xpad[0:nx, 0:ny], xtmp[b])
            else:
                self.xpad[0:nx, 0:ny] = xtmp[b]*self.beam[b]
            r2c(self.xpad, axes=(0, 1), nthreads=self.nthreads,
                forward=True, inorm=0, out=self.xhat)
            ne.evaluate('xhat * abspsf',
                        out=self.xhat,
                        local_dict={
                            'xhat': self.xhat,
                            'abspsf': self.abspsf[b]},
                        casting='unsafe')
            c2r(self.xhat, axes=(0, 1), forward=False, out=self.xpad,
                lastsize=self.ny_psf, inorm=2, nthreads=self.nthreads,
                allow_overwriting_input=True)
            if self.beam[b] is None:
                np.copyto(self.xout[b], self.xpad[0:nx, 0:ny])
            else:
                self.xout[b] = self.xpad[0:nx, 0:ny]*self.beam[b]
        ne.evaluate('xout + xtmp * eta',
                    out=self.xout,
                    local_dict={
                        'xout': self.xout,
                        'xtmp': xtmp,
                        'eta': self.eta[:, None, None]},
                    casting='unsafe')
        # print('ttot = ', time() - tii)
        return self.xout

    # def dot(self, x):
    #     if len(x.shape) == 3:
    #         xtmp = x
    #     elif len(x.shape) == 2:
    #         xtmp = x[None, :, :]
    #     else:
    #         raise ValueError("Unsupported number of input dimensions")

    #     nband, nx, ny = xtmp.shape
    #     assert nband == self.nband
    #     assert nx == self.nx
    #     assert ny == self.ny

    #     tii = time()
    #     ti = time()
    #     self.xpad.fill(0.0)
    #     tfill = time() - ti
    #     ti = time()
    #     if self.beam is None:
    #         np.copyto(self.xpad[:, 0:nx, 0:ny], xtmp)
    #     else:
    #         self.xpad[:, 0:nx, 0:ny] = xtmp*self.beam
    #     tpad = time() - ti
    #     ti = time()
    #     r2c(self.xpad, axes=(1, 2), nthreads=self.nthreads,
    #         forward=True, inorm=0, out=self.xhat)
    #     tr2c = time() - ti
    #     ti = time()
    #     ne.evaluate('xhat * abspsf',
    #                 out=self.xhat,
    #                 local_dict={
    #                     'xhat': self.xhat,
    #                     'abspsf': self.abspsf},
    #                 casting='unsafe')
    #     tconv = time() - ti
    #     ti = time()
    #     c2r(self.xhat, axes=(1, 2), forward=False, out=self.xpad,
    #         lastsize=self.ny_psf, inorm=2, nthreads=self.nthreads,
    #         allow_overwriting_input=True)
    #     tc2r = time() - ti
    #     ti = time()
    #     if self.beam is None:
    #         np.copyto(self.xout, self.xpad[:, 0:nx, 0:ny])
    #     else:
    #         self.xout = self.xpad[:, 0:nx, 0:ny]*self.beam
    #     tcopy = time() - ti
    #     ti = time()
    #     ne.evaluate('xout + xtmp * eta',
    #                 out=self.xout,
    #                 local_dict={
    #                     'xout': self.xout,
    #                     'xtmp': xtmp,
    #                     'eta': self.eta[:, None, None]},
    #                 casting='unsafe')
    #     tplus = time() - ti
    #     ttot = time() - tii
    #     tfill /= ttot
    #     tpad /= ttot
    #     tr2c /= ttot
    #     tconv /= ttot
    #     tc2r /= ttot
    #     tcopy /= ttot
    #     tplus /= ttot
    #     # print('tfill = ', tfill)
    #     # print('tpad = ', tpad)
    #     # print('tr2c = ', tr2c)
    #     # print('tconv = ', tconv)
    #     # print('tc2r = ', tc2r)
    #     # print('tcopy = ', tcopy)
    #     # print('tplus = ', tplus)
    #     print('ttot = ', time() - tii)
    #     return self.xout

    def hdot(self, x):
        # Hermitian operator
        return self.dot(x)

    def idot(self, x, mode='psf', x0=None):
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
            # initialise with direct estimate
            x0 = np.zeros_like(xtmp)
            for b in range(self.nband):
                x0[b] = hess_direct_slice(xtmp[b],
                                    xpad=self.xpad,
                                    xhat=self.xhat,
                                    xout=self.xout[b],
                                    abspsf=self.abspsf[b],
                                    taperxy=self.taperxy,
                                    lastsize=self.ny_psf,
                                    nthreads=self.nthreads,
                                    eta=self.eta[b]*np.sqrt(nx*ny),
                                    mode='backward')
                if self.beam[b] is not None:
                    mask = (self.xout[b] > 0) & (self.beam[b] > self.min_beam)
                    self.xout[b, mask] /= self.beam[b, mask]**2

        if mode=='direct':
            for b in range(self.nband):
                self.xout[b] = hess_direct_slice(xtmp[b],
                                    xpad=self.xpad,
                                    xhat=self.xhat,
                                    xout=self.xout[b],
                                    abspsf=self.abspsf[b],
                                    taperxy=self.taperxy,
                                    lastsize=self.ny_psf,
                                    nthreads=self.nthreads,
                                    eta=self.eta[b]*np.sqrt(nx*ny),
                                    mode='backward')
                if self.beam[b] is not None:
                    mask = (self.xout[b] > 0) & (self.beam[b] > self.min_beam)
                    self.xout[b, mask] /= self.beam[b, mask]**2

        elif mode=='psf':
            for b in range(self.nband):
                hess = partial(_hessian_psf_slice,
                            xpad=self.xpad,
                            xhat=self.xhat,
                            xout=self.xout[b],
                            abspsf=self.abspsf[b],
                            beam=self.beam[b],
                            lastsize=self.ny_psf,
                            nthreads=self.nthreads,
                            eta=self.eta[b])
                self.xout[b] = pcg(hess,
                            xtmp[b],
                            x0=x0[b],
                            tol=self.cgtol,
                            maxit=self.cgmaxit,
                            minit=3,
                            verbosity=self.cgverbose,
                            report_freq=self.cgrf,
                            backtrack=False,
                            return_resid=False)
        else:
            raise ValueError(f"Unknown mode {mode}")

        return self.xout.copy()


##################### jax #####################################
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,1,2,3,4))
def hessian_slice_jax(
                    nx, ny,
                    nx_psf, ny_psf,
                    eta,
                    psfhat,
                    x):
    psfh = jax.lax.stop_gradient(psfhat)
    xhat = jnp.fft.rfft2(x,
                         s=(nx_psf, ny_psf),
                         norm='backward')
    xout = jnp.fft.irfft2(xhat*psfh,
                          s=(nx_psf, ny_psf),
                          norm='backward')[0:nx, 0:ny]
    return xout + eta*x

@partial(jax.jit, static_argnums=(0,1,2,3,4))
def hessian_jax(nx, ny,
                nx_psf, ny_psf,
                eta,
                psfhat,
                x):
    psfh = jax.lax.stop_gradient(psfhat)
    xhat = jnp.fft.rfft2(x,
                         s=(nx_psf, ny_psf),
                         norm='backward')
    xout = jnp.fft.irfft2(xhat*psfh,
                          s=(nx_psf, ny_psf),
                          norm='backward')[:, 0:nx, 0:ny]
    return xout + eta*x