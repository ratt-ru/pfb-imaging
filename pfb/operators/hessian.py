import numpy as np
import dask
import dask.array as da
from daskms.optimisation import inlined_array
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.misc import make_noncritical
from uuid import uuid4
from pfb.operators.psf import (psf_convolve_slice,
                               psf_convolve_cube)


def hessian_xds(x, xds, hessopts, wsum, sigmainv, mask,
                compute=True, use_beam=True):
    '''
    Vis space Hessian reduction over dataset.
    Hessian will be applied to x
    '''
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1),
                          name="x-" + uuid4().hex)

    if not isinstance(mask, da.Array):
        mask = da.from_array(mask, chunks=(-1, -1),
                             name="mask-" + uuid4().hex)

    assert mask.ndim == 2

    nband, nx, ny = x.shape

    # LB - what is the point of specifying name?
    convims = [da.zeros((nx, ny),
               chunks=(-1, -1), name="zeros-" + uuid4().hex)
               for _ in range(nband)]

    for ds in xds:
        wgt = ds.WEIGHT.data
        vis_mask = ds.MASK.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        b = ds.bandid
        if use_beam:
            beam = ds.BEAM.data * mask
        else:
            # TODO - separate implementation without
            # unnecessary beam application
            beam = mask

        convim = hessian(x[b], uvw, wgt, vis_mask, freq, beam, hessopts)

        # convim = inlined_array(convim, uvw)

        convims[b] += convim

    convim = da.stack(convims)/wsum

    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim


def _hessian_impl(x, uvw, weight, vis_mask, freq, beam,
                  x0=0.0,
                  y0=0.0,
                  cell=None,
                  wstack=None,
                  epsilon=None,
                  double_accum=None,
                  nthreads=None):
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
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wgridding=wstack,
                    divide_by_n=False)

    convim = vis2dirty(uvw=uvw,
                      freq=freq,
                      vis=mvis,
                      wgt=weight,
                      mask=vis_mask,
                      npix_x=nx,
                      npix_y=ny,
                      pixsize_x=cell,
                      pixsize_y=cell,
                      center_x=x0,
                      center_y=y0,
                      epsilon=epsilon,
                      nthreads=nthreads,
                      do_wgridding=wstack,
                      double_precision_accumulation=double_accum,
                      divide_by_n=False)

    if beam is not None:
        convim *= beam

    return convim


def _hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
    return _hessian_impl(x, uvw[0][0], weight[0][0], vis_mask[0][0], freq[0],
                         beam, **hessopts)

def hessian(x, uvw, weight, vis_mask, freq, beam, hessopts):
    if beam is None:
        bout = None
    else:
        bout = ('nx', 'ny')
    return da.blockwise(_hessian, ('nx', 'ny'),
                        x, ('nx', 'ny'),
                        uvw, ('row', 'three'),
                        weight, ('row', 'chan'),
                        vis_mask, ('row', 'chan'),
                        freq, ('chan',),
                        beam, bout,
                        hessopts, None,
                        dtype=x.dtype)


def _hessian_psf_slice(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    psfhat,
                    beam,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=None):
    """
    Tikhonov regularised Hessian approx
    """
    if beam is not None:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat, lastsize, x*beam,
                           nthreads=nthreads)
    else:
        psf_convolve_slice(xpad, xhat, xout,
                           psfhat, lastsize, x,
                           nthreads=nthreads)

    if beam is not None:
        xout *= beam

    if wsum is not None:
        xout /= wsum

    return xout + x * sigmainv

from pfb.operators.hessian import _hessian_impl
class hessian_psf_slice(object):
    def __init__(self, ds, nbasis, nmax, nthreads, sigmainv, cell, wstack, epsilon, double_accum):
        self.nthreads = nthreads
        self.sigmainv = sigmainv
        self.cell = cell
        self.wstack = wstack
        self.epsilon = epsilon
        self.double_accum = double_accum
        self.lastsize = ds.PSF.shape[-1]
        self.bandid = ds.bandid
        tmp = np.require(ds.DIRTY.values,
                         dtype=ds.DIRTY.dtype,
                         requirements='CAW')
        self.dirty = make_noncritical(tmp)
        tmp = np.require(ds.PSFHAT.values,
                         dtype=ds.PSFHAT.dtype,
                         requirements='CAW')
        self.psfhat = make_noncritical(tmp)
        tmp = np.require(ds.PSF.values,
                         dtype=ds.PSF.dtype,
                         requirements='CAW')
        self.psf = make_noncritical(tmp)
        tmp = np.require(ds.BEAM.values,
                         dtype=ds.BEAM.dtype,
                         requirements='CAW')
        self.beam = make_noncritical(tmp)
        self.wsumb = ds.WSUM.values[0]
        if 'MODEL' in ds:
            tmp = np.require(ds.MODEL.values,
                             dtype=ds.MODEL.dtype,
                             requirements='CAW')
        else:
            tmp = np.zeros_like(self.dirty)
        self.model = make_noncritical(tmp)
        if 'DUAL' in ds:
            tmp = np.require(ds.DUAL.values,
                             dtype=ds.DUAL.dtype,
                             requirements='CAW')
            assert tmp.shape == (nbasis, nmax)
        else:
            tmp = np.zeros((nbasis, nmax), dtype=self.dirty.dtype)
        self.dual = make_noncritical(tmp)
        if 'RESIDUAL' in ds:
            tmp = np.require(ds.RESIDUAL.values,
                             dtype=ds.RESIDUAL.dtype,
                             requirements='CAW')
        else:
            tmp = self.dirty.copy()
        self.residual = make_noncritical(tmp)

        self.uvw = ds.UVW.values
        self.wgt = ds.WEIGHT.values
        self.vmask = ds.VIS_MASK.values
        self.freq = ds.FREQ.values

        # pre-allocate tmp arrays
        tmp = np.empty(self.dirty.shape, dtype=self.dirty.dtype, order='C')
        self.xout = make_noncritical(tmp)
        tmp = np.empty(self.psfhat.shape, dtype=self.psfhat.dtype, order='C')
        self.xhat = make_noncritical(tmp)
        tmp = np.empty(self.psf.shape, dtype=self.psf.dtype, order='C')
        self.xpad = make_noncritical(tmp)

    def __call__(self, x):
        return _hessian_psf_slice(self.xpad,
                                  self.xhat,
                                  self.xout,
                                  self.psfhat,
                                  self.beam,
                                  self.lastsize,
                                  x,
                                  nthreads=self.nthreads,
                                  sigmainv=self.sigmainv,
                                  wsum=self.wsum)

    def compute_residual(self, x):
        return self.dirty - _hessian_impl(self.beam * x,
                                        self.uvw,
                                        self.wgt,
                                        self.vmask,
                                        self.freq,
                                        None,
                                        cell=self.cell,
                                        wstack=self.wstack,
                                        epsilon=self.epsilon,
                                        double_accum=self.double_accum,
                                        nthreads=self.nthreads)


    def set_wsum(self, wsum):
        self.wsum = wsum


def hessian_psf_cube(
                    xpad,  # preallocated array to store padded image
                    xhat,  # preallocated array to store FTd image
                    xout,  # preallocated array to store output image
                    beam,
                    psfhat,
                    lastsize,
                    x,     # input image, not overwritten
                    nthreads=1,
                    sigmainv=1,
                    wsum=None):
    """
    Tikhonov regularised Hessian approx
    """
    if beam is not None:
        psf_convolve_cube(xpad, xhat, xout, psfhat, lastsize, x*beam,
                          nthreads=nthreads)
    else:
        psf_convolve_cube(xpad, xhat, xout, psfhat, lastsize, x,
                          nthreads=nthreads)

    if beam is not None:
        xout *= beam

    if wsum is not None:
        xout /= wsum

    return xout + x * sigmainv


# from ducc0.fft import r2c, c2r
# def hess_psf(xpad,    # preallocated array to store padded image
#              xhat,    # preallocated array to store FTd image
#              xout,    # preallocated array to store output image
#              psfhat,
#              lastsize,
#              x,       # input image, not overwritten
#              nthreads=1,
#              sigmainv=1.0):
#     _, _, nx, ny = x.shape
#     xpad[...] = 0.0
#     xpad[:, :, 0:nx, 0:ny] = x
#     r2c(xpad, axes=(-2, -1), nthreads=nthreads,
#         forward=True, inorm=0, out=xhat)
#     xhat *= psfhat
#     c2r(xhat, axes=(-2, -1), forward=False, out=xpad,
#         lastsize=lastsize, inorm=2, nthreads=nthreads,
#         allow_overwriting_input=True)
#     xout[...] = xpad[:, :, 0:nx, 0:ny]
#     return xout + sigmainv*x


def hess_vis(xds,
             dds,
             xout,
             x,
             sigmainv=1.0,
             wstack=True,
             nthreads=1,
             epsilon=1e-7,
             divide_by_n=False):
    for ds in xds:
        b = ds.bandid
        t = ds.timeid
        vis_mask = ds.MASK.values
        if np.all(vis_mask == 0):
            continue

        # accumulate model vis for this band and time
        mvis = np.zeros(ds.VIS.data.shape, dtype=ds.VIS.dtype)
        for field in dds.keys():

            x0 = dds[field][f't{t}b{b}']['x0']
            y0 = dds[field][f't{t}b{b}']['y0']
            cell = dds[field][f't{t}b{b}']['cell']
            nx = dds[field][f't{t}b{b}']['nx']
            ny = dds[field][f't{t}b{b}']['ny']

            mvis += dirty2vis(uvw=ds.UVW.values,
                              freq=ds.FREQ.values,
                              dirty=x[field][f't{t}b{b}'],
                              pixsize_x=cell,
                              pixsize_y=cell,
                              center_x=x0,
                              center_y=y0,
                              epsilon=epsilon,
                              do_wgridding=wstack,
                              nthreads=nthreads,
                              divide_by_n=divide_by_n)

        # project to image space
        for field in dds.keys():
            x0 = dds[field][f't{t}b{b}']['x0']
            y0 = dds[field][f't{t}b{b}']['y0']
            cell = dds[field][f't{t}b{b}']['cell']
            nx = dds[field][f't{t}b{b}']['nx']
            ny = dds[field][f't{t}b{b}']['ny']
            xout[field][f't{t}b{b}'] = vis2dirty(uvw=ds.UVW.values,
                                                freq=ds.FREQ.values,
                                                vis=mvis,
                                                wgt=ds.WEIGHT.values,
                                                npix_x=nx,
                                                npix_y=ny,
                                                pixsize_x=cell,
                                                pixsize_y=cell,
                                                center_x=x0,
                                                center_y=y0,
                                                epsilon=epsilon,
                                                do_wgridding=wstack,
                                                nthreads=nthreads,
                                                divide_by_n=divide_by_n)
            xout[field][f't{t}b{b}'] += sigmainv * x[field][f't{t}b{b}']
    return xout
