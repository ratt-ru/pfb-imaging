import numpy as np
import dask
import dask.array as da
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
                  do_wgridding=None,
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
                    do_wgridding=do_wgridding,
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
                      do_wgridding=do_wgridding,
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

    # if sigmainv:
    #     xout += x * sigmainv

    return xout + x * sigmainv


class hessian_psf_slice(object):
    def __init__(self, ds, nthreads, wsum):
        self.nthreads = nthreads
        self.lastsize = ds.PSF.shape[-1]
        self.psfhat = ds.PSFHAT.values
        self.beam = ds.BEAM.values
        self.wsumb = ds.WSUM.values[0]
        self.wsum = wsum

        # pre-allocate tmp arrays
        tmp = np.empty(ds.DIRTY.shape,
                       dtype=ds.DIRTY.dtype, order='C')
        self.xout = make_noncritical(tmp)
        tmp = np.empty(self.psfhat.shape,
                       dtype=self.psfhat.dtype,
                       order='C')
        self.xhat = make_noncritical(tmp)
        tmp = np.empty(ds.PSF.shape,
                       dtype=ds.PSF.dtype,
                       order='C')
        self.xpad = make_noncritical(tmp)

    def __call__(self, x, sigmainv):
        return _hessian_psf_slice(self.xpad,
                                  self.xhat,
                                  self.xout,
                                  self.psfhat,
                                  self.beam,
                                  self.lastsize,
                                  x,
                                  nthreads=self.nthreads,
                                  sigmainv=sigmainv,
                                  wsum=self.wsum)


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


def hess_vis(xds,
             dds,
             xout,
             x,
             sigmainv=1.0,
             do_wgridding=True,
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
                              do_wgridding=do_wgridding,
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
                                                do_wgridding=do_wgridding,
                                                nthreads=nthreads,
                                                divide_by_n=divide_by_n)
            xout[field][f't{t}b{b}'] += sigmainv * x[field][f't{t}b{b}']
    return xout
