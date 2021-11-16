
import numpy as np
import dask
import dask.array as da
from daskms.optimisation import inlined_array
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r, c2c, good_size
from pfb.operators.psi import im2coef, coef2im
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def hessian_wgt_xds(x, xdss, hessopts, wsum, sigmainv, compute=True):
    '''
    Vis space Hessian reduction over dataset
    '''

    convims = []
    for xds in xdss:
        wgt = xds.WEIGHT.data
        uvw = xds.UVW.data
        freq = xds.FREQ.data
        fbin_idx = xds.FBIN_IDX.data
        fbin_counts = xds.FBIN_COUNTS.data
        beam = xds.BEAM.data

        convim = hessian_wgt(uvw, wgt, freq, x, beam, fbin_idx,
                             fbin_counts, hessopts)
        convim = inlined_array(convim, uvw)

        convims.append(convim)

    # LB - it's not this simple when there are multiple spw's to consider
    convim = da.stack(convims).sum(axis=0)/wsum

    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim

def hessian_wgt_alpha_xds(alpha, xdss, waveopts, hessopts, wsum,
                          sigmainv, compute=True):
    '''
    Vis space Hessian reduction over dataset
    '''
    pmask = waveopts['pmask']
    bases = waveopts['bases']
    padding = waveopts['padding']
    iy = waveopts['iy']
    sy = waveopts['sy']
    nx = waveopts['nx']
    ny = waveopts['ny']
    ntot = waveopts['ntot']
    nmax = waveopts['nmax']
    nlevels = waveopts['nlevels']

    alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)

    # coeff to image
    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)
    x = inlined_array(x, [pmask, alpha, bases, padding])

    convims = []
    for xds in xdss:
        wgt = xds.WEIGHT.data
        uvw = xds.UVW.data
        freq = xds.FREQ.data
        fbin_idx = xds.FBIN_IDX.data
        fbin_counts = xds.FBIN_COUNTS.data
        beam = xds.BEAM.data

        convim = hessian_wgt(uvw, wgt, freq, x, beam, fbin_idx,
                             fbin_counts, hessopts)
        convim = inlined_array(convim, uvw)

        convims.append(convim)

    # LB - it's not this simple when there are multiple spw's mapping to
    # different imaging bands
    convim = da.stack(convims).sum(axis=0)/wsum


    alpha_rec = im2coef(convim, pmask, bases, ntot, nmax, nlevels)
    alpha_rec = inlined_array(alpha_rec, [pmask, bases, ntot])

    if sigmainv:
        alpha_rec += alpha * sigmainv**2

    if compute:
        return alpha_rec.compute()
    else:
        return alpha_rec

def _hessian_wgt_impl(uvw, weight, freq, x, beam, fbin_idx, fbin_counts,
                      cell=None,
                      wstack=None,
                      epsilon=None,
                      double_accum=None,
                      nthreads=None):
    """
    Tikhonov regularised Hessian of coeffs
    """
    nband, nx, ny = beam.shape
    convim = np.zeros((nband, nx, ny), dtype=x.dtype)
    fbin_idx2 = fbin_idx - fbin_idx.min()  # adjust for freq chunking
    for b in range(nband):
        flow = fbin_idx2[b]
        fhigh = fbin_idx2[b] + fbin_counts[b]

        mvis = dirty2ms(uvw=uvw,
                        freq=freq[flow:fhigh],
                        dirty=beam[b] * x[b],
                        wgt=weight[:, flow:fhigh],
                        pixsize_x=cell,
                        pixsize_y=cell,
                        epsilon=epsilon,
                        nthreads=nthreads,
                        do_wstacking=wstack)

        convim[b] = ms2dirty(uvw=uvw,
                             freq=freq[flow:fhigh],
                             ms=mvis,
                             wgt=weight[:, flow:fhigh],
                             npix_x=nx,
                             npix_y=ny,
                             pixsize_x=cell,
                             pixsize_y=cell,
                             epsilon=epsilon,
                             nthreads=nthreads,
                             do_wstacking=wstack,
                             double_precision_accumulation=double_accum)

        convim[b] *= beam[b]

    return convim


def _hessian_wgt(uvw, weight, freq, x, beam, fbin_idx, fbin_counts, hessopts):
    return _hessian_wgt_impl(uvw[0][0], weight[0], freq, x, beam,
                             fbin_idx, fbin_counts, **hessopts)

def hessian_wgt(uvw, weight, freq, x, beam, fbin_idx, fbin_counts, hessopts):

    return da.blockwise(_hessian_wgt, ('chan', 'nx', 'ny')
,                       uvw, ("row", 'three'),
                        weight, ('row', 'chan'),
                        freq, ('chan',),
                        x, ('chan', 'nx', 'ny'),
                        beam, ('chan', 'nx', 'ny'),
                        fbin_idx, ('chan',),
                        fbin_counts, ('chan',),
                        hessopts, None,
                        align_arrays=False,
                        adjust_chunks={'chan': x.chunks[0]},
                        dtype=x.dtype)


def hessian_psf_xds(x, xdss, hessopts, wsum, sigmainv, compute=True):
    '''
    Vis space Hessian reduction over dataset
    '''

    convims = []
    for xds in xdss:
        psf = xds.PSF.data
        beam = xds.BEAM.data

        convim = hessian_psf(uvw, wgt, freq, x, beam, fbin_idx,
                             fbin_counts, hessopts)

        convims.append(convim)

    # LB - it's not this simple when there are multiple spw's to consider
    convim = da.stack(convims).sum(axis=0)/wsum

    if sigmainv:
        convim += x * sigmainv**2

    if compute:
        return convim.compute()
    else:
        return convim

def _hessian_psf_impl(x, beam, psfhat,
                      nthreads=None,
                      padding_psf=None,
                      unpad_x=None,
                      unpad_y=None,
                      lastsize=None):
    """
    Tikhonov regularised Hessian of coeffs
    """
    nband, nx, ny = x.shape
    convim = np.zeros_like(x)
    for b in range(nband):
        xhat = iFs(np.pad(beam[l]*x[l], padding_psf, mode='constant'), axes=(0, 1))
        xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                    forward=True, inorm=0)
        xhat = c2r(xhat * psfhat[l], axes=(0, 1), forward=False,
                    lastsize=lastsize, inorm=2, nthreads=nthreads)
        convim[l] = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

    return convim

def _hessian_psf(x, beam, psfhat, hessopts):
    return _hessian_psf_impl(x, beam, psfhat, **hessopts)

def hessian_psf(x, beam, psfhat, hessopts):
    return da.blockwise(_hessian_psf_impl, ('nband', 'nx', 'ny'),
                        x, ('nband', 'nx', 'ny'),
                        beam, ('nband', 'nx', 'ny'),
                        psfhat, ('nband', 'nx', 'ny'),
                        hessopts, None,
                        align_arrays=False,
                        dtype=x.dtype)
