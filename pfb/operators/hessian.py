
import numpy as np
import dask
import dask.array as da
from daskms.optimisation import inlined_array
from ducc0.wgridder import ms2dirty, dirty2ms
from pfb.operators.psi import im2coef, coef2im

def hessian_xds(x, xds, hessopts, wsum, sigmainv, mask,
                compute=True, use_beam=True):
    '''
    Vis space Hessian reduction over dataset
    '''
    if not isinstance(x, da.Array):
        x = da.from_array(x, chunks=(1, -1, -1), name=False)

    if not isinstance(mask, da.Array):
        mask = da.from_array(mask, chunks=(1, -1, -1), name=False)

    convims = []
    for ds in xds:
        wgt = ds.WEIGHT.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        fbin_idx = ds.FBIN_IDX.data
        fbin_counts = ds.FBIN_COUNTS.data
        if use_beam:
            beam = ds.BEAM.data * mask
        else:
            # TODO - separate implementation without
            # unnecessary beam application
            beam = mask

        convim = hessian(uvw, wgt, freq, x, beam, fbin_idx,
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

def hessian_alpha_xds(alpha, xds, waveopts, hessopts, wsum,
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

    if not isinstance(alpha, da.Array):
        alpha = da.from_array(alpha, chunks=(1, -1, -1), name=False)

    # coeff to image
    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)
    x = inlined_array(x, [pmask, alpha, bases, padding])

    convims = []
    for ds in xds:
        wgt = ds.WEIGHT.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        fbin_idx = ds.FBIN_IDX.data
        fbin_counts = ds.FBIN_COUNTS.data
        beam = ds.BEAM.data

        convim = hessian(uvw, wgt, freq, x, beam, fbin_idx,
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

def _hessian_impl(uvw, weight, freq, x, beam, fbin_idx, fbin_counts,
                  cell=None,
                  wstack=None,
                  epsilon=None,
                  double_accum=None,
                  nthreads=None):
    nband, nx, ny = beam.shape
    convim = np.zeros((nband, nx, ny), dtype=x.dtype)  # no row chunking
    fbin_idx2 = fbin_idx - fbin_idx.min()  # adjust for freq chunking
    for b in range(nband):
        flow = fbin_idx2[b]
        fhigh = fbin_idx2[b] + fbin_counts[b]

        mvis = dirty2ms(uvw=uvw,
                        freq=freq[flow:fhigh],
                        dirty=[x[b] if beam is None else x[b] * beam[b]],
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

        if beam is not None:
            convim[b] *= beam[b]

    return convim


def _hessian(uvw, weight, freq, x, beam, fbin_idx, fbin_counts, hessopts):
    return _hessian_impl(uvw[0][0], weight[0], freq, x, beam,
                             fbin_idx, fbin_counts, **hessopts)

def hessian(uvw, weight, freq, x, beam, fbin_idx, fbin_counts, hessopts):
    if beam is None:
        bout = None
    else:
        bout = ('chan', 'nx', 'ny')
    return da.blockwise(_hessian, ('chan', 'nx', 'ny'),
                        uvw, ("row", 'three'),
                        weight, ('row', 'chan'),
                        freq, ('chan',),
                        x, ('chan', 'nx', 'ny'),
                        beam, bout,
                        fbin_idx, ('chan',),
                        fbin_counts, ('chan',),
                        hessopts, None,
                        align_arrays=False,
                        adjust_chunks={'chan': x.chunks[0]},
                        dtype=x.dtype)
