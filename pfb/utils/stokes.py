import numpy as np
import numexpr as ne
import dask
import dask.array as da
from xarray import Dataset
from africanus.gridding.wgridder.dask import dirty as vis2im
from africanus.calibration.utils.dask import corrupt_vis
from pfb.utils.weighting import compute_wsum
from daskms.optimisation import inlined_array

def single_stokes(data=None,
                  weight=None,
                  imaging_weight=None,
                  ant1=None,
                  ant2=None,
                  jones=None,
                  flag=None,
                  frow=None,
                  uvw=None,
                  time=None,
                  fid=None,
                  ddid=None,
                  scanid=None,
                  row_out_chunk=None,
                  nthreads=None,
                  epsilon=None,
                  wstack=None,
                  double_accum=None,
                  flipv=None,
                  freq=None,
                  fbin_idx=None,
                  fbin_counts=None,
                  band_mapping=None,
                  freq_out=None,
                  tbin_idx=None,
                  tbin_counts=None,
                  nx=None,
                  ny=None,
                  nx_psf=None,
                  ny_psf=None,
                  cell_rad=None,
                  radec=None,
                  idx0=None,
                  idxf=None,
                  sign=None,
                  csign=None,
                  do_dirty=True,
                  do_psf=True,
                  do_weights=True,
                  check_wsum=False):

    data_type = data.dtype
    data_shape = data.shape
    data_chunks = data.chunks
    real_type = data.real.dtype

    # compute the mueller term
    nrow, nchan, ncorr = data.shape
    if ncorr > 2:
        acol = da.ones((nrow, nchan, 1, 2, 2),
                       chunks=(data_chunks[0], data_chunks[1], 1, 2, 2),
                       dtype=data_type)
    else:
        acol = da.ones((nrow, nchan, 1, ncorr),
                       chunks=(data_chunks[0], data_chunks[1], 1, ncorr),
                       dtype=data_type)

    jones = da.swapaxes(jones, 1, 2)
    mueller = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                          jones, acol).reshape(nrow, nchan, ncorr)

    dw = weight_data(data, weight, imaging_weight, mueller, flag, frow,
                     idx0, idxf, sign, csign)

    dw = inlined_array(dw, frow)

    w = dw[1].astype(real_type)

    data_vars = {
                'FIELD_ID':(('row',), da.full_like(time,
                            fid, chunks=row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(time,
                            ddid, chunks=row_out_chunk)),
                'SCAN_NUMBER':(('row',), da.full_like(time,
                            scanid, chunks=row_out_chunk)),
                'UVW':(('row', 'uvw'), uvw.rechunk({0:row_out_chunk})),
                'FREQ': (('chan',), freq),
                'FBIN_IDX':(('band',), fbin_idx),
                'FBIN_COUNTS':(('band',), fbin_counts)
            }

    if do_dirty:
        dirty = vis2im(uvw,
                       freq,
                       dw[0],
                       fbin_idx,
                       fbin_counts,
                       nx,
                       ny,
                       cell_rad,
                       weights=w,
                       # flag=mask.astype(np.uint8),
                       nthreads=nthreads,
                       epsilon=epsilon,
                       do_wstacking=wstack,
                       double_accum=double_accum)
        dirty = inlined_array(dirty, uvw)
        data_vars['DIRTY'] = (('band', 'nx', 'ny'), dirty)

    if do_psf:
        psf = vis2im(uvw,
                    freq,
                    dw[1],
                    fbin_idx,
                    fbin_counts,
                    nx_psf,
                    ny_psf,
                    cell_rad,
                    # flag=mask.astype(np.uint8),
                    nthreads=nthreads,
                    epsilon=epsilon,
                    do_wstacking=wstack,
                    double_accum=double_accum)
        psf = inlined_array(psf, uvw)
        wsum = da.max(psf, axis=(1, 2))
        data_vars['PSF'] = (('band', 'nx_psf', 'ny_psf'), psf)
        data_vars['WSUM'] = (('band',), wsum)


    if 'WSUM' not in data_vars.keys() or check_wsum:
        wsum2 = compute_wsum(w, fbin_idx, fbin_counts)
        # if check_wsum:
        #     try:
        #         assert np.allclose(wsum, wsum2, atol=epsilon)
        #     except Exception as e:
        #         print(wsum.compute())
        #         print(wsum2.compute())

        #         quit()
        #         raise RuntimeError("The peak of the PSF does not match "
        #                            "the sum of the weights in each "
        #                            "imaging band to within the gridding "
        #                            "precision. You may need to enable "
        #                            "double accumulation.")
        if 'WSUM' not in data_vars.keys():
            data_vars['WSUM'] = (('band',), wsum2)

    if do_weights:
        # uvw_scale = nu_out/nu  (nrow, 3) + nchan -> (nrow, 3*nchan)
        # exp(-2\pi i nu/c (ul + vm + w(n-1)))

        # grad F = R.H W (V - Rx) = R.H W V - R.H W R x = ID - hess(x)
        # TODO - BDA weights
        data_vars['WEIGHT'] = (('row', 'chan'), w.rechunk({0:row_out_chunk}))

    freq_out = da.from_array(freq_out, chunks=1, name=False)

    coords = {
        'chan': (('chan',), freq),
        'band': (('band',), freq_out[band_mapping]),
    }

    attrs = {
        'cell_rad': cell_rad,
        'ra' : radec[0],
        'dec': radec[1]
    }

    out_ds = Dataset(data_vars, coords, attrs=attrs)

    return out_ds

def weight_data(data, weight, imaging_weight, mueller, flag, frow,
                idx0, idxf, sign, csign):
    if weight is not None:
        wout = ('row', 'chan', 'corr')
    else:
        wout = None

    if imaging_weight is not None:
        iwout = ('row', 'chan', 'corr')
    else:
        iwout = None

    if mueller is not None:
        mout = ('row', 'chan', 'corr')
    else:
        mout = None

    if flag is not None:
        fout = ('row', 'chan', 'corr')
    else:
        fout = None

    if frow is not None:
        frout = ('row',)
    else:
        frout = None

    return da.blockwise(_weight_data_wrapper, ('2', 'row', 'chan'),
                        data, ('row', 'chan', 'corr'),
                        weight, wout,
                        imaging_weight, iwout,
                        mueller, mout,
                        flag, fout,
                        frow, frout,
                        idx0, None,
                        idxf, None,
                        sign, None,
                        csign, None,
                        new_axes={'2': 2},
                        dtype=data.dtype)

def _weight_data_wrapper(data, weight, imaging_weight, mueller, flag, frow,
                         idx0, idxf, sign, csign):
    if weight is not None:
        wout = weight[0]
    else:
        wout = None

    if imaging_weight is not None:
        iwout = imaging_weight[0]
    else:
        iwout = None

    if mueller is not None:
        mout = mueller[0]
    else:
        mout = None

    if flag is not None:
        fout = flag[0]
    else:
        fout = None


    return _weight_data_impl(data[0], wout, iwout, mout, fout, frow,
                             idx0, idxf, sign, csign)

def _weight_data_impl(data, weight, imaging_weight, mueller, flag, frow,
                      idx0, idxf, sign, csign):


    if weight is not None:
        weightxx = weight[:, :, idx0]
        weightyy = weight[:, :, idxf]
    else:
        weightxx = 1.0
        weightyy = 1.0

    if imaging_weight is not None:
        # weightxx = ne.evaluate('i * w', local_dict={
        #                        'i':imaging_weight[:, :, idx0],
        #                        'w':weightxx},
        #                        casting='same_kind')
        # weightyy = ne.evaluate('i * w', local_dict={
        #                        'i':imaging_weight[:, :, idxf],
        #                        'w':weightyy},
        #                        casting='same_kind')
        weightxx = weightxx * imaging_weight[:, :, idx0]
        weightyy = weightyy * imaging_weight[:, :, idxf]

    if mueller is not None:
        # apply adjoint of Mueller term to weighted data
        data = (data[:, :, idx0] * mueller[:, :, idx0].conj() * weightxx +
                sign * data[:, :, idxf] * mueller[:, :, idxf].conj() * weightyy) * csign
        weightxx = weightxx * np.absolute(mueller[:, :, idx0])**2
        weightyy = weightyy * np.absolute(mueller[:, :, idxf])**2

    else:
        # d = ne.evaluate('(dxx * wxx + s * dyy * wyy) * c', local_dict={
        #                    'dxx': data[:, :, idx0],
        #                    'wxx': weightxx,
        #                    's': sign,
        #                    'dyy': data[:, :, idxf],
        #                    'wyy': weightyy,
        #                    'c': csign}, casting='same_kind')
        data = (data[:, :, idx0] * weightxx +
                sign * data[:, :, idxf] * weightyy) * csign

    # w = ne.evaluate('weightxx + weightyy', casting='same_kind')
    weight = weightxx + weightyy

    if flag is not None:
        flag = flag[:, :, idx0] | flag[:, :, idxf]
        if frow is not None:
            flag = da.logical_or(flag, frow[:, None])

    weight[flag] = 0

    idx = weight != 0
    data[idx] = data[idx] / weight[idx]

    # weight -> complex (use blocker?)
    return np.concatenate([data[None], weight[None]])

