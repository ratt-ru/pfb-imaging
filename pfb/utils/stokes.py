import numpy as np
import numexpr as ne
import dask
import dask.array as da
from xarray import Dataset
from africanus.gridding.wgridder.dask import dirty as vis2im

def single_stokes(data, weight, imaging_weight, mueller, flag, frow, uvw,
                  time, fid, ddid,
                  pol_type, row_out_chunk, nvthreads, epsilon, wstack,
                  double_accum, flipv, freq, fbin_idx, fbin_counts, nx, ny,
                  nx_psf, ny_psf, cell_rad, idx0, idxf, sign, csign):

    data_type = data.dtype
    data_shape = data.shape
    data_chunks = data.chunks
    real_type = data.real.dtype

    dw = weight_data(data, weight, imaging_weight, mueller, flag, # frow,
                     idx0, idxf, sign, csign)

    dirty = vis2im(uvw,
                   freq,
                   dw[0],
                   fbin_idx,
                   fbin_counts,
                   nx,
                   ny,
                   cell_rad,
                   weights=dw[1].astype(real_type),
                   # flag=mask.astype(np.uint8),
                   nthreads=nvthreads,
                   epsilon=epsilon,
                   do_wstacking=wstack,
                   double_accum=double_accum)

    psf = vis2im(uvw,
                 freq,
                 dw[1],
                 fbin_idx,
                 fbin_counts,
                 nx_psf,
                 ny_psf,
                 cell_rad,
                 # flag=mask.astype(np.uint8),
                 nthreads=nvthreads,
                 epsilon=epsilon,
                 do_wstacking=wstack,
                 double_accum=double_accum)

    data_vars = {
                'FIELD_ID':(('row',), da.full_like(time,
                            fid, chunks=row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(time,
                            ddid, chunks=row_out_chunk)),
                'WEIGHT':(('row', 'chan'), dw[1].rechunk({0:row_out_chunk})),  # why no 'f4'?
                'UVW':(('row', 'uvw'), uvw.rechunk({0:row_out_chunk})),
                'DIRTY':(('band', 'nx', 'ny'), dirty),
                'PSF':(('band', 'nx_psf', 'ny_psf'), psf)
            }

    coords = {
        'chan': (('chan',), freq)
    }

    out_ds = Dataset(data_vars, coords)

    return out_ds

def weight_data(data, weight, imaging_weight, mueller, flag, # frow,
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

    # if frow is not None:
    #     frout = ('row',)
    # else:
    #     frout = None

    # import pdb; pdb.set_trace()

    return da.blockwise(_weight_data_wrapper, ('2', 'row', 'chan'),
                        data, ('row', 'chan', 'corr'),
                        weight, wout,
                        imaging_weight, iwout,
                        mueller, mout,
                        flag, fout,
                        # frow, frout,
                        idx0, None,
                        idxf, None,
                        sign, None,
                        csign, None,
                        new_axes={'2': 2},
                        dtype=data.dtype)

def _weight_data_wrapper(data, weight, imaging_weight, mueller, flag, # frow,
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


    return _weight_data_impl(data[0], wout, iwout, mout, fout, # frow,
                             idx0, idxf, sign, csign)

def _weight_data_impl(data, weight, imaging_weight, mueller, flag, # frow,
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
        # d = ne.evaluate('(dxx * conj(mxx) * wxx + s * dyy * conj(myy) * wyy) * c', local_dict={
        #                    'dxx': data[:, :, idx0],
        #                    'mxx': mueller[:, :, idx0],
        #                    'wxx': weightxx,
        #                    's': sign,
        #                    'dyy': data[:, :, idxf],
        #                    'myy': mueller[:, :, idxf],
        #                    'wyy': weightyy,
        #                    'c': csign}, casting='same_kind')
        # ne.evaluate('wxx * abs(mxx).real**2', local_dict={
        #             'wxx': weightxx,
        #             'mxx': mueller[:, :, idx0]}, out= weightxx, casting='same_kind')
        # ne.evaluate('wyy * abs(myy).real**2', local_dict={
        #             'wyy': weightyy,
        #             'myy': mueller[:, :, idxf]}, out= weightyy, casting='same_kind')
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
        # if frow is not None:
        #     flag = da.logical_or(flag, frow[:, None])

    weight[flag] = 0

    idx = weight != 0
    data[idx] = data[idx] / weight[idx]

    # weight -> complex (use blocker?)
    return np.concatenate([data[None], weight[None]])

