import numpy as np
import numexpr as ne
import dask.array as da
from xarray import Dataset
from africanus.gridding.wgridder.dask import dirty as vis2im

def single_stokes(ds, data_column, weight_column, imaging_weight_column,
                  mueller_column, flag_column,
                  frow, pol_type, row_out_chunk, nvthreads, epsilon, wstack,
                  double_accum, flipv, freq, fbin_idx, fbin_counts, nx, ny,
                  nx_psf, ny_psf, cell_rad, idx0, idxf, sign, csign):

    data = getattr(ds, data_column).data
    dataxx = data[:, :, idx0]
    datayy = data[:, :, idxf]

    data_type = getattr(ds, data_column).data.dtype
    data_shape = getattr(ds, data_column).data.shape
    data_chunks = getattr(ds, data_column).data.chunks

    weight = getattr(ds, weight_column).data

    if imaging_weight_column is not None:
        imaging_weight = getattr(ds, imaging_weight_column).data
    else:
        imaging_weight = None

    # adjoint of mueller term
    if mueller_column is not None:
        mueller = getattr(ds, mueller_column).data
    else:
        mueller = None

    real_type = data.real.dtype

    dw = weight_data(data, weight, imaging_weight, mueller, idx0, idxf, sign, csign)
    data = dw[0]
    weight = dw[1]

    # only keep data where both corrs are unflagged
    flag = getattr(ds, flag_column).data
    flagxx = flag[:, :, idx0]
    flagyy = flag[:, :, idxf]
    # ducc0 uses uint8 mask not flag
    mask = ~ da.logical_or((flagxx | flagyy), frow[:, None])
    uvw = ds.UVW.data

    dirty = vis2im(uvw,
                   freq,
                   data,
                   fbin_idx,
                   fbin_counts,
                   nx,
                   ny,
                   cell_rad,
                   weights=weight.astype(real_type),
                   flag=mask.astype(np.uint8),
                   nthreads=nvthreads,
                   epsilon=epsilon,
                   do_wstacking=wstack,
                   double_accum=double_accum)

    psf = vis2im(uvw,
                 freq,
                 weight,
                 fbin_idx,
                 fbin_counts,
                 nx_psf,
                 ny_psf,
                 cell_rad,
                 flag=mask.astype(np.uint8),
                 nthreads=nvthreads,
                 epsilon=epsilon,
                 do_wstacking=wstack,
                 double_accum=double_accum)

    weight = da.where(mask, weight, 0.0)

    data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.FIELD_ID, chunks=row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.DATA_DESC_ID, chunks=row_out_chunk)),
                'WEIGHT':(('row', 'chan'), weight.rechunk({0:row_out_chunk})),  # why no 'f4'?
                'UVW':(('row', 'uvw'), uvw.rechunk({0:row_out_chunk}))
            }

    coords = {
        'chan': (('chan',), freq)
    }

    out_ds = Dataset(data_vars, coords)

    return dirty, psf, out_ds

def weight_data(data, weight, imaging_weight, mueller, idx0, idxf, sign, csign):
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


    return da.blockwise(_weight_data_wrapper, ('2', 'row', 'chan'),
                        data, ('row', 'chan', 'corr'),
                        weight, wout,
                        imaging_weight, iwout,
                        mueller, mout,
                        idx0, None,
                        idxf, None,
                        sign, None,
                        csign, None,
                        new_axes={'2': 2},
                        dtype=data.dtype)

def _weight_data_wrapper(data, weight, imaging_weight, mueller, idx0, idxf, sign, csign):
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
    return _weight_data_impl(data[0], wout, iwout, mout, idx0, idxf, sign, csign)

def _weight_data_impl(data, weight, imaging_weight, mueller, idx0, idxf, sign, csign):

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
        weightxx *= imaging_weight[:, :, idx0]
        weightyy *= imaging_weight[:, :, idxf]

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
        weightxx *= np.absolute(mueller[:, :, idx0])**2
        weightyy *= np.absolute(mueller[:, :, idxf])**2

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

    idx = weight != 0
    data[idx] = data[idx] / weight[idx]

    # weight -> complex (use blocker?)
    return np.concatenate([data[None], weight[None]])

