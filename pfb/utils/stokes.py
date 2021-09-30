import numpy as np
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

    weights = getattr(ds, weight_column).data

    if imaging_weight_column is not None:
        imaging_weights = getattr(ds, imaging_weight_column).data

        weightsxx = imaging_weights[:, :, idx0] * weights[:, :, idx0]
        weightsyy = imaging_weights[:, :, idxf] * weights[:, :, idxf]
    else:
        weightsxx = weights[:, :, idx0]
        weightsyy = weights[:, :, idxf]

    # adjoint of mueller term
    if mueller_column is not None:
        mueller = getattr(ds, mueller_column).data
        data = (dataxx *  mueller[:, :, idx0].conj() * weightsxx +
                sign * datayy * mueller[:, :, idxf].conj() * weightsyy) * csign
        weightsxx *= da.absolute(mueller[:, :, idx0])**2
        weightsyy *= da.absolute(mueller[:, :, idxf])**2

    else:
        data = (weightsxx * dataxx + sign * weightsyy * datayy) * csign

    weights = weightsxx + weightsyy
    # TODO - turn off this stupid warning
    data = da.where(weights, data / weights, 0.0j)
    real_type = data.real.dtype

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
                   weights=weights,
                   flag=mask.astype(np.uint8),
                   nthreads=nvthreads,
                   epsilon=epsilon,
                   do_wstacking=wstack,
                   double_accum=double_accum)

    psf = vis2im(uvw,
                 freq,
                 weights.astype(data_type),
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

    weights = da.where(mask, weights, 0.0)

    data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.FIELD_ID, chunks=row_out_chunk)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data,
                            ds.DATA_DESC_ID, chunks=row_out_chunk)),
                'WEIGHT':(('row', 'chan'), weights.rechunk({0:row_out_chunk})),  # why no 'f4'?
                'UVW':(('row', 'uvw'), uvw.rechunk({0:row_out_chunk}))
            }

    coords = {
        'chan': (('chan',), freq)
    }

    out_ds = Dataset(data_vars, coords)

    return dirty, psf, out_ds