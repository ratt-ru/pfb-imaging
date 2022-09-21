import numpy as np
from numba import generated_jit, njit
from numba.types import literal
from dask.graph_manipulation import clone
import dask.array as da
from xarray import Dataset
from pfb.operators.gridder import vis2im
from pfb.utils.misc import coerce_literal
from daskms.optimisation import inlined_array
from operator import getitem
from pfb.utils.beam import interp_beam

def single_corr(ds=None,
                  jones=None,
                  opts=None,
                  freq=None,
                  freq_out=None,
                  chan_width=None,
                  bandid=None,
                  cell_rad=None,
                  tbin_idx=None,
                  tbin_counts=None,
                  radec=None):

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    data = getattr(ds, opts.data_column).data
    nrow, nchan, _ = data.shape

    ant1 = ds.ANTENNA1.data
    ant2 = ds.ANTENNA2.data

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = ds.FLAG_ROW.data | (ant1 == ant2)
    else:
        frow = (ant1 == ant2)

    frow = inlined_array(frow, [ant1, ant2])

    if opts.weight_column is not None:
        weight = getattr(ds, opts.weight_column).data
    else:
        weight = da.ones_like(data, dtype=real_type)

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    if jones is None:
        # TODO - remove unnecessary jones multiplies
        ntime = tbin_idx.size
        nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
        jones = da.ones((ntime, nchan, nant, 1, 2),
                        chunks=(tbin_idx.chunks[0][0], -1, -1, 1, 2),
                        dtype=complex_type)
    elif jones.dtype != complex_type:
        jones = jones.astype(complex_type)

    # qcal has chan and ant axes reversed compared to pfb implementation
    jones = da.swapaxes(jones, 1, 2)

    vis, wgt = weight_data(data, weight, jones, tbin_idx, tbin_counts,
                           ant1, ant2)

    vis = inlined_array(vis, [ant1, ant2, tbin_idx, tbin_counts, jones])
    wgt = inlined_array(wgt, [ant1, ant2, tbin_idx, tbin_counts, jones])

    if opts.flag_column is not None:
        flag = getattr(ds, opts.flag_column).data
        flag = da.any(flag, axis=2)
        flag = da.logical_or(flag, frow[:, None])
    else:
        flag = da.broadcast_to(frow[:, None], (nrow, nchan))

    mask = ~flag
    mask = inlined_array(mask, [frow])
    uvw = ds.UVW.data

    data_vars = {'FREQ': (('chan',), freq)}

    wsum = da.sum(wgt[mask])
    data_vars['WSUM'] = (('scalar'), da.array((wsum,)))

    # wgt = wgt.rechunk({0:opts.row_out_chunk})
    data_vars['WEIGHT'] = (('row', 'chan'), wgt)

    # uvw = uvw.rechunk({0:opts.row_out_chunk})
    data_vars['UVW'] = (('row', 'uvw'), uvw)

    # vis = vis.rechunk({0:opts.row_out_chunk})
    data_vars['VIS'] = (('row', 'chan'), vis)

    # MASK = ~FLAG.astype(np.uint8) for wgridder convention
    # mask = mask.rechunk({0:opts.row_out_chunk})
    data_vars['MASK'] = (('row', 'chan'), mask.astype(np.uint8))

    # if opts.weight:
    #     wgt = da.where(mask, wgt, 0.0)
    #     # TODO - BDA over frequency
    #     if opts.bda_decorr < 1:
    #         raise NotImplementedError("BDA not working yet")
    #         from africanus.averaging.dask import bda

    #         w_avs = []
    #         uvw = uvw.compute()
    #         t = time.compute()
    #         a1 = ant1.compute()
    #         a2 = ant2.compute()
    #         intv = interval.compute()
    #         fr = frow.compute()[:, None, None]

    #         res = bda(ds.TIME.data,
    #                   ds.INTERVAL.data,
    #                   ant1, ant2,
    #                   uvw=uvw,
    #                   flag=f[:, :, None],
    #                   weight_spectrum=wgt[:, :, None],
    #                   chan_freq=freq,
    #                   chan_width=chan_width,
    #                   decorrelation=0.95,
    #                   min_nchan=freq.size)

    #         uvw = res.uvw.reshape(-1, nchan, 3)[:, 0, :]
    #         wgt = res.weight_spectrum.reshape(-1, nchan).squeeze()

    #         uvw = uvw.rechunk({0:opts.row_out_chunk})
    #         data_vars['UVW'] = (('row', 'uvw'), uvw)

    #     wgt = wgt.rechunk({0:opts.row_out_chunk})
    #     data_vars['WEIGHT'] = (('row', 'chan'), wgt)


    # TODO - interpolate beam in time and freq
    npix = int(np.deg2rad(opts.max_field_of_view)/cell_rad)
    freq_out = np.mean(freq)
    beam = interp_beam(freq_out/1e6, npix, npix, np.rad2deg(cell_rad), opts.beam_model)

    data_vars['BEAM'] = (('scalar'), beam)

    attrs = {
        'ra' : radec[0],
        'dec': radec[1],
        'fieldid': ds.FIELD_ID,
        'ddid': ds.DATA_DESC_ID,
        'scanid': ds.SCAN_NUMBER,
        'bandid': int(bandid),
        'freq_out': freq_out
    }

    out_ds = Dataset(data_vars, attrs=attrs)

    return out_ds


def weight_data(data, weight, jones, tbin_idx, tbin_counts,
                ant1, ant2):
    # data are not necessarily 2x2 so we need separate labels
    # for jones correlations and data/weight correlations
    if jones.ndim == 5:
        jout = 'rafdx'
    elif jones.ndim == 6:
        jout = 'rafdxx'
        # TODO - how do we know if we should return
        # jones[0][0][0] or jones[0][0][0][0] in function wrapper?
        # Not required with delayed
        raise NotImplementedError("Not yet implemented")
    res = da.blockwise(_weight_data, 'rf',
                       data, 'rfc',
                       weight, 'rfc',
                       jones, jout,
                       tbin_idx, 'r',
                       tbin_counts, 'r',
                       ant1, 'r',
                       ant2, 'r',
                       align_arrays=False,
                       meta=np.empty((0, 0), dtype=object))

    vis = da.blockwise(getitem, 'rf', res, 'rf', 0, None, dtype=data.dtype)
    wgt = da.blockwise(getitem, 'rf', res, 'rf', 1, None, dtype=weight.dtype)


    return vis, wgt

def _weight_data(data, weight, jones, tbin_idx, tbin_counts,
                      ant1, ant2):
    return _weight_data_impl(data[0], weight[0], jones[0][0][0],
                             tbin_idx, tbin_counts, ant1, ant2)

@generated_jit(nopython=True, nogil=True, cache=True)
def _weight_data_impl(data, weight, jones, tbin_idx, tbin_counts,
                      ant1, ant2):

    vis_func, wgt_func = corr_funcs(data, jones)

    def _impl(data, weight, jones, tbin_idx, tbin_counts,
              ant1, ant2):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        tbin_idx -= tbin_idx.min()
        nt = np.shape(tbin_idx)[0]
        nrow, nchan, ncorr = data.shape
        vis = np.zeros((nrow, nchan), dtype=data.dtype)
        wgt = np.zeros((nrow, nchan), dtype=data.real.dtype)

        for t in range(nt):
            for row in range(tbin_idx[t],
                             tbin_idx[t] + tbin_counts[t]):
                p = int(ant1[row])
                q = int(ant2[row])
                gp = jones[t, p, :, 0]
                gq = jones[t, q, :, 0]
                for chan in range(nchan):
                    wval = wgt_func(gp[chan], gq[chan],
                                    weight[row, chan])
                    wgt[row, chan] = wval
                    vis[row, chan] = vis_func(gp[chan], gq[chan],
                                              weight[row, chan],
                                              data[row, chan])  #/wval

        return vis, wgt
    return _impl


def corr_funcs(data, jones):
    # The expressions for DIAG_DIAG and DIAG mode are essentially the same
    if jones.ndim == 5:
        # I and Q have identical weights
        @njit(nogil=True, fastmath=True, inline='always')
        def wfunc(gp, gq, W):
            gp00 = gp[0]
            gq00 = gq[0]
            W0 = W[0]
            return np.real(W0*gp00*gq00*np.conjugate(gp00)*np.conjugate(gq00))

        @njit(nogil=True, fastmath=True, inline='always')
        def vfunc(gp, gq, W, V):
            gp00 = gp[0]
            gq00 = gq[0]
            W0 = W[0]
            v00 = V[0]
            return W0*gq00*v00*np.conjugate(gp00)

        return vfunc, wfunc

    # Full mode
    elif jones.ndim == 6:
        raise NotImplementedError("Full polarisation imaging not yet supported")

    else:
        raise ValueError("jones array has an unsupported number of dimensions")
