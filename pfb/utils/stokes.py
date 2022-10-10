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
import dask


def single_stokes(ds=None,
                  jones=None,
                  opts=None,
                  freq=None,
                  chan_width=None,
                  bandid=None,
                  cell_rad=None,
                  utime=None,
                  tbin_idx=None,
                  tbin_counts=None,
                  timeid=None,
                  radec=None,
                  antpos=None,
                  poltype=None):

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    data = getattr(ds, opts.data_column).data
    nrow, nchan, _ = data.shape

    ant1 = clone(ds.ANTENNA1.data)
    ant2 = clone(ds.ANTENNA2.data)
    uvw = clone(ds.UVW.data)

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = clone(ds.FLAG_ROW.data) | (ant1 == ant2)
    else:
        frow = (ant1 == ant2)

    frow = inlined_array(frow, [ant1, ant2])

    if opts.flag_column is not None:
        flag = getattr(ds, opts.flag_column).data
        flag = da.any(flag, axis=2)
        flag = da.logical_or(flag, frow[:, None])
    else:
        flag = da.broadcast_to(frow[:, None], (nrow, nchan))

    if opts.sigma_column is not None:
        sigma = getattr(ds, opts.sigma_column).data
        weight = 1.0/sigma**2
    elif opts.weight_column is not None:
        weight = getattr(ds, opts.weight_column).data
    else:
        weight = da.ones_like(data, dtype=real_type)

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    if jones is not None:
        if jones.dtype != complex_type:
            jones = jones.astype(complex_type)
        # qcal has chan and ant axes reversed compared to pfb implementation
        jones = da.swapaxes(jones, 1, 2)
    else:
        ntime = utime.size
        nchan = freq.size
        nant = antpos.shape[0]
        # import pdb; pdb.set_trace()
        jones = da.ones((ntime, nchan, nant, 1, 2),
                        chunks=(-1,)*5,
                        dtype=complex_type)

    # Note we do not chunk at this level since all the chunking happens upfront
    # we cast to dask arrays simply to defer the compute
    tbin_idx = da.from_array(tbin_idx, chunks=(-1))
    tbin_counts = da.from_array(tbin_counts, chunks=(-1))
    vis, wgt = weight_data(data, weight, jones, tbin_idx, tbin_counts,
                        ant1, ant2, pol=poltype, product=opts.product)

    vis = inlined_array(vis, [ant1, ant2, tbin_idx, tbin_counts, jones])
    wgt = inlined_array(wgt, [ant1, ant2, tbin_idx, tbin_counts, jones])

    if isinstance(opts.radec, str):
        raise NotImplementedError()
    elif isinstance(opts.radec, np.ndarray) and not np.array_equal(radec, opts.radec):
        raise NotImplementedError()


    flag = inlined_array(flag, [frow])

    # do this before casting to dask array otherwise
    # serialisation of attrs fails
    freq_out = np.mean(freq)

    # simple average over channels
    if opts.chan_average > 1:
        from africanus.averaging.dask import time_and_channel

        res = time_and_channel(
                    ds.TIME.data,
                    ds.INTERVAL.data,
                    ant1,
                    ant2,
                    uvw=uvw,
                    flag=flag[:, :, None],
                    weight_spectrum=wgt[:, :, None],
                    visibilities=vis[:, :, None],
                    chan_freq=da.from_array(freq, chunks=-1),
                    chan_width=da.from_array(chan_width, chunks=-1),
                    time_bin_secs=1e-15,
                    chan_bin_size=opts.chan_average)

        # map_blocks to deal with nan row chunks.
        # only freq average -> row chunks are preserved
        cchunk = nchan//opts.chan_average
        vis = res.visibilities.map_blocks(lambda x: x,
                                          chunks=(nrow, cchunk, 1))[:, :, 0]
        wgt = res.weight_spectrum.map_blocks(lambda x: x,
                                             chunks=(nrow, cchunk, 1))[:, :, 0]
        flag = res.flag.map_blocks(lambda x: x,
                                   chunks=(nrow, cchunk, 1))[:, :, 0]
        uvw = res.uvw.map_blocks(lambda x: x,
                                 chunks=(nrow, 3))

        freq = res.chan_freq.map_blocks(lambda x: x, chunks=(cchunk,))

    # if opts.bda_decorr < 1:
    #     wgt = da.where(mask, wgt, 0.0)
    #     from africanus.averaging.dask import bda

    #     w_avs = []
    #     uvw = uvw.compute()
    #     t = time.compute()
    #     a1 = ant1.compute()
    #     a2 = ant2.compute()
    #     intv = interval.compute()
    #     fr = frow.compute()[:, None, None]

    #     res = bda(ds.TIME.data,
    #                 ds.INTERVAL.data,
    #                 ant1, ant2,
    #                 uvw=uvw,
    #                 flag=f[:, :, None],
    #                 weight_spectrum=wgt[:, :, None],
    #                 chan_freq=freq,
    #                 chan_width=chan_width,
    #                 decorrelation=0.95,
    #                 min_nchan=freq.size)

    #     uvw = res.uvw.reshape(-1, nchan, 3)[:, 0, :]
    #     wgt = res.weight_spectrum.reshape(-1, nchan).squeeze()

    #     uvw = uvw.rechunk({0:opts.row_out_chunk})

    mask = ~flag

    # data_coords = {}
    # data_coords['FREQ'] = (('chan',), freq)
    # this breaks concat because not all datasets have the same
    # number of times in them.
    # data_coords['TIME'] = (('time',), utime)

    data_vars = {}
    data_vars['FREQ'] = (('chan',), freq)
    data_vars['WEIGHT'] = (('row', 'chan'), wgt)
    data_vars['UVW'] = (('row', 'uvw'), uvw)
    data_vars['VIS'] = (('row', 'chan'), vis)
    data_vars['MASK'] = (('row', 'chan'), mask.astype(np.uint8))

    # TODO - interpolate beam in time and freq
    npix = int(np.deg2rad(opts.max_field_of_view)/cell_rad)
    beam = interp_beam(freq_out/1e6, npix, npix, np.rad2deg(cell_rad), opts.beam_model)


    # Instead of BEAM we should have a pre-init step which computes
    # per facet best approximations to smooth beams as describedin
    # https://www.overleaf.com/read/yzrsrdwxhxrd
    data_vars['BEAM'] = (('scalar'), beam)

    # TODO - provide time and freq centroids
    attrs = {
        'ra' : radec[0],
        'dec': radec[1],
        'fieldid': ds.FIELD_ID,
        'ddid': ds.DATA_DESC_ID,
        'scanid': ds.SCAN_NUMBER,
        'bandid': int(bandid),
        'freq_out': freq_out,
        'timeid': int(timeid),
        'time_out': np.mean(utime)
    }

    out_ds = Dataset(data_vars, # coords=data_coords,
                     attrs=attrs).chunk({'row':100000,
                                         'chan':128})

    return out_ds.unify_chunks()


def weight_data(data, weight, jones, tbin_idx, tbin_counts,
                ant1, ant2, pol='linear', product='I'):
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
                       pol, None,
                       product, None,
                       align_arrays=False,
                       meta=np.empty((0, 0), dtype=object))

    vis = da.blockwise(getitem, 'rf', res, 'rf', 0, None, dtype=data.dtype)
    wgt = da.blockwise(getitem, 'rf', res, 'rf', 1, None, dtype=weight.dtype)

    return vis, wgt

def _weight_data(data, weight, jones, tbin_idx, tbin_counts,
                 ant1, ant2, pol, product):

    return _weight_data_impl(data[0], weight[0], jones[0][0][0],
                             tbin_idx, tbin_counts,
                             ant1, ant2, pol, product)

@generated_jit(nopython=True, nogil=True, cache=True)
def _weight_data_impl(data, weight, jones, tbin_idx, tbin_counts,
                      ant1, ant2, pol, product):

    coerce_literal(_weight_data, ["product", "pol"])

    vis_func, wgt_func = stokes_funcs(data, jones, product, pol=pol)

    def _impl(data, weight, jones, tbin_idx, tbin_counts,
              ant1, ant2, pol, product):
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
                    if wval > 1e-6:
                        wgt[row, chan] = wval
                        vis[row, chan] = vis_func(gp[chan], gq[chan],
                                                weight[row, chan],
                                                data[row, chan])/wval

        return vis, wgt
    return _impl


def stokes_funcs(data, jones, product, pol):
    if pol != literal('linear'):
        raise NotImplementedError("Circular polarisation not yet supported")
    # The expressions for DIAG_DIAG and DIAG mode are essentially the same
    if jones.ndim == 5:
        # I and Q have identical weights
        @njit(nogil=True, fastmath=True, inline='always')
        def wfunc(gp, gq, W):
            gp00 = gp[0]
            gp11 = gp[1]
            gq00 = gq[0]
            gq11 = gq[1]
            W0 = W[0]
            W3 = W[-1]
            return np.real(W0*gp00*gq00*np.conjugate(gp00)*np.conjugate(gq00) +
                    W3*gp11*gq11*np.conjugate(gp11)*np.conjugate(gq11))

        if product == literal('I'):
            @njit(nogil=True, fastmath=True, inline='always')
            def vfunc(gp, gq, W, V):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W0 = W[0]
                W3 = W[-1]
                v00 = V[0]
                v11 = V[-1]
                return (W0*gq00*v00*np.conjugate(gp00) +
                        W3*gq11*v11*np.conjugate(gp11))

        elif product == literal('Q'):
            @njit(nogil=True, fastmath=True, inline='always')
            def vfunc(gp, gq, W, V):
                gp00 = gp[0]
                gp11 = gp[1]
                gq00 = gq[0]
                gq11 = gq[1]
                W0 = W[0]
                W3 = W[-1]
                v00 = V[0]
                v11 = V[-1]
                return (W0*gq00*v00*np.conjugate(gp00) -
                        W3*gq11*v11*np.conjugate(gp11))

        else:
            raise ValueError("The requested product is not available from input data")

        return vfunc, wfunc

    # Full mode
    elif jones.ndim == 6:
        raise NotImplementedError("Full polarisation imaging not yet supported")

    else:
        raise ValueError("jones array has an unsupported number of dimensions")
