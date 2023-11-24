import numpy as np
import numexpr as ne
from numba import generated_jit, njit, prange
from numba.types import literal
from dask.graph_manipulation import clone
import dask.array as da
from xarray import Dataset
from pfb.operators.gridder import vis2im
from quartical.utils.numba import coerce_literal
from operator import getitem
from pfb.utils.beam import interp_beam
import dask
from quartical.utils.dask import Blocker


def weight_from_sigma(sigma):
    weight = ne.evaluate('1.0/(sigma*sigma)',
                         casting='same_kind')
    return weight


def single_stokes(ds=None,
                  jones=None,
                  opts=None,
                  freq=None,
                  chan_width=None,
                  cell_rad=None,
                  utime=None,
                  tbin_idx=None,
                  tbin_counts=None,
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
    nrow, nchan, ncorr = data.shape

    # clone shared nodes
    ant1 = clone(ds.ANTENNA1.data)
    ant2 = clone(ds.ANTENNA2.data)
    uvw = clone(ds.UVW.data)

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = clone(ds.FLAG_ROW.data) | (ant1 == ant2)
    else:
        frow = (ant1 == ant2)

    if opts.flag_column is not None:
        flag = getattr(ds, opts.flag_column).data
        flag = da.any(flag, axis=2)
        flag = da.logical_or(flag, frow[:, None])
    else:
        flag = da.broadcast_to(frow[:, None], (nrow, nchan))

    if opts.sigma_column is not None:
        sigma = getattr(ds, opts.sigma_column).data
        # weight = 1.0/sigma**2
        weight = da.map_blocks(weight_from_sigma,
                               sigma,
                               chunks=sigma.chunks)
    elif opts.weight_column is not None:
        weight = getattr(ds, opts.weight_column).data
        if opts.weight_column=='WEIGHT':
            weight = da.broadcast_to(weight[:, None, :],
                                     (nrow, nchan, ncorr),
                                     chunks=data.chunks)
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
        # data are not necessarily 2x2 so we need separate labels
        # for jones correlations and data/weight correlations
        if jones.ndim == 5:
            jout = 'rafdx'
        elif jones.ndim == 6:
            jout = 'rafdxx'
            jones = jones.reshape(-1, nchan, 2, 2)
    else:
        ntime = utime.size
        nchan = freq.size
        nant = antpos.shape[0]
        jones = da.ones((ntime, nant, nchan, 1, 2),
                        chunks=(-1,)*5,
                        dtype=complex_type)
        jout = 'rafdx'

    # Note we do not chunk at this level since all the chunking happens upfront
    # we cast to dask arrays simply to defer the compute
    tbin_idx = da.from_array(tbin_idx, chunks=(-1))
    tbin_counts = da.from_array(tbin_counts, chunks=(-1))



    # compute Stokes data and weights
    blocker = Blocker(weight_data, 'rf')
    blocker.add_input("data", data, 'rfc')
    blocker.add_input('weight', weight, 'rfc')
    blocker.add_input('flag', flag, 'rf')
    blocker.add_input('jones', jones, jout)
    blocker.add_input('tbin_idx', tbin_idx, 'r')
    blocker.add_input('tbin_counts', tbin_counts, 'r')
    blocker.add_input('ant1', ant1, 'r')
    blocker.add_input('ant2', ant2, 'r')
    blocker.add_input('pol', poltype)
    blocker.add_input('product', opts.product)

    nrow, nchan, _ = data.shape
    blocker.add_output('vis', 'rf', ((nrow,),(nchan,)), data.dtype)
    blocker.add_output('wgt', 'rf', ((nrow,),(nchan,)), weight.dtype)

    output_dict = blocker.get_dask_outputs()
    vis = output_dict['vis']
    wgt = output_dict['wgt']

    if isinstance(opts.radec, str):
        raise NotImplementedError()
    elif isinstance(opts.radec, np.ndarray) and not np.array_equal(radec, opts.radec):
        raise NotImplementedError()

    # do this before casting to dask array otherwise
    # serialisation of attrs fails
    freq_out = np.mean(freq)
    freq_min = freq.min()
    freq_max = freq.max()

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

    data_vars = {}
    data_vars['FREQ'] = (('chan',), freq)
    data_vars['WEIGHT'] = (('row', 'chan'), wgt)
    data_vars['UVW'] = (('row', 'uvw'), uvw)
    data_vars['VIS'] = (('row', 'chan'), vis)
    data_vars['MASK'] = (('row', 'chan'), mask.astype(np.uint8))

    # TODO - interpolate beam in time and frequency
    # Instead of BEAM we should have a pre-init step which computes
    # per facet best approximations to smooth beams as described in
    # https://www.overleaf.com/read/yzrsrdwxhxrd
    npix = int(np.deg2rad(opts.max_field_of_view)/cell_rad)
    beam, l_beam, m_beam = interp_beam(freq_out/1e6, npix, npix,
                                       np.rad2deg(cell_rad),
                                       opts.beam_model,
                                       utime=utime,
                                       ant_pos=antpos,
                                       phase_dir=radec)
    data_vars['BEAM'] = (('l_beam','m_beam'), beam)

    coords = {'chan': (('chan',), freq),
              'l_beam': (('l_beam',), l_beam),
              'm_beam': (('m_beam',), m_beam)}
            #   'row': (('row',), ds.ROWID.values)}

    # TODO - provide time and freq centroids
    attrs = {
        'ra' : radec[0],
        'dec': radec[1],
        'fieldid': ds.FIELD_ID,
        'ddid': ds.DATA_DESC_ID,
        'scanid': ds.SCAN_NUMBER,
        'freq_out': freq_out,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'time_out': np.mean(utime),
        'time_min': utime.min(),
        'time_max': utime.max(),
        'product': opts.product
    }

    out_ds = Dataset(data_vars, coords=coords,
                     attrs=attrs).chunk({'row':100000,
                                         'chan':128})

    return out_ds.unify_chunks()


def weight_data(data, weight, flag, jones, tbin_idx, tbin_counts,
                ant1, ant2, pol, product):

    vis, wgt = _weight_data_impl(data, weight, flag, jones,
                                 tbin_idx, tbin_counts,
                                 ant1, ant2, pol, product)

    out_dict = {}
    out_dict['vis'] = vis
    out_dict['wgt'] = wgt

    return out_dict


@generated_jit(nopython=True, nogil=True, cache=True, parallel=True)
def _weight_data_impl(data, weight, flag, jones, tbin_idx, tbin_counts,
                      ant1, ant2, pol, product):

    coerce_literal(weight_data, ["product", "pol"])

    vis_func, wgt_func = stokes_funcs(data, jones, product, pol=pol)

    def _impl(data, weight, flag, jones, tbin_idx, tbin_counts,
              ant1, ant2, pol, product):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        tbin_idx -= tbin_idx.min()
        nt = np.shape(tbin_idx)[0]
        nrow, nchan, ncorr = data.shape
        vis = np.zeros((nrow, nchan), dtype=data.dtype)
        wgt = np.zeros((nrow, nchan), dtype=data.real.dtype)

        for t in prange(nt):
            for row in range(tbin_idx[t],
                             tbin_idx[t] + tbin_counts[t]):
                p = int(ant1[row])
                q = int(ant2[row])
                gp = jones[t, p, :, 0]
                gq = jones[t, q, :, 0]
                for chan in range(nchan):
                    if flag[row, chan]:
                        continue
                    wgt[row, chan] = wgt_func(gp[chan], gq[chan],
                                              weight[row, chan])
                    vis[row, chan] = vis_func(gp[chan], gq[chan],
                                              weight[row, chan],
                                              data[row, chan])

        return vis, wgt
    return _impl


def stokes_funcs(data, jones, product, pol):
    import sympy as sm
    from sympy.physics.quantum import TensorProduct
    from sympy.utilities.lambdify import lambdify
    # set up symbolic expressions
    gp00, gp10, gp01, gp11 = sm.symbols("gp00 gp10 gp01 gp11", real=False)
    gq00, gq10, gq01, gq11 = sm.symbols("gq00 gq10 gq01 gq11", real=False)
    w0, w1, w2, w3 = sm.symbols("W0 W1 W2 W3", real=True)
    v00, v10, v01, v11 = sm.symbols("v00 v10 v01 v11", real=False)

    # Jones matrices
    Gp = sm.Matrix([[gp00, gp01],[gp10, gp11]])
    Gq = sm.Matrix([[gq00, gq01],[gq10, gq11]])

    # Mueller matrix (row major form)
    Mpq = TensorProduct(Gp, Gq.conjugate())
    Mpqinv = TensorProduct(Gq.conjugate().inv(), Gp.inv())

    # inverse noise covariance
    Sinv = sm.Matrix([[w0, 0, 0, 0],
                      [0, w1, 0, 0],
                      [0, 0, w2, 0],
                      [0, 0, 0, w3]])
    S = Sinv.inv()

    # visibilities
    Vpq = sm.Matrix([[v00], [v01], [v10], [v11]])

    # Full Stokes to corr operator
    # Is this the only difference between linear and circular pol?
    # What about paralactic angle rotation?
    if pol == literal('linear'):
        T = sm.Matrix([[1.0, 1.0, 0, 0],
                       [0, 0, 1.0, -1.0j],
                       [0, 0, 1.0, 1.0j],
                       [1, -1, 0, 0]])
    elif pol == literal('circular'):
        T = sm.Matrix([[1.0, 0, 0, 1.0],
                       [0, 1.0, 1.0j, 0],
                       [0, 1.0, -1.0j, 0],
                       [1, 0, 0, -1]])
    Tinv = T.inv()

    # Full Stokes weights
    W = T.H * Mpq.H * Sinv * Mpq * T
    Winv = Tinv * Mpqinv * S * Mpqinv.H * Tinv.H

    # Full Stokes coherencies
    C = Winv * (T.H * (Mpq.H * (Sinv * Vpq)))

    if jones.ndim == 6:  # Full mode
        if product == literal('I'):
            i = 0
        elif product == literal('Q'):
            i = 1
        elif product == literal('U'):
            i = 2
        elif product == literal('V'):
            i = 3
        else:
            raise ValueError(f"Unknown polarisation product {product}")

        Wsymb = lambdify((gp00, gp01, gp10, gp11,
                            gq00, gq01, gq10, gq11,
                            w0, w1, w2, w3),
                            sm.simplify(sm.expand(W[i,i])))
        Wjfn = njit(nogil=True, fastmath=True, inline='always')(Wsymb)


        Dsymb = lambdify((gp00, gp01, gp10, gp11,
                            gq00, gq01, gq10, gq11,
                            w0, w1, w2, w3,
                            v00, v01, v10, v11),
                            sm.simplify(smexpand(C[i])))
        Djfn = njit(nogil=True, fastmath=True, inline='always')(Dsymb)

        @njit(nogil=True, fastmath=True, inline='always')
        def wfunc(gp, gq, W):
            gp00 = gp[0,0]
            gp01 = gp[0,1]
            gp10 = gp[1,0]
            gp11 = gp[1,1]
            gq00 = gq[0,0]
            gq01 = gq[0,1]
            gq10 = gq[1,0]
            gq11 = gq[1,1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            return Wjfn(gp00, gp01, gp10, gp11,
                        gq00, gq01, gq10, gq11,
                        W00, W01, W10, W11).real

        @njit(nogil=True, fastmath=True, inline='always')
        def vfunc(gp, gq, W, V):
            gp00 = gp[0,0]
            gp01 = gp[0,1]
            gp10 = gp[1,0]
            gp11 = gp[1,1]
            gq00 = gq[0,0]
            gq01 = gq[0,1]
            gq10 = gq[1,0]
            gq11 = gq[1,1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            V00 = V[0]
            V01 = V[1]
            V10 = V[2]
            V11 = V[3]
            return Djfn(gp00, gp01, gp10, gp11,
                        gq00, gq01, gq10, gq11,
                        W00, W01, W10, W11,
                        V00, V01, V10, V11)

    elif jones.ndim == 5:  # DIAG mode
        W = W.subs(gp10, 0)
        W = W.subs(gp01, 0)
        W = W.subs(gq10, 0)
        W = W.subs(gq01, 0)
        C = C.subs(gp10, 0)
        C = C.subs(gp01, 0)
        C = C.subs(gq10, 0)
        C = C.subs(gq01, 0)

        if product == literal('I'):
            i = 0
        elif product == literal('Q'):
            i = 1
        elif product == literal('U'):
            i = 2
        elif product == literal('V'):
            i = 3
        else:
            raise ValueError(f"Unknown polarisation product {product}")

        Wsymb = lambdify((gp00, gp11,
                            gq00, gq11,
                            w0, w1, w2, w3),
                            sm.simplify(sm.expand(W[i,i])))
        Wjfn = njit(nogil=True, fastmath=True, inline='always')(Wsymb)


        Dsymb = lambdify((gp00, gp11,
                            gq00, gq11,
                            w0, w1, w2, w3,
                            v00, v01, v10, v11),
                            sm.simplify(sm.expand(C[i])))
        Djfn = njit(nogil=True, fastmath=True, inline='always')(Dsymb)

        @njit(nogil=True, fastmath=True, inline='always')
        def wfunc(gp, gq, W):
            gp00 = gp[0]
            gp11 = gp[1]
            gq00 = gq[0]
            gq11 = gq[1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            return Wjfn(gp00, gp11,
                        gq00, gq11,
                        W00, W01, W10, W11).real

        @njit(nogil=True, fastmath=True, inline='always')
        def vfunc(gp, gq, W, V):
            gp00 = gp[0]
            gp11 = gp[1]
            gq00 = gq[0]
            gq11 = gq[1]
            W00 = W[0]
            W01 = W[1]
            W10 = W[2]
            W11 = W[3]
            V00 = V[0]
            V01 = V[1]
            V10 = V[2]
            V11 = V[3]
            return Djfn(gp00, gp11,
                        gq00, gq11,
                        W00, W01, W10, W11,
                        V00, V01, V10, V11)

    else:
        raise ValueError(f"Jones term has incorrect number of dimensions")

    return vfunc, wfunc
