import numpy as np
import numexpr as ne
import xarray as xr
from numba import njit, prange, literally
from dask.graph_manipulation import clone
from distributed import get_client, worker_client
import dask.array as da
from xarray import Dataset
from pfb.operators.gridder import vis2im
# from quartical.utils.numba import coerce_literal
from operator import getitem
from pfb.utils.beam import interp_beam
from pfb.utils.misc import weight_from_sigma, combine_columns
import dask
from quartical.utils.dask import Blocker
from pfb.utils.stokes import stokes_funcs
from pfb.utils.weighting import weight_data
from uuid import uuid4
import gc
from casacore.quanta import quantity
from datetime import datetime


def single_stokes(
                dc1=None,
                dc2=None,
                operator=None,
                ds=None,
                jones=None,
                opts=None,
                freq=None,
                chan_width=None,
                utime=None,
                tbin_idx=None,
                tbin_counts=None,
                radec=None,
                antpos=None,
                poltype=None,
                fieldid=None,
                ddid=None,
                scanid=None,
                xds_store=None,
                bandid=None,
                timeid=None,
                msid=None,
                wid=None,
                max_freq=None,
                uv_max=None):

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    with worker_client() as client:
        (ds, jones) = client.compute([clone(ds),
                                      clone(jones)],
                                     sync=True,
                                     workers=wid,
                                     key='read-'+uuid4().hex)
    data = getattr(ds, dc1).values
    ds = ds.drop_vars(dc1)
    if dc2 is not None:
        try:
            assert (operator=='+' or operator=='-')
        except Exception as e:
            raise e
        ne.evaluate(f'data {operator} data2',
                    local_dict={'data': data,
                                'data2': getattr(ds, dc2).values},
                    out=data,
                    casting='same_kind')
        ds = ds.drop_vars(dc2)

    time = ds.TIME.values
    ds = ds.drop_vars('TIME')
    interval = ds.INTERVAL.values
    ds = ds.drop_vars('INTERVAL')
    ant1 = ds.ANTENNA1.values
    ds = ds.drop_vars('ANTENNA1')
    ant2 = ds.ANTENNA2.values
    ds = ds.drop_vars('ANTENNA2')
    uvw = ds.UVW.values
    ds = ds.drop_vars('UVW')
    flag = ds.FLAG.values
    ds = ds.drop_vars('FLAG')
    # MS may contain auto-correlations
    frow = ds.FLAG_ROW.values | (ant1 == ant2)


    nrow, nchan, ncorr = data.shape
    ds = ds.drop_vars('FLAG_ROW')
    flag = np.any(flag, axis=2)
    if opts.sigma_column is not None:
        weight = ne.evaluate('1.0/sigma**2',
                             local_dict={'sigma': getattr(ds, opts.sigma_column).values})
        ds = ds.drop_vars(opts.sigma_column)
    elif opts.weight_column is not None:
        weight = getattr(ds, opts.weight_column).values
        ds = ds.drop_vars(opts.weight_column)
    else:
        weight = np.ones((nrow, nchan, ncorr),
                         dtype=real_type)

    # this seems to help with memory consumption
    # note the ds.drop_vars above
    del ds
    gc.collect()

    nrow, nchan, ncorr = data.shape
    ntime = utime.size
    nant = antpos.shape[0]
    time_out = np.mean(utime)

    freq_out = np.mean(freq)
    freq_min = freq.min()
    freq_max = freq.max()

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    if jones is not None:
        if jones.dtype != complex_type:
            jones = jones.astype(complex_type)
        # qcal has chan and ant axes reversed compared to pfb implementation
        jones = np.swapaxes(jones, 1, 2)
        # data are not 2x2 so we need separate labels
        # for jones correlations and data/weight correlations
        # reshape to dispatch with overload
        jones_ncorr = jones.shape[-1]
        if jones_ncorr == 4:
            jones = jones.reshape(ntime, nant, nchan, 1, 2, 2)
        elif jones_ncorr == 2:
            pass
        else:
            raise ValueError("Incorrect number of correlations of "
                            f"{jones_ncorr} for product {opts.product}")
    else:
        jones = np.ones((ntime, nant, nchan, 1, 2),
                        dtype=complex_type)

    # we currently need this extra loop through the data because
    # we don't have access to the grid
    data, weight = weight_data(data, weight, flag, jones,
                            tbin_idx, tbin_counts,
                            ant1, ant2,
                            literally(poltype),
                            literally(opts.product),
                            literally(str(ncorr)))

    # do after weight_data otherwise mappings need to be recomputed
    # dropped flagged rows
    mrow = ~frow
    data = data[mrow]
    time = time[mrow]
    interval = interval[mrow]
    ant1 = ant1[mrow]
    ant2 = ant2[mrow]
    uvw = uvw[mrow]
    flag = flag[mrow]
    weight = weight[mrow]

    gc.collect()

    # simple average over channels
    if opts.chan_average > 1:
        from africanus.averaging import time_and_channel

        res = time_and_channel(
                    time,
                    interval,
                    ant1,
                    ant2,
                    uvw=uvw,
                    flag=flag[:, :, None],
                    weight_spectrum=weight[:, :, None],
                    visibilities=data[:, :, None],
                    chan_freq=freq,
                    chan_width=chan_width,
                    time_bin_secs=1e-15,
                    chan_bin_size=opts.chan_average)

        data = res.visibilities[:, :, 0]
        weight = res.weight_spectrum[:, :, 0]
        flag = res.flag[:, :, 0]
        freq = res.chan_freq
        chan_width = res.chan_width
        uvw = res.uvw
        nrow, nchan = data.shape

    if opts.bda_decorr < 1:
        from africanus.averaging import bda

        res = bda(time,
                  interval,
                  ant1, ant2,
                  uvw=uvw,
                  flag=flag[:, :, None],
                  weight_spectrum=weight[:, :, None],
                  visibilities=data[:, :, None],
                  chan_freq=freq,
                  chan_width=chan_width,
                  decorrelation=opts.bda_decorr,
                  min_nchan=freq.size)

        offsets = res.offsets
        uvw = res.uvw[offsets[:-1], :]
        weight = res.weight_spectrum.reshape(-1, nchan).squeeze()
        data = res.visibilities.reshape(-1, nchan).squeeze()
        flag = res.flag.reshape(-1, nchan).squeeze()

    mask = (~flag).astype(np.uint8)


    # TODO - just a fake beam for now
    fov = 10  # max fov in degrees
    npix = 512
    cell_deg = fov/npix
    l_beam = -(fov/2) + np.arange(npix)
    m_beam = -(fov/2) + np.arange(npix)
    beam = np.ones((npix, npix), dtype=real_type)

    # set after averaging
    # negate w for wgridder bug
    # uvw[:, 2] = -uvw[:, 2]
    data_vars = {}
    data_vars['VIS'] = (('row', 'chan'), data)
    data_vars['WEIGHT'] = (('row', 'chan'), weight)
    data_vars['MASK'] = (('row', 'chan'), mask)
    data_vars['UVW'] = (('row', 'three'), uvw)
    data_vars['FREQ'] = (('chan',), freq)
    data_vars['BEAM'] = (('l_beam','m_beam'), beam)



    coords = {'chan': (('chan',), freq),
            #   'time': (('time',), utime),
              'l_beam': (('l_beam',), l_beam),
              'm_beam': (('m_beam',), m_beam)
    }

    unix_time = quantity(f'{time_out}s').to_unix_time()
    utc = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')

    # TODO - provide time and freq centroids
    attrs = {
        'ra' : radec[0],
        'dec': radec[1],
        'fieldid': fieldid,
        'ddid': ddid,
        'scanid': scanid,
        'freq_out': freq_out,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'bandid': bandid,
        'time_out': time_out,
        'time_min': utime.min(),
        'time_max': utime.max(),
        'timeid': timeid,
        'product': opts.product,
        'utc': utc,
        'max_freq': max_freq,
        'uv_max': uv_max,
    }

    out_ds = xr.Dataset(data_vars, coords=coords,
                        attrs=attrs)
    oname = f'ms{msid:04d}_spw{ddid:04d}_scan{scanid:04d}' \
            f'_band{bandid:04d}_time{timeid:04d}'
    out_store = out_ds.to_zarr(f'{xds_store}/{oname}.zarr',
                                mode='w')
    return out_store, time_out, freq_out
