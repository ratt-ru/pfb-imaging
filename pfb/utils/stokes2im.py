import numpy as np
import numexpr as ne
from numba import literally
import dask
from distributed import get_client, worker_client
import xarray as xr
import dask
from pfb.utils.stokes import stokes_funcs
from pfb.utils.weighting import (_compute_counts, _counts_to_weights,
                                 _weight_data, _filter_extreme_counts)
from pfb.utils.misc import eval_coeffs_to_slice
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from casacore.quanta import quantity
from datetime import datetime

# for old style vs new style warnings
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def single_stokes_image(
                    data=None,
                    data2=None,
                    operator=None,
                    ant1=None,
                    ant2=None,
                    uvw=None,
                    frow=None,
                    flag=None,
                    sigma=None,
                    weight=None,
                    mds=None,
                    jones=None,
                    opts=None,
                    nx=None,
                    ny=None,
                    freq=None,
                    cell_rad=None,
                    utime=None,
                    tbin_idx=None,
                    tbin_counts=None,
                    radec=None,
                    antpos=None,
                    poltype=None,
                    fieldid=None,
                    ddid=None,
                    scanid=None,
                    fds_store=None,
                    bandid=None,
                    timeid=None,
                    wid=None):


    with worker_client() as client:
        (data, data2, ant1, ant2, uvw, frow, flag, sigma, weight,
        jones) = client.compute([data, data2, ant1, ant2, uvw, frow,
                                flag, sigma, weight, jones],
                                sync=True,
                                workers=wid)

    # (data, data2, ant1, ant2, uvw, frow, flag, sigma, weight,
    #     jones) = dask.compute(data, data2, ant1, ant2, uvw, frow,
    #                         flag, sigma, weight, jones)

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    nrow, nchan, ncorr = data.shape
    ntime = utime.size
    nant = antpos.shape[0]
    time_out = np.mean(utime)

    freq_out = np.mean(freq)
    freq_min = freq.min()
    freq_max = freq.max()

    # MS may contain auto-correlations
    if frow is not None:
        frow = frow | (ant1 == ant2)
    else:
        frow = (ant1 == ant2)

    if flag is not None:
        flag = np.any(flag, axis=2)
        flag = np.logical_or(flag, frow[:, None])
    else:
        flag = np.broadcast_to(frow[:, None], (nrow, nchan))

    if sigma is not None:
        weight = ne.evaluate('1.0/sigma**2')
    elif weight is not None:
        if weight.ndim == 2:
            weight = np.broadcast_to(weight[:, None, :],
                                    (nrow, nchan, ncorr))
    else:
        weight = np.ones((nrow, nchan, ncorr),
                        dtype=real_type)

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if data2 is not None:
        try:
            assert (operator=='+' or operator=='-')
        except Exception as e:
            raise e
        data2 = data2.astype(complex_type)
        data = ne.evaluate(f'data {operator} data2',
                        out=data,
                        casting='same_kind')

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
        if jones_ncorr == 2:
            jout = 'rafdx'
        elif jones_ncorr == 4:
            jout = 'rafdxx'
            jones = jones.reshape(ntime, nant, nchan, 1, 2, 2)
        else:
            raise ValueError("Incorrect number of correlations of "
                            f"{jones_ncorr} for product {opts.product}")
    else:
        jones = np.ones((ntime, nant, nchan, 1, 2),
                        dtype=complex_type)
        jout = 'rafdx'

    # compute lm coordinates of target
    if opts.target is not None:
        tmp = opts.target.split(',')
        if len(tmp) == 1 and tmp[0] == opts.target:
            obs_time = time_out
            tra, tdec = get_coordinates(obs_time, target=opts.target)
        else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(u.hourangle, u.deg))
            tra = np.deg2rad(c.ra.value)
            tdec = np.deg2rad(c.dec.value)

        tcoords=np.zeros((1,2))
        tcoords[0,0] = tra
        tcoords[0,1] = tdec
        coords0 = np.array((ds.ra, ds.dec))
        lm0 = radec_to_lm(tcoords, coords0).squeeze()
        # LB - why the negative?
        x0 = -lm0[0]
        y0 = -lm0[1]
    else:
        x0 = 0.0
        y0 = 0.0
        tra = radec[0]
        tdec = radec[1]


    # we currently need this extra loop through the data because
    # we don't have access to the grid
    vis, wgt = _weight_data(data, weight, flag, jones,
                            tbin_idx, tbin_counts,
                            ant1, ant2,
                            literally(poltype),
                            literally(opts.product),
                            literally(str(ncorr)))

    mask = (~flag).astype(np.uint8)


    # TODO - we may want to apply this to bigger chunks of data
    if opts.robustness is not None:
        counts = _compute_counts(
                uvw,
                freq,
                mask,
                nx,
                ny,
                cell_rad,
                cell_rad,
                np.float64,  # same type as uvw
                ngrid=opts.nvthreads)
        # get rid of artificially high weights corresponding to
        # nearly empty cells
        if opts.filter_extreme_counts:
            counts = _filter_extreme_counts(counts,
                                            nbox=opts.filter_nbox,
                                            nlevel=opts.filter_level)

        counts = counts.sum(axis=0)


    if mds is not None:
        # mds = xr.open_zarr(mds).compute()

        model = eval_coeffs_to_slice(
            time_out,
            freq_out,
            mds.coefficients.values,
            mds.location_x.values,
            mds.location_y.values,
            mds.parametrisation,
            mds.params.values,
            mds.texpr,
            mds.fexpr,
            mds.npix_x, mds.npix_y,
            mds.cell_rad_x, mds.cell_rad_y,
            mds.center_x, mds.center_y,
            # TODO - currently needs to be the same, need FFT interpolation
            mds.npix_x, mds.npix_y,
            mds.cell_rad_x, mds.cell_rad_y,
            mds.center_x, mds.center_y,
        )

        # do not apply weights in this direction
        # do not change model resolution
        model_vis = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=model,
            pixsize_x=mds.cell_rad_x,
            pixsize_y=mds.cell_rad_y,
            center_x=mds.center_x,
            center_y=mds.center_y,
            epsilon=opts.epsilon,
            do_wgridding=opts.do_wgridding,
            flip_v=False,
            nthreads=opts.nvthreads,
            divide_by_n=False,
            sigma_min=1.1, sigma_max=3.0,
            verbosity=0)

        ne.evaluate('(vis-model_vis)*mask', out=vis)

        if opts.l2reweight_dof:
            ressq = (vis*vis.conj()).real
            wcount = mask.sum()
            if wcount:
                ovar = ressq.sum()/wcount  # use 67% quantile?
                wgt = (l2reweight_dof + 1)/(l2reweight_dof + ressq/ovar)/ovar
            else:
                wgt = None

    if opts.robustness is not None:
        if counts is None:
            raise ValueError('counts are None but robustness specified. '
                            'This is probably a bug!')
        imwgt = _counts_to_weights(
            counts,
            uvw,
            freq,
            nx, ny,
            cell_rad, cell_rad,
            opts.robustness)
        if wgt is not None:
            wgt *= imwgt
        else:
            wgt = imwgt

    wsum = wgt[~flag].sum()

    residual = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=wgt,
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cell_rad, pixsize_y=cell_rad,
        center_x=x0, center_y=y0,
        epsilon=opts.epsilon,
        flip_v=False,  # hardcoded for now
        do_wgridding=opts.do_wgridding,
        divide_by_n=False,  # hardcoded for now
        nthreads=opts.nvthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=opts.double_accum,
        verbosity=0)

    rms = np.std(residual)

    data_vars = {}
    data_vars['RESIDUAL'] = (('x', 'y'), residual.astype(np.float32))

    coords = {'chan': (('chan',), freq),
            'time': (('time',), utime),
    }

    unix_time = quantity(f'{time_out}s').to_unix_time()
    utc = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')


    # TODO - provide time and freq centroids
    attrs = {
        'ra' : tra,
        'dec': tdec,
        'x0': x0,
        'y0': y0,
        'cell_rad': cell_rad,
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
        'wsum':wsum,
        'rms':rms
    }

    out_ds = xr.Dataset(data_vars,  #coords=coords,
                    attrs=attrs)
    oname = f'spw{ddid:04d}_scan{scanid:04d}_band{bandid:04d}_time{timeid:04d}'
    with worker_client() as client:
        out_store = out_ds.to_zarr(f'{fds_store}/{oname}.zarr',
                                   compute=True)
    return out_store
