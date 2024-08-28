import numpy as np
import numexpr as ne
from numba import literally
from distributed import worker_client
import xarray as xr
from uuid import uuid4
from pfb.utils.stokes import stokes_funcs
from pfb.utils.weighting import (_compute_counts, counts_to_weights,
                                 weight_data, filter_extreme_counts)
from pfb.utils.misc import eval_coeffs_to_slice
from pfb.utils.fits import set_wcs, save_fits
from pfb.operators.gridder import wgridder_conventions
from ducc0.wgridder import vis2dirty
from casacore.quanta import quantity
from datetime import datetime
from ducc0.fft import c2r, r2c, good_size
from africanus.constants import c as lightspeed
import gc
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def single_stokes_image(
                    dc1=None,
                    dc2=None,
                    operator=None,
                    ds=None,
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
                    fds_store=None,
                    bandid=None,
                    timeid=None,
                    wid=None):

    fieldid = ds.FIELD_ID
    ddid = ds.DATA_DESC_ID
    scanid = ds.SCAN_NUMBER
    oname = f'ms{fieldid:04d}_spw{ddid:04d}_scan{scanid:04d}' \
            f'_band{bandid:04d}_time{timeid:04d}'

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    with worker_client() as client:
        (ds, jones) = client.compute([ds,
                                      jones],
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
    ds = ds.drop_vars('FLAG_ROW')

    # flag if any of the correlations flagged
    flag = np.any(flag, axis=2)
    # combine flag and frow
    flag = np.logical_or(flag, frow[:, None])

    # we rely on this to check the number of output bands and
    # to ensure we don't end up with fully flagged chunks
    if flag.all():
        return 1

    nrow, nchan, ncorr = data.shape

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

    if opts.model_column is not None:
        model_vis = getattr(ds, opts.model_column).values.astype(complex_type)
        ds = ds.drop(opts.model_column)
        if opts.product.lower() == 'i':
            model_vis = (model_vis[:, :, 0] + model_vis[:, :, -1])/2.0
        else:
            raise NotImplementedError(f'Model subtraction not supported for product {opts.product}')

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

    # check that there are no missing antennas
    ant1u = np.unique(ant1)
    ant2u = np.unique(ant2)
    try:
        assert (ant1u[1:] - ant1u[0:-1] == 1).all()
        assert (ant2u[1:] - ant2u[0:-1] == 1).all()
    except Exception as e:
        raise NotImplementedError('You seem to have missing antennas. '
                                  'This is not currently supported.')

    # check that antpos gives the correct size table
    antmax = np.maximum(ant1.max(), ant2.max()) + 1
    try:
        assert antmax == nant
    except Exception as e:
        raise ValueError('Inconsistent ANTENNA table. '
                         'Shape does not match max number of antennas '
                         'as inferred from ant1 and ant2. '
                         f'Table size is {antpos.shape} but got {antmax}. '
                         f'{oname}')

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
        x0 = lm0[0]
        y0 = lm0[1]
    else:
        x0 = 0.0
        y0 = 0.0
        tra = radec[0]
        tdec = radec[1]


    # we currently need this extra loop through the data because
    # we don't have access to the grid
    data, weight = weight_data(data, weight, flag, jones,
                            tbin_idx, tbin_counts,
                            ant1, ant2,
                            literally(poltype),
                            literally(opts.product),
                            literally(str(ncorr)))

    mask = (~flag).astype(np.uint8)

    # TODO - this subtraction would be better to do inside weight_data
    if opts.model_column is not None:
        ne.evaluate('(data-model_vis)*mask', out=data)

    if opts.l2reweight_dof:
        # data should contain residual_vis at this point
        ressq = (data*data.conj()).real * mask
        wcount = mask.sum()
        if wcount:
            ovar = ressq.sum()/wcount  # use 67% quantile?
            weight = (opts.l2reweight_dof + 1)/(opts.l2reweight_dof + ressq/ovar)/ovar
        else:
            weight = None

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(x0, y0)

    if opts.robustness is not None:
        counts = _compute_counts(uvw,
                                 freq,
                                 mask,
                                 weight,
                                 nx, ny,
                                 cell_rad, cell_rad,
                                 uvw.dtype,
                                 ngrid=1,
                                 usign=1.0 if flip_u else -1.0,
                                 vsign=1.0 if flip_v else -1.0)

        imwgt = counts_to_weights(
            counts,
            uvw,
            freq,
            nx, ny,
            cell_rad, cell_rad,
            opts.robustness,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0)
        if weight is not None:
            weight *= imwgt
        else:
            weight = imwgt

    wsum = weight[~flag].sum()

    residual = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=data,
        wgt=weight,
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cell_rad, pixsize_y=cell_rad,
        center_x=x0, center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=opts.epsilon,
        do_wgridding=opts.do_wgridding,
        divide_by_n=True,  # no rephasing or smooth beam so do it here
        nthreads=opts.nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=opts.double_accum,
        verbosity=0)

    rms = np.std(residual/wsum)

    if opts.natural_grad:
        from pfb.opt.pcg import pcg
        from pfb.operators.hessian import _hessian_impl
        from functools import partial

        hess = partial(_hessian_impl,
                       uvw=uvw,
                       weight=weight,
                       vis_mask=mask,
                       freq=freq,
                       beam=None,
                       cell=cell_rad,
                       x0=x0,
                       y0=y0,
                       do_wgridding=opts.do_wgridding,
                       epsilon=opts.epsilon,
                       double_accum=opts.double_accum,
                       nthreads=opts.nthreads,
                       sigmainvsq=opts.sigmainvsq*wsum,
                       wsum=1.0)  # we haven't normalised residual

        x = pcg(hess,
                residual,
                x0=np.zeros_like(residual),
                # M=precond,
                minit=1,
                tol=opts.cg_tol,
                maxit=opts.cg_maxit,
                verbosity=opts.cg_verbose,
                report_freq=opts.cg_report_freq,
                backtrack=False,
                return_resid=False)
    else:
        x = None

    unix_time = quantity(f'{time_out}s').to_unix_time()
    utc = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')

    cell_deg = np.rad2deg(cell_rad)
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, [tra, tdec],
                  freq_out, GuassPar=(1, 1, 0),  # fake for now
                  ms_time=time_out)

    oname = f'spw{ddid:04d}_scan{scanid:04d}_band{bandid:04d}_time{timeid:04d}'
    if opts.output_format == 'zarr':
        data_vars = {}
        data_vars['RESIDUAL'] = (('x', 'y'), residual.astype(np.float32))
        if x is not None:
            data_vars['NATGRAD'] = (('x', 'y'), residual.astype(np.float32))

        coords = {'chan': (('chan',), freq),
                'time': (('time',), utime),
        }


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
            'rms':rms,
            # 'header': hdr.items()
        }

        out_ds = xr.Dataset(data_vars,  #coords=coords,
                        attrs=attrs)
        out_ds.to_zarr(f'{fds_store.url}/{oname}.zarr',
                                mode='w')
    elif opts.output_format == 'fits':
        save_fits(residual/wsum, f'{fds_store.full_path}/{oname}.fits', hdr)
    return 1
