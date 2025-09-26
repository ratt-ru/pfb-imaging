import ray
from functools import partial
import numpy as np
import numexpr as ne
import xarray as xr
from pfb.utils.weighting import (_compute_counts, counts_to_weights,
                                 weight_data, filter_extreme_counts)
from pfb.utils.beam import reproject_and_interp_beam
from pfb.utils.fits import set_wcs, save_fits
from pfb.utils.misc import fitcleanbeam
from pfb.operators.gridder import wgridder_conventions
from casacore.quanta import quantity
from datetime import datetime, timezone
from ducc0.fft import good_size
from pfb.utils.astrometry import get_coordinates
from scipy.constants import c as lightspeed
import gc
from astropy import units
from astropy.coordinates import SkyCoord
from africanus.coordinates import radec_to_lm

iFs = np.fft.ifftshift
Fs = np.fft.fftshift

@ray.remote
def batch_stokes_image(
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
                msid=None,
                attrs=None,
                integrations_per_image=None,
                all_times=None,
                time_slice=None):
    # load chunk
    ds.load(scheduler='sync')

    # slice out rows corresponding to single images and submit compute task
    ntime = utime.size
    tasks = []
    for t0 in range(0, ntime, integrations_per_image):
        nmax = np.minimum(ntime, t0+integrations_per_image)
        It = slice(t0, nmax)
        ridx = tbin_idx[It]
        rcnts = tbin_counts[It]
        Irow = slice(ridx[0], ridx[-1]+rcnts[-1])
        dsi = ds[{'row': Irow}]

        if jones is not None:
            jones_slice = jones[It]
        else:
            jones_slice = None

        task = stokes_image.remote(
                    dc1=dc1,
                    dc2=dc2,
                    operator=operator,
                    ds=ds,
                    jones=jones_slice,
                    opts=opts,
                    nx=nx,
                    ny=ny,
                    freq=freq,
                    cell_rad=cell_rad,
                    utime=utime[It],
                    tbin_idx=tbin_idx[It],
                    tbin_counts=tbin_counts[It],
                    radec=radec,
                    antpos=antpos,
                    poltype=poltype,
                    fds_store=fds_store,
                    bandid=bandid,
                    timeid=timeid,
                    msid=msid,
                    attrs=attrs)
        
        tasks.append(task)

    # wait for all tasks to finish and get result
    dso = ray.get(tasks)

    # manually create the region (region='auto' fails?)
    region = {
        'STOKES': slice(0, len(opts.product)),
        'FREQ': slice(bandid, bandid+1),
        'TIME': time_slice,
        'Y': slice(0, dso[0].Y.size),
        'X': slice(0, dso[0].X.size)
    }

    # if the output is a dataset we stack and write the output
    if isinstance(dso[0], xr.Dataset):
        dso = xr.concat(dso, dim='TIME')
        dso.to_zarr(fds_store.url, region=region)
    

    return timeid, bandid
    

@ray.remote
def stokes_image(
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
                msid=None,
                attrs=None):
    # serialization fails for these if we import them above
    from ducc0.misc import resize_thread_pool
    from ducc0.wgridder import vis2dirty
    
    resize_thread_pool(opts.nthreads)
    fieldid = ds.FIELD_ID
    ddid = ds.DATA_DESC_ID
    scanid = ds.SCAN_NUMBER
    oname = f'ms{msid:04d}_fid{fieldid:04d}_spw{ddid:04d}_scan{scanid:04d}' \
            f'_band{bandid:04d}_time{timeid:04d}'

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    # LB - is this the correct way to do this? 
    # we don't want it to end up in the distributed object store 
    ds = ds.load(scheduler='sync')
    if jones is not None:
        # we do it this way to force using synchronous scheduler
        jones = jones.load(scheduler='sync').values
    
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

    # combine flag and frow
    flag = np.logical_or(flag, frow[:, None, None])

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
    else:
        model_vis = None
    # this seems to help with memory consumption
    # note the ds.drop_vars above
    del ds
    gc.collect()

    nrow, nchan, ncorr = data.shape
    ntime = utime.size
    nant = antpos.shape[0]
    time_out = np.mean(utime)

    freq_out = np.mean(freq)

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
    allants = np.unique(np.concatenate((ant1u, ant2u)))

    # check that antpos gives the correct size table
    antmax = allants.size
    if opts.check_ants:
        try:
            assert antmax == nant
        except Exception as e:
            raise ValueError('Inconsistent ANTENNA table. '
                            'Shape does not match max number of antennas '
                            'as inferred from ant1 and ant2. '
                            f'Table size is {antpos.shape} but got {antmax}. '
                            f'{oname}')

    # relabel antennas by index
    # this only works because allants is sorted in ascending order
    for a, ant in enumerate(allants):
        ant1 = np.where(ant1==ant, a, ant1)
        ant2 = np.where(ant2==ant, a, ant2)

    cell_deg = np.rad2deg(cell_rad)

    # make sure ra is in (0, 2pi)
    # need a copy since write only
    radec = radec.copy()
    if radec[0] < 0:
        radec[0] += 2*np.pi
    elif radec[0] > 2*np.pi:
        radec[0] -= 2*np.pi

    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0, 0)
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0
    freqfactor = -2j*np.pi*freq[None, :]/lightspeed

    # rephase if asked
    if opts.phase_dir is not None:
        new_ra, new_dec = opts.phase_dir.split(',')
        c = SkyCoord(new_ra, new_dec, frame='fk5', unit=(units.hourangle, units.deg))
        new_ra_rad = np.deg2rad(c.ra.value)
        new_dec_rad = np.deg2rad(c.dec.value)
        radec_new = np.array((new_ra_rad, new_dec_rad))

        # print(c.ra.value, c.dec.value, cell_deg)
        
        from pfb.utils.astrometry import synthesize_uvw
        uvw_new = synthesize_uvw(antpos, time, ant1, ant2, radec_new)
        # uo = uvw[:, 0:1] * freq[None, :]/lightspeed
        # vo = uvw[:, 1:2] * freq[None, :]/lightspeed
        wo = uvw[:, 2:] * freq[None, :]/lightspeed
        # un = uvw_new[:, 0:1] * freq[None, :]/lightspeed
        # vn = uvw_new[:, 1:2] * freq[None, :]/lightspeed
        wn = uvw_new[:, 2:] * freq[None, :]/lightspeed
        
        # TODO - why is this incorrect?
        # from africanus.coordinates import radec_to_lmn
        # l, m, n = radec_to_lmn(radec_new[None, :], radec)[0]
        # # original phase direction is [0,0,1]
        # dl = l
        # dm = m
        # dn = n-1
        # phase2 = uo * dl
        # phase2 += vo * dm
        # phase2 += wo * dn
        # tmp2 = np.exp(-2j*np.pi*phase2)

        # TODO - this copies chgcentre but not sure why it gives
        # better results than computing the phase with lmn differences
        phase = 2j*np.pi*(wn - wo)
        data *= np.exp(-phase)[:, :, None]
        if model_vis is not None:
            model_vis *= np.exp(-phase)[:, :, None]
        
        uvw = uvw_new
    else:
        radec_new = radec

    if opts.beam_model is not None:
        # should we compute a weighted mean over freq instead of interpolating here? 
        bds = xr.open_zarr(opts.beam_model, chunks=None).interp(chan=[freq_out])
        l_beam = bds.l_beam.values
        m_beam = bds.m_beam.values
        # the beam is in feed plane direction cosine coordinates
        # we need to flip the beam upside down because of the beam orientation
        # see https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/index.html
        # shape is (corr, chan, X, Y) -> squeeze out freq
        beam = bds.BEAM.values[:, 0, :, :]
        # are the MdV beams transmissive or receptive?
        # reshape for feed and spatial rotations
        beam = beam.reshape(2, 2, l_beam.size, m_beam.size)
        cell_deg_in = l_beam[1] - l_beam[0]
        pbeam = reproject_and_interp_beam(beam, time, antpos,
                                          radec, radec_new,
                                          cell_deg_in, cell_deg, nx, ny,
                                          poltype, opts.product,
                                          weight=weight, nthreads=opts.nthreads)
        
        # this is a hack to get the images to align
        pbeam = np.transpose(pbeam.astype(np.float32),
                             axes=(0, 2, 1))
        pbeam = pbeam[:, ::-1, :]
        
    else:
        pbeam = np.ones((len(opts.product), nx, ny), dtype=real_type)
    
    
    # compute lm coordinates of target if requested
    if opts.target is not None:
        tmp = opts.target.split(',')
        if len(tmp) == 1 and tmp[0] == opts.target:
            obs_time = time_out
            tra, tdec = get_coordinates(obs_time, target=opts.target)
        else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
            c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(units.hourangle, units.deg))
            tra = np.deg2rad(c.ra.value)
            tdec = np.deg2rad(c.dec.value)

        tcoords=np.zeros((1,2))
        tcoords[0,0] = tra
        tcoords[0,1] = tdec
        lm0 = radec_to_lm(tcoords, radec_new[None, :]).squeeze()
        # flip for wgridder conventions
        x0 = -lm0[0]
        y0 = -lm0[1]
    else:
        x0 = 0.0
        y0 = 0.0
        tra = radec_new[0]
        tdec = radec_new[1]

    ra_deg = np.rad2deg(tra)
    dec_deg = np.rad2deg(tdec)
    # why was this required?
    # if ra_deg > 180:
    #     ra_deg -= 360

    # we currently need this extra loop through the data because
    # we don't have access to the grid
    data, weight = weight_data(
            data, weight, flag, jones,
            tbin_idx, tbin_counts,
            ant1, ant2,
            poltype,
            opts.product,
            str(ncorr))

    # flag if any correlation is flagged
    flag = flag.any(axis=-1)
    mask = (~flag).astype(np.uint8)

    # TODO - this subtraction would be better to do inside weight_data
    if opts.model_column is not None:
        ne.evaluate('(data-model_vis)*mask', out=data)

    if opts.l2_reweight_dof:
        # data should contain residual_vis at this point
        ressq = (data*data.conj()).real * mask
        wcount = mask.sum()
        if wcount:
            ovar = ressq.sum()/wcount  # use 67% quantile?
            weight = (opts.l2_reweight_dof + 1)/(opts.l2_reweight_dof + ressq/ovar)/ovar
        else:
            weight = None

    
    n = np.sqrt(1 - x0**2 - y0**2)
    psf_vis = np.exp(freqfactor*(signu*uvw[:, 0:1]*x0*signx +
                                 signv*uvw[:, 1:2]*y0*signy -
                                 uvw[:, 2:]*(n-1))).astype(complex_type)

    # TODO - polarisation parameters and handle Stokes axis more elegantly
    # TODO - add beam application to injected transients
    # Should this go before weight_data?
    if opts.inject_transients is not None:
        transient_name = opts.inject_transients.removesuffix('yaml') + 'zarr'
        transient_ds = xr.open_zarr(transient_name, chunks=None)
        all_times = transient_ds.TIME.values
        all_freqs = transient_ds.FREQ.values
        names = transient_ds.names
        ras = transient_ds.ras
        decs = transient_ds.decs
        for name, ra, dec in zip(names, ras, decs):
            ra_rad = np.deg2rad(ra)
            dec_rad = np.deg2rad(dec)
            tcoords=np.zeros((1,2))
            tcoords[0,0] = ra_rad
            tcoords[0,1] = dec_rad
            coords0 = np.array((radec[0], radec[1]))
            lm0t = radec_to_lm(tcoords, coords0).squeeze()
            x0t = lm0t[0]
            y0t = lm0t[1]
            n0t = np.sqrt(1 - x0t**2 - y0t**2)

            # these are the profiles on the full domain so interpolate
            tprofile = getattr(transient_ds, f'{name}_time_profile').values
            fprofile = getattr(transient_ds, f'{name}_freq_profile').values
            
            tprofile = np.interp(time, all_times, tprofile)  # note reorder to ms times
            fprofile = np.interp(freq, all_freqs, fprofile)

            # outer product gives dynamic spectrum
            dspec = tprofile[:, None] * fprofile[None, :]

            # inject transient at x0t, y0t and convert to complex values
            dspec = dspec * np.exp(freqfactor*(
                                 signu*uvw[:, 0:1]*x0t*signx +
                                 signv*uvw[:, 1:2]*y0t*signy -
                                 uvw[:, 2:]*(n0t-1)))
            
            # currently Stokes I only
            data[:, :, 0] += dspec

    # TODO - why do we need to cast here?
    data = data.transpose(2, 0, 1).astype(complex_type)
    weight = weight.transpose(2, 0, 1).astype(real_type)
    if opts.robustness is not None:
        # we need to compute the weights on the padded grid
        # but we don't have control over the optimal gridding
        # parameters so assume a minimum
        nx_pad = int(np.ceil(opts.min_padding*nx))
        if nx_pad%2:
            nx_pad += 1
        ny_pad = int(np.ceil(opts.min_padding*ny))
        if ny_pad%2:
            ny_pad += 1
        counts = _compute_counts(uvw,
                                 freq,
                                 mask,
                                 weight,
                                 nx_pad, ny_pad,
                                 cell_rad, cell_rad,
                                 real_type,
                                 k=0,
                                 ngrid=1,
                                 usign=1.0 if flip_u else -1.0,
                                 vsign=1.0 if flip_v else -1.0)

        counts = filter_extreme_counts(counts,
                                       level=opts.filter_counts_level)

        weight = counts_to_weights(
            counts,
            uvw,
            freq,
            weight,
            mask,
            nx_pad, ny_pad,
            cell_rad, cell_rad,
            opts.robustness,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0)

    psf_min_size = 128
    if opts.psf_relative_size is None:
        nx_psf = psf_min_size
        ny_psf = psf_min_size
    else:
        nx_psf = good_size(int(nx * opts.psf_relative_size))
        while nx_psf % 2:
            nx_psf = good_size(nx_psf + 1)
        ny_psf = good_size(int(ny * opts.psf_relative_size))
        while ny_psf % 2:
            ny_psf = good_size(ny_psf + 1)
        if nx_psf < psf_min_size or ny_psf < psf_min_size:
            nx_psf = psf_min_size
            ny_psf = psf_min_size

    nstokes = weight.shape[0]
    wsum = np.zeros(nstokes)
    residual = np.zeros((nstokes, nx, ny), dtype=real_type)
    psf = np.zeros((nstokes, nx_psf, ny_psf), dtype=real_type)
    # TODO - the wgridder doesn't check if wgridding is actually
    # required and always makes a minimum of nsupp number of wplanes
    # where nsupp is the gridding kernel support. We could check this
    # with pfb.utils.misc.wplanar if we can figure out the relationship
    # between epsilon on the wplanar threshold parameter. 
    for c in range(nstokes):
        wsum[c] = weight[c, ~flag].sum()
        vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=data[c],
            wgt=weight[c],
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
            sigma_min=opts.min_padding, sigma_max=3.0,
            double_precision_accumulation=opts.double_accum,
            verbosity=0,
            dirty=residual[c])

        vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=psf_vis,
            wgt=weight[c],
            mask=mask,
            npix_x=nx_psf, npix_y=ny_psf,
            pixsize_x=cell_rad, pixsize_y=cell_rad,
            center_x=x0, center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=opts.epsilon,
            do_wgridding=opts.do_wgridding,
            divide_by_n=True,  # no rephasing or smooth beam so do it here
            nthreads=opts.nthreads,
            sigma_min=opts.min_padding, sigma_max=3.0,
            double_precision_accumulation=opts.double_accum,
            verbosity=0,
            dirty=psf[c])

    # these will be in units of pixels
    GaussPars = fitcleanbeam(psf, level=0.5, pixsize=cell_deg)

    if opts.natural_grad:
        # TODO - add beam application
        from pfb.operators.hessian import hessian_jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import cg
        iFs = jnp.fft.ifftshift

        abspsf = jnp.abs(jnp.fft.rfft2(
                                    iFs(psf/wsum[:, None, None], axes=(1,2)),
                                    axes=(1, 2),
                                    norm='backward'))

        hess = partial(hessian_jax,
                       nx, ny,
                       2*nx, 2*ny,
                       opts.eta,
                       abspsf)

        residual = cg(hess,
               residual/wsum[:, None, None],
               tol=opts.cg_tol,
               maxiter=opts.cg_maxit)[0]
        
    else:
        residual /= wsum[:, None, None]
        residual *= pbeam / (pbeam**2 + opts.eta)

    rms = np.std(residual, axis=(1,2))

    unix_time = quantity(f'{time_out}s').to_unix_time()
    utc = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # set corr coords (removing duplicates and sorting)
    corr = "".join(dict.fromkeys(sorted(opts.product)))

    out_ras = ra_deg + np.arange(nx//2, -(nx//2), -1) * cell_deg
    out_decs = dec_deg + np.arange(-(ny//2), ny//2) * cell_deg
    out_ras = np.round(out_ras, decimals=12)
    out_decs = np.round(out_decs, decimals=12)

    # save outputs
    if opts.output_format == 'zarr':
        coords = {
            'FREQ': (('FREQ',), np.array([freq_out])),
            'TIME': (('TIME',), np.mean(utime, keepdims=True)),
            'STOKES': (('STOKES',), list(corr)),
            'X': (('X',), out_ras),
            'Y': (('Y',), out_decs),
        }
        # X and Y are transposed for compatibility with breifast
        data_vars = {}
        residual = np.transpose(residual.astype(np.float32),
                                axes=(0, 2, 1))
        data_vars['cube'] = (('STOKES', 'FREQ', 'TIME', 'Y', 'X'),
                             residual[:, None, None, :, :])
        if opts.psf_out:
            if opts.psf_relative_size == 1:
                xpsf = "X"
                ypsf = "Y"
            else:
                xpsf = "X_PSF"
                ypsf = "Y_PSF"
                coords['X_PSF'] = (('X_PSF',), ra_deg + np.arange(nx_psf//2, -(nx_psf//2), -1) * cell_deg)
                coords['Y_PSF'] = (('Y_PSF',), dec_deg + np.arange(-(ny_psf//2), ny_psf//2) * cell_deg)
            psf /= wsum[:, None, None]
            psf = np.transpose(psf.astype(np.float32),
                               axes=(0, 2, 1))
            data_vars['psf'] = (('STOKES', 'FREQ', 'TIME', ypsf, xpsf),
                                psf[:, None, None, :, :])
        
        if opts.robustness is not None and opts.weight_grid_out:
            ic, ix, iy = np.where(counts > 0)
            wgt = np.zeros_like(counts)
            wgt[ic, ix, iy] = 1.0/counts[ic, ix, iy]
            coords['X_PAD'] = (('X_PAD',), np.arange(nx_pad) * cell_deg)
            coords['Y_PAD'] = (('Y_PAD',), np.arange(ny_pad) * cell_deg)
            wgt = np.transpose(wgt.astype(np.float32),
                               axes=(0, 2, 1))
            data_vars['wgtgrid'] = (('STOKES', 'FREQ', 'TIME', 'Y_PAD', 'X_PAD'),
                                    wgt[:, None, None, :, :])
        
        data_vars['weight'] = (('STOKES','FREQ','TIME'), wsum[:, None, None])
        
        if opts.beam_model is not None:
            weight = pbeam**2 + opts.eta
            weight = np.transpose(weight.astype(np.float32),
                                  axes=(0, 2, 1))
            data_vars['beam_weight'] = (('STOKES', 'FREQ', 'TIME', 'Y', 'X'),
                                    weight[:, None, None, :, :])
        
        data_vars['rms'] = (('STOKES', 'FREQ', 'TIME'), rms[:, None, None].astype(np.float32))
        nonzero = wsum > 0
        data_vars['nonzero'] = (('STOKES', 'FREQ', 'TIME'), nonzero[:, None, None])
        bmaj = np.array([gp[0] for gp in GaussPars], dtype=np.float32)
        bmin = np.array([gp[1] for gp in GaussPars], dtype=np.float32)
        # convert bpa to degrees
        bpa = np.array([gp[2]*180/np.pi for gp in GaussPars], dtype=np.float32)
        data_vars['psf_maj'] = (('STOKES', 'FREQ', 'TIME'), bmaj[:, None, None])
        data_vars['psf_min'] = (('STOKES', 'FREQ', 'TIME'), bmin[:, None, None])
        data_vars['psf_pa'] = (('STOKES', 'FREQ', 'TIME'), bpa[:, None, None])

        if attrs is None:
            attrs = {
                'ra' : tra,
                'dec': tdec,
                'x0': x0,
                'y0': y0,
                'cell_rad': cell_rad,
                'fieldid': fieldid,
                'ddid': ddid,
                'scanid': scanid,
                'bandid': bandid,
                'timeid': timeid,
                'robustness': opts.robustness,
                'utc': utc,
            }

        out_ds = xr.Dataset(
            data_vars,
            coords=coords,
            attrs=attrs)
        if opts.stack:
            return out_ds
        else:
            out_ds.to_zarr(f'{fds_store.url}/{oname}.zarr', mode='w')
    elif opts.output_format == 'fits':
        # if there is more than one polarisation product
        # we currently assume all have the same beam
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, [tra, tdec],
                    freq_out, GuassPar=GaussPars[0],
                    ms_time=time_out, ncorr=len(corr))

        hdr['STOKES'] = corr
        save_fits(residual/wsum[:, None, None],
                  f'{fds_store.full_path}/{oname}_image.fits', hdr)
        if opts.robustness is not None and opts.weight_grid_out:
            ic, ix, iy = np.where(counts > 0)
            wgt = np.zeros_like(counts)
            wgt[ic, ix, iy] = 1.0/counts[ic, ix, iy]
            # TODO - add mirror image
            save_fits(wgt,
                    f'{fds_store.full_path}/{oname}_weight.fits', hdr)

        if opts.psf_out:
            hdr_psf = set_wcs(cell_deg, cell_deg, nx_psf, ny_psf, [tra, tdec],
                  freq_out, GuassPar=GaussPars[0],  # fake for now
                  ms_time=time_out)
            hdr_psf['STOKES'] = corr
            save_fits(psf/wsum[:, None, None],
                  f'{fds_store.full_path}/{oname}_psf.fits', hdr_psf)

        if opts.beam_model is not None:
            # weight = np.transpose(weight.astype(np.float32),
            #                      axes=(0, 2, 1))
            # weight = weight[:, ::-1, :]
            weight = pbeam**2 + opts.eta
            save_fits(weight,
                  f'{fds_store.full_path}/{oname}_weight.fits', hdr)
    return 1
