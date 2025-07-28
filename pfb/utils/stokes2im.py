import ray
from functools import partial
import numpy as np
import numexpr as ne
import xarray as xr
from pfb.utils.weighting import (_compute_counts, counts_to_weights,
                                 weight_data, filter_extreme_counts)
from pfb.utils.stokes import stokes_funcs
from pfb.utils.fits import set_wcs, save_fits, add_beampars
from pfb.utils.misc import fitcleanbeam
from pfb.operators.gridder import wgridder_conventions
from casacore.quanta import quantity
from datetime import datetime, timezone
from ducc0.fft import good_size
from pfb.utils.astrometry import get_coordinates
from scipy.constants import c as lightspeed
import gc
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from astropy import units
from astropy.coordinates import SkyCoord
from africanus.coordinates import radec_to_lm
from katbeam import JimBeam
from scipy import ndimage
from reproject import reproject_interp
from astropy.wcs import WCS

@ray.remote
def compute_dataset(dset):
    """Ray remote function to compute dataset"""
    return dset

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
                wid=None):
    # serialization fails for these if we import them above
    from ducc0.misc import resize_thread_pool
    from ducc0.wgridder import vis2dirty
    
    resize_thread_pool(opts.nthreads)
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

    ds = ray.get(compute_dataset.remote(ds))
    jones = ray.get(compute_dataset.remote(jones))
    
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

    # weight *= ds.IMAGING_WEIGHT_SPECTRUM.values

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

    # compute lm coordinates of target
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
        coords0 = np.array((radec[0], radec[1]))
        lm0 = radec_to_lm(tcoords, coords0).squeeze()
        x0 = lm0[0]
        y0 = lm0[1]
    else:
        x0 = 0.0
        y0 = 0.0
        tra = radec[0]
        tdec = radec[1]

    ra_deg = np.rad2deg(tra)
    if ra_deg > 180:
        ra_deg -= 360
    dec_deg = np.rad2deg(tdec)

    # rephase if asked
    if opts.phase_dir is not None:
        new_ra, new_dec = opts.phase_dir.split(',')
        c = SkyCoord(new_ra, new_dec, frame='fk5', unit=(units.hourangle, units.deg))
        new_ra_rad = np.deg2rad(c.ra.value)
        new_dec_rad = np.deg2rad(c.dec.value)
        from pfb.utils.astrometry import (rephase, synthesize_uvw)
        data = rephase(data, uvw, freq, 
                       (new_ra_rad, new_dec_rad),
                       (tra, tdec), phasesign=-1)
        if model_vis is not None:
            model_vis = rephase(model_vis, uvw,freq, 
                                (new_ra_rad, new_dec_rad),
                                (tra, tdec), phasesign=-1)
        
        uvw = synthesize_uvw(antpos, time, ant1, ant2, (tra, tdec))
        # import ipdb; ipdb.set_trace()
        # uvw = uvwn

        # now for the beam interpolation/reprojection
        # load and interpolate beam to output frequency
        if opts.beam_model is None:
            pass
            raise RuntimeError("You have to provide a beam model when changing the phase center")
        bds = xr.open_zarr(opts.beam_model, chunks=None).interp(chan=[freq_out])
        l_beam = bds.l_beam.values
        m_beam = bds.m_beam.values
        freq = bds.chan.values
        corr = bds.corr.values
        beami = bds.BEAM.values

        # header for reference field
        hdr_ref = set_wcs(cell_deg, cell_deg, nx, ny, [tra, tdec],
                          freq_out, ms_time=time_out)
        wcs_ref = WCS(hdr_ref).dropaxis(-1).dropaxis(-1)
        # header for target field
        hdr_target = set_wcs(cell_deg, cell_deg, nx, ny,
                          [new_ra_rad, new_dec_rad],
                          freq_out, ms_time=time_out)
        wcs_target = WCS(hdr_target).dropaxis(-1).dropaxis(-1)

        pbeam, footprint = reproject_interp((beami, wcs_ref),
                                            wcs_target,
                                            shape_out=(nx, ny),
                                            block_size='auto',
                                            parallel=4)

        

    # vis_func, wgt_func = stokes_funcs(data, jones, opts.product, poltype, str(ncorr))
    # data, weight = weight_data_np(
    #         data, weight, flag, jones,
    #         tbin_idx, tbin_counts,
    #         ant1, ant2,
    #         len(opts.product),
    #         vis_func, 
    #         wgt_func)

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

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(x0, y0)
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0
    n = np.sqrt(1 - x0**2 - y0**2)
    freqfactor = -2j*np.pi*freq[None, :]/lightspeed
    psf_vis = np.exp(freqfactor*(signu*uvw[:, 0:1]*x0*signx +
                                 signv*uvw[:, 1:2]*y0*signy -
                                 uvw[:, 2:]*(n-1))).astype(complex_type)

    # TODO - polarisation parameters?
    if opts.inject_transients is not None:
        raise NotImplementedError("Transient injection not yet implemented")
        import yaml
        from pfb.utils.misc import dynamic_spectrum
        with open(opts.inject_transient, 'r') as f:
            transient_list = yaml.safe_load(f)
        for transient in transient_list:
            # parametric dspec model
            dspec = dynamic_spectrum(time, freq, transient_list[transient])

            # convert radec to lm
            tmp = transient_list[transient]['radec'].split(',')
            if len(tmp) != 2:
                raise ValueError(f"Invalid radec format {tmp} for transient {transient}")
            c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(units.hourangle, units.deg))
            ra_rad = np.deg2rad(c.ra.value)
            dec_rad = np.deg2rad(c.dec.value)
            tcoords=np.zeros((1,2))
            tcoords[0,0] = ra_rad
            tcoords[0,1] = dec_rad
            coords0 = np.array((radec[0], radec[1]))
            lm0t = radec_to_lm(tcoords, coords0).squeeze()
            x0t = lm0t[0]
            y0t = lm0t[1]

            # inject transient at x0t, y0t
            data += dspec * np.exp(freqfactor*(
                                 signu*uvw[:, 0:1]*x0t*signx +
                                 signv*uvw[:, 1:2]*y0t*signy -
                                 uvw[:, 2:]*(n-1)))

    # TODO - why do we need to cast here?
    data = data.transpose(2, 0, 1).astype(complex_type)
    weight = weight.transpose(2, 0, 1).astype(real_type)

    # the fact that uvw and freq are in double precision
    # complicates the numba implementation so we just cast
    # them to the appropriate type for this step.
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

        # # combine mirror image
        # # this should not be necessary
        # counts += counts[:, ::-1, ::-1]

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
    cell_deg = np.rad2deg(cell_rad)
    GaussPars = fitcleanbeam(psf, level=0.5, pixsize=cell_deg)

    rms = np.std(residual/wsum[:, None, None], axis=(1,2))

    if opts.natural_grad:
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

        x = cg(hess,
               residual/wsum[:, None, None],
               tol=opts.cg_tol,
               maxiter=opts.cg_maxit)[0]
    else:
        x = None

    unix_time = quantity(f'{time_out}s').to_unix_time()
    utc = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # set corr coords (removing duplicates and sorting)
    corr = "".join(dict.fromkeys(sorted(opts.product)))

    # if there is more than one polarisation product
    # we currently assume all have the same beam
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, [tra, tdec],
                  freq_out, GuassPar=GaussPars[0],
                  ms_time=time_out, ncorr=len(corr))

    # save outputs
    oname = f'spw{ddid:04d}_scan{scanid:04d}_band{bandid:04d}_time{timeid:04d}'
    if opts.output_format == 'zarr':
        tchunk = 1
        fchunk = 1
        xchunk = 128
        ychunk = 128
        cchunk = 1
        coords = {
            'FREQ': (('FREQ',), np.array([freq_out])),
            'TIME': (('TIME',), np.mean(utime, keepdims=True)),
            'STOKES': (('STOKES',), list(corr)),
            'X': (('X',), ra_deg + np.arange(nx//2, -(nx//2), -1) * cell_deg),
            'Y': (('Y',), dec_deg + np.arange(-(ny//2), ny//2) * cell_deg),
        }
        # X and Y are transposed for compatibility with breifast
        data_vars = {}
        residual /= wsum[:, None, None]
        residual = np.transpose(residual[:, None, :, :].astype(np.float32),
                                axes=(0, 1, 3, 2))
        data_vars['cube'] = (('STOKES', 'TIME', 'Y', 'X'), residual)
        if opts.psf_out:
            coords['X_PSF'] = (('X_PSF',), np.arange(nx_psf) * cell_deg)
            coords['Y_PSF'] = (('Y_PSF',), np.arange(ny_psf) * cell_deg)
            psf /= wsum[:, None, None]
            psf = np.transpose(psf[:, None, :, :].astype(np.float32),
                               axes=(0, 1, 3, 2))
            data_vars['psf'] = (('corr', 'TIME', 'Y_PSF', 'X_PSF'), psf)
        if x is not None:
            x = np.transpose(x[:, None, :, :].astype(np.float32),
                             axes=(0, 1, 3, 2))
            data_vars['xhat'] = (('STOKES', 'TIME', 'Y', 'X'), x)
        if opts.robustness is not None and opts.weight_grid_out:
            ic, ix, iy = np.where(counts > 0)
            wgt = np.zeros_like(counts)
            wgt[ic, ix, iy] = 1.0/counts[ic, ix, iy]
            coords['X_PAD'] = (('X_PAD',), np.arange(nx_pad) * cell_deg)
            coords['Y_PAD'] = (('Y_PAD',), np.arange(ny_pad) * cell_deg)
            wgt = np.transpose(wgt[:, None, :, :].astype(np.float32),
                               axes=(0, 1, 3, 2))
            data_vars['wgtgrid'] = (('STOKES', 'TIME', 'Y_PAD', 'X_PAD'), wgt)

        data_vars['rms'] = (('STOKES', 'TIME'), rms[:, None].astype(np.float32))
        data_vars['wsum'] = (('STOKES', 'TIME'), wsum[:, None].astype(np.float32))
        bmaj = np.array([gp[0] for gp in GaussPars], dtype=np.float32)
        bmin = np.array([gp[1] for gp in GaussPars], dtype=np.float32)
        # convert bpa to degrees
        bpa = np.array([gp[2]*180/np.pi for gp in GaussPars], dtype=np.float32)
        data_vars['psf_maj'] = (('STOKES', 'TIME'), bmaj[:, None])
        data_vars['psf_min'] = (('STOKES', 'TIME'), bmin[:, None])
        data_vars['psf_pa'] = (('STOKES', 'TIME'), bpa[:, None])

        # convert header to cards to maintain 
        # order when writing to and from zarr
        cards = []
        for key, value in hdr.items():
            cards.append((key, value))

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
            'header': cards
        }

        out_ds = xr.Dataset(
            data_vars,
            coords=coords,
            attrs=attrs)
        out_ds.to_zarr(f'{fds_store.url}/{oname}.zarr',
                       mode='w')
    elif opts.output_format == 'fits':
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
        if x is not None:
            save_fits(x,
                  f'{fds_store.full_path}/{oname}_x.fits', hdr)
    return 1


# 'SIMPLE': True,
# 'BITPIX': -32,
# 'NAXIS': 4,
# 'NAXIS1': 3072,
# 'NAXIS2': 3072,
# 'NAXIS3': 3644,
# 'NAXIS4': 2,
# 'EXTEND': True,
# 'BSCALE': 1.0,
# 'BZERO': 0.0,
# 'BUNIT': 'JY/BEAM',
# 'EQUINOX': 2000.0,
# 'LONPOLE': 180.0,
# 'BTYPE': 'Intensity',
# 'TELESCOP': 'MeerKAT',
# 'OBSERVER': 'Sarah Buchner',
# 'OBJECT': 'J2009-2026',
# 'ORIGIN': 'WSClean',
# 'CTYPE1': 'RA---SIN',
# 'CRPIX1': 1537.0,
# 'CRVAL1': -57.5966666666667,
# 'CDELT1': -0.000666666666666667,
# 'CUNIT1': 'deg',
# 'CTYPE2': 'DEC--SIN',
# 'CRPIX2': 1537.0,
# 'CRVAL2': -20.4461111111111,
# 'CDELT2': 0.000666666666666667,
# 'CUNIT2': 'deg',
# 'CTYPE3': 'INTEGRATION',
# 'CRPIX3': 1,
# 'CRVAL3': 0,
# 'CDELT3': 1,
# 'CUNIT3': 's',
# 'CTYPE4': 'STOKES',
# 'CRPIX4': 1,
# 'CRVAL4': np.int64(1),
# 'CDELT4': np.int64(3),
# 'CUNIT4': '',
# 'SPECSYS': 'TOPOCENT',
# 'DATE-OBS': '2021-06-20T19:46:24.8',
# 'WSCDATAC': 'DATA',
# 'WSCVDATE': '2025-02-07',
# 'WSCVERSI': '3.6',
# 'WSCWEIGH': "Briggs'(0)",
# 'WSCENVIS': 260048.182542329,
# 'WSCFIELD': 0.0,
# 'WSCGAIN': 0.1,
# 'WSCGKRNL': 7.0,
# 'WSCIMGWG': 1028445.46788349,
# 'WSCMAJOR': 0.0,
# 'WSCMGAIN': 1.0,
# 'WSCMINOR': 0.0,     
# 'WSCNEGCM': 1.0,
# 'WSCNEGST': 0.0,
# 'WSCNITER': 0.0, 
# 'WSCNORMF': 1028445.46788349, 
# 'WSCNVIS': 715461.0,
# 'WSCNWLAY': 1.0, 
# 'WSCVWSUM': 11443570.8579464,     
# 'FREQ0MHZ': 855.8955078125,
# 'FREQ1MHZ': 1711.8955078125, 
# 'OBSLABEL': 'L2',
# 'TIMESCAL': -7.899998664855957}