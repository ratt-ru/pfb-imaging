# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('MODEL2COMPS')


from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.model2comps)
def model2comps(**kw):
    '''
    Convert model in dds to components.
    '''
    opts = OmegaConf.create(kw)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    OmegaConf.set_struct(opts, True)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/model2comps_{timestamp}.log'
    pyscilog.log_to_file(logname)
    log.info(f'Logs will be written to {logname}')

    # TODO - prettier config printing
    log.info('Input Options:')
    for key in opts.keys():
        log.info('     %25s = %s' % (key, opts[key]))

    with ExitStack() as stack:
        ti = time.time()
        if opts.from_fits:
            _model2comps_fits(**opts)
        else:
            _model2comps(**opts)


    log.info(f"All done after {time.time() - ti}s")

def _model2comps(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import os
    import numpy as np
    from pfb.utils.naming import xds_from_url, xds_from_list
    from africanus.constants import c as lightspeed
    from pfb.utils.misc import fit_image_cube
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.misc import eval_coeffs_to_slice, norm_diff
    import xarray as xr
    import fsspec as fs
    from daskms.fsspec_store import DaskMSStore
    import json
    from casacore.quanta import quantity
    from pfb.utils.misc import eval_coeffs_to_slice

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds, dds_list = xds_from_url(dds_name)

    if opts.model_out is not None:
        coeff_name = opts.model_out
        fits_name = opts.model_out.rstrip('.mds') + '.fits'
    else:
        coeff_name = f'{basename}_{opts.suffix}_{opts.model_name.lower()}.mds'
        fits_name = f'{basename}_{opts.suffix}_{opts.model_name.lower()}.fits'

    mdsstore = DaskMSStore(coeff_name)
    if mdsstore.exists():
        if opts.overwrite:
            log.info(f"Overwriting {coeff_name}")
            mdsstore.rm(recursive=True)
        else:
            raise ValueError(f"{coeff_name} exists. "
                             "Set --overwrite to overwrite it. ")


    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    # get cube info
    nband = 0
    ntime = 0
    mfreqs = []
    mtimes = []
    for ds in dds:
        mfreqs.append(ds.freq_out)
        mtimes.append(ds.time_out)
    mfreqs = np.unique(np.array(mfreqs))
    mtimes = np.unique(np.array(mtimes))
    nband = mfreqs.size
    ntime = mtimes.size

    # stack cube
    nx = dds[0].x.size
    ny = dds[0].y.size
    x0 = dds[0].x0
    y0 = dds[0].y0

    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    wsums = np.zeros((ntime, nband), dtype=np.float64)
    for ds in dds:
        b = int(ds.bandid)
        t = int(ds.timeid)
        model[t, b] = getattr(ds, opts.model_name).values
        wsums[t, b] += ds.WSUM.values[0]

    if not opts.use_wsum:
        wsums[...] = 1.0
    else:
        # normalise so ridge param has more intuitive meaning
        wsums /= wsums.max()

    # render input model onto model grid
    # TODO - this is not perfectly flux conservative
    if opts.mds is not None:
        mdsi_store = DaskMSStore(opts.mds)
        mdsi = xr.open_zarr(mdsi_store.url)
        # we only want to load these once
        coeffsi = mdsi.coefficients.values
        locxi = mdsi.location_x.values
        locyi = mdsi.location_y.values
        paramsi = mdsi.params.values
        for t, mtime in enumerate(mtimes):
            for b, mfreq in enumerate(mfreqs):
                model[t, b] += eval_coeffs_to_slice(
                    mtime,
                    mfreq,
                    coeffsi,
                    locxi, locyi,
                    mdsi.parametrisation,
                    paramsi,
                    mdsi.texpr,
                    mdsi.fexpr,
                    mdsi.npix_x, mdsi.npix_y,
                    mdsi.cell_rad_x, mdsi.cell_rad_y,
                    mdsi.center_x, mdsi.center_y,
                    nx, ny,
                    cell_rad, cell_rad,
                    x0, y0
                )

    if not np.any(model):
        raise ValueError(f'Model is empty')
    radec = (dds[0].ra, dds[0].dec)

    if opts.out_freqs is not None:
        flow, fhigh, step = list(map(float, opts.out_freqs.split(':')))
        fsel = np.all(wsums > 0, axis=0)
        if flow < mfreqs.min():
            log.info(f"Linearly extrapolating to {flow:.3e}Hz")
            # linear extrapolation from first two non-null bands
            Ilow = np.argmax(fsel)  # first non-null index
            Ihigh = Ilow + 1
            while not fsel[Ihigh]:
                Ihigh += 1
            nudiff = mfreqs[Ihigh] - mfreqs[Ilow]
            slopes = (model[:, Ihigh] - model[:, Ilow])/nudiff
            intercepts = model[:, Ihigh] - slopes * mfreqs[Ihigh]
            mlow = slopes * flow + intercepts
            model = np.concatenate((mlow[:, None], model), axis=1)
            mfreqs = np.concatenate((np.array((flow,)), mfreqs))
            # TODO - duplicate first non-null value?
            wsums = np.concatenate((0.5*wsums[:, Ilow][:, None], wsums),
                                   axis=1)
            nband = mfreqs.size
        if fhigh > mfreqs.max():
            log.info(f"Linearly extrapolating to {fhigh:.3e}Hz")
            # linear extrapolation from last two non-null bands
            Ihigh = nband - np.argmax(fsel[::-1]) - 1  # last non-null index
            Ilow = Ihigh - 1
            while not fsel[Ilow]:
                Ilow -= 1
            nudiff = mfreqs[Ihigh] - mfreqs[Ilow]
            slopes = (model[:, Ihigh] - model[:, Ilow])/nudiff
            intercepts = model[:, Ihigh] - slopes * mfreqs[Ihigh]
            mhigh = slopes * fhigh + intercepts
            model = np.concatenate((model, mhigh[:, None]), axis=1)
            mfreqs = np.concatenate((mfreqs, np.array((fhigh,))))
            wsums = np.concatenate((wsums, 0.5*wsums[:, Ihigh][:, None]),
                                   axis=1)
            nband = mfreqs.size

    if opts.min_val is not None:
        model = np.where(model > opts.min_val, model, 0.0)

    if not np.any(model):
        raise ValueError(f'Model has no components above {opts.min_val}')

    if opts.nbasisf is None:
        nbasisf = nband-1
    else:
        nbasisf = opts.nbasisf

    nbasis = nbasisf
    log.info(f"Fitting {nband} bands with {nbasis} basis functions")
    try:
        coeffs, Ix, Iy, expr, params, texpr, fexpr = \
            fit_image_cube(mtimes, mfreqs, model,
                           wgt=wsums,
                           nbasisf=nbasisf,
                           method=opts.fit_mode,
                           sigmasq=opts.sigmasq)
    except np.linalg.LinAlgError as e:
        log.info(f"Exception {e} raised during fit ."
              f"Do you perhaps have empty sub-bands?"
              f"Decreasing nbasisf")
        raise e

    # save interpolated dataset
    data_vars = {
        'coefficients': (('par', 'comps'), coeffs),
    }
    coords = {
        'location_x': (('x',), Ix),
        'location_y': (('y',), Iy),
        # 'shape_x':,
        'params': (('par',), params),  # already converted to list
        'times': (('t',), mtimes),  # to allow rendering to original grid
        'freqs': (('f',), mfreqs)
    }
    attrs = {
        'spec': 'genesis',
        'cell_rad_x': cell_rad,
        'cell_rad_y': cell_rad,
        'npix_x': nx,
        'npix_y': ny,
        'texpr': texpr,
        'fexpr': fexpr,
        'center_x': x0,
        'center_y': y0,
        'flip_u': dds[0].flip_u,
        'flip_v': dds[0].flip_v,
        'flip_w': dds[0].flip_w,
        'ra': dds[0].ra,
        'dec': dds[0].dec,
        'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
        'parametrisation': expr  # already converted to str
    }


    coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
    log.info(f'Writing interpolated model to {coeff_name}')

    if opts.out_format == 'zarr':
        coeff_dataset.to_zarr(mdsstore.url)
    elif opts.out_format == 'json':
        coeff_dict = coeff_dataset.to_dict()
        with fs.open(mdsstore.url, 'w') as f:
            json.dump(coeff_dict, f)


    # interpolation error
    modelo = np.zeros((nband, nx, ny))
    for b in range(nband):
        modelo[b] = eval_coeffs_to_slice(mtimes[0],
                                         mfreqs[b],
                                         coeffs,
                                         Ix, Iy,
                                         expr,
                                         params,
                                         texpr,
                                         fexpr,
                                         nx, ny,
                                         cell_rad, cell_rad,
                                         x0, y0,
                                         nx, ny,
                                         cell_rad, cell_rad,
                                         x0, y0)

    eps = norm_diff(modelo, model[0])
    log.info(f"Fractional interpolation error is {eps:.3e}")


    # TODO - doesn't work with multiple fields
    # render model to cube
    if opts.out_freqs is not None:
        flow, fhigh, step = list(map(float, opts.out_freqs.split(':')))
        nbando = int((fhigh - flow)/step)
        log.info(f"Rendering model cube to {nbando} output bands")
        freq_out = np.linspace(flow, fhigh, nbando)
        ra = dds[0].ra
        dec  = dds[0].dec
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, [ra, dec],
                    freq_out, GuassPar=(1, 1, 0),  # fake for now
                    ms_time=mtimes[0])
        modelo = np.zeros((nbando, nx, ny))
        for b in range(nbando):
            modelo[b] = eval_coeffs_to_slice(mtimes[0],
                                            freq_out[b],
                                            coeffs,
                                            Ix, Iy,
                                            expr,
                                            params,
                                            texpr,
                                            fexpr,
                                            nx, ny,
                                            cell_rad, cell_rad,
                                            x0, y0,
                                            nx, ny,
                                            cell_rad, cell_rad,
                                            x0, y0)

        save_fits(modelo,
                  fits_name,
                  hdr,
                  overwrite=True)
        
def _model2comps_fits(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import os
    import numpy as np
    from africanus.constants import c as lightspeed
    from pfb.utils.misc import fit_image_cube
    from pfb.utils.fits import set_wcs, save_fits, load_fits
    from pfb.utils.misc import eval_coeffs_to_slice, norm_diff
    import xarray as xr
    import fsspec as fs
    from daskms.fsspec_store import DaskMSStore
    import json
    from casacore.quanta import quantity
    from pfb.utils.misc import eval_coeffs_to_slice
    from astropy.io import fits
    from glob import glob

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    if opts.model_out is not None:
        coeff_name = opts.model_out
        fits_name = opts.model_out.rstrip('.mds') + '.fits'
    else:
        coeff_name = f'{basename}_{opts.suffix}_{opts.model_name.lower()}.mds'
        fits_name = f'{basename}_{opts.suffix}_{opts.model_name.lower()}.fits'

    mdsstore = DaskMSStore(coeff_name)
    if mdsstore.exists():
        if opts.overwrite:
            log.info(f"Overwriting {coeff_name}")
            mdsstore.rm(recursive=True)
        else:
            raise ValueError(f"{coeff_name} exists. "
                             "Set --overwrite to overwrite it. ")


    images_list = sorted(
            glob(f"{opts.from_fits}-[0-9][0-9][0-9][0-9]-model.fits"),
            key=os.path.getctime)
    if len(images_list) == 0:
        raise ValueError(f"No images found at {opts.from_fits}")

    # get cube info
    nband = len(images_list)
    ntime = 1
    mfreqs = []
    model_list = []
    cellx_deg = None
    celly_deg = None
    nx = None
    ny = None
    ra = None
    dec = None
    wsums = []
    for image in images_list:
        log.info(f"Loading {image}")
        hdu = fits.open(image)
        model_list.append(hdu[0].data.squeeze().T)
        mfreqs.append(hdu[0].header['CRVAL3'])
        if 'WSCVWSUM' in hdu[0].header:
            wsums.append(hdu[0].header['WSCVWSUM'])
        elif 'WSCIMGWG' in hdu[0].header:
            wsums.append(hdu[0].header['WSCIMGWG'])
        else:
            wsums.append(1.0)
        if cellx_deg is None:
            cellx_deg = np.abs(hdu[0].header['CDELT2'])
        else:
            assert cellx_deg == np.abs(hdu[0].header['CDELT2'])
        if celly_deg is None:
            celly_deg = np.abs(hdu[0].header['CDELT1'])
        else:
            assert celly_deg == np.abs(hdu[0].header['CDELT1'])
        if ny is None:
            ny = hdu[0].header['NAXIS2']
        else:
            assert ny == hdu[0].header['NAXIS2']
        if nx is None:
            nx = hdu[0].header['NAXIS1']
        else:
            assert nx == hdu[0].header['NAXIS1']
        if ra is None:
            ra = hdu[0].header['CRVAL1']
        else:
            assert ra == hdu[0].header['CRVAL1']
        if dec is None:
            dec = hdu[0].header['CRVAL2']
        else:
            assert dec == hdu[0].header['CRVAL2']
        hdu.close()
    mfreqs = np.unique(np.array(mfreqs))
    assert mfreqs.size == nband
    for b in range(nband):
        log.info(f"Found {mfreqs[b]}Hz model at index {b}")
    assert cellx_deg == celly_deg
    cell_deg = cellx_deg
    cell_rad = np.deg2rad(cell_deg)
    x0 = 0.0
    y0 = 0.0
    wsums = np.array(wsums, dtype=np.float64)[None, :]
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    mtimes = np.ones((1,), dtype=np.float64)
    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    for b in range(nband):
        model[0, b] = model_list[b]

    if not opts.use_wsum:
        wsums[...] = 1.0
    else:
        # normalise so ridge param has more intuitive meaning
        wsums /= wsums.max()

    if not np.any(model):
        raise ValueError(f'Model is empty')
    radec = (ra, dec)

    if opts.out_freqs is not None:
        flow, fhigh, step = list(map(float, opts.out_freqs.split(':')))
        fsel = np.all(wsums > 0, axis=0)
        if flow < mfreqs.min():
            log.info(f"Linearly extrapolating to {flow:.3e}Hz")
            # linear extrapolation from first two non-null bands
            Ilow = np.argmax(fsel)  # first non-null index
            Ihigh = Ilow + 1
            while not fsel[Ihigh]:
                Ihigh += 1
            nudiff = mfreqs[Ihigh] - mfreqs[Ilow]
            slopes = (model[:, Ihigh] - model[:, Ilow])/nudiff
            intercepts = model[:, Ihigh] - slopes * mfreqs[Ihigh]
            mlow = slopes * flow + intercepts
            model = np.concatenate((mlow[:, None], model), axis=1)
            mfreqs = np.concatenate((np.array((flow,)), mfreqs))
            # TODO - duplicate first non-null value?
            wsums = np.concatenate((0.5*wsums[:, Ilow][:, None], wsums),
                                   axis=1)
            nband = mfreqs.size
        if fhigh > mfreqs.max():
            log.info(f"Linearly extrapolating to {fhigh:.3e}Hz")
            # linear extrapolation from last two non-null bands
            Ihigh = nband - np.argmax(fsel[::-1]) - 1  # last non-null index
            Ilow = Ihigh - 1
            while not fsel[Ilow]:
                Ilow -= 1
            nudiff = mfreqs[Ihigh] - mfreqs[Ilow]
            slopes = (model[:, Ihigh] - model[:, Ilow])/nudiff
            intercepts = model[:, Ihigh] - slopes * mfreqs[Ihigh]
            mhigh = slopes * fhigh + intercepts
            model = np.concatenate((model, mhigh[:, None]), axis=1)
            mfreqs = np.concatenate((mfreqs, np.array((fhigh,))))
            wsums = np.concatenate((wsums, 0.5*wsums[:, Ihigh][:, None]),
                                   axis=1)
            nband = mfreqs.size

    if opts.min_val is not None:
        model = np.where(model > opts.min_val, model, 0.0)

    if not np.any(model):
        raise ValueError(f'Model has no components above {opts.min_val}')

    if opts.nbasisf is None:
        nbasisf = nband-1
    else:
        nbasisf = opts.nbasisf

    nbasis = nbasisf
    log.info(f"Fitting {nband} bands with {nbasis} basis functions")
    try:
        coeffs, Ix, Iy, expr, params, texpr, fexpr = \
            fit_image_cube(mtimes, mfreqs, model,
                           wgt=wsums,
                           nbasisf=nbasisf,
                           method=opts.fit_mode,
                           sigmasq=opts.sigmasq)
    except np.linalg.LinAlgError as e:
        log.info(f"Exception {e} raised during fit ."
              f"Do you perhaps have empty sub-bands?"
              f"Decreasing nbasisf")
        raise e

    # save interpolated dataset
    data_vars = {
        'coefficients': (('par', 'comps'), coeffs),
    }
    coords = {
        'location_x': (('x',), Ix),
        'location_y': (('y',), Iy),
        # 'shape_x':,
        'params': (('par',), params),  # already converted to list
        'times': (('t',), mtimes),  # to allow rendering to original grid
        'freqs': (('f',), mfreqs)
    }
    attrs = {
        'spec': 'genesis',
        'cell_rad_x': cell_rad,
        'cell_rad_y': cell_rad,
        'npix_x': nx,
        'npix_y': ny,
        'texpr': texpr,
        'fexpr': fexpr,
        'center_x': x0,
        'center_y': y0,
        'flip_u': False,
        'flip_v': True,
        'flip_w': False,
        'ra': ra,
        'dec': dec,
        'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
        'parametrisation': expr  # already converted to str
    }


    coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
    log.info(f'Writing interpolated model to {coeff_name}')

    if opts.out_format == 'zarr':
        coeff_dataset.to_zarr(mdsstore.url)
    elif opts.out_format == 'json':
        coeff_dict = coeff_dataset.to_dict()
        with fs.open(mdsstore.url, 'w') as f:
            json.dump(coeff_dict, f)


    # interpolation error
    modelo = np.zeros((nband, nx, ny))
    for b in range(nband):
        modelo[b] = eval_coeffs_to_slice(mtimes[0],
                                         mfreqs[b],
                                         coeffs,
                                         Ix, Iy,
                                         expr,
                                         params,
                                         texpr,
                                         fexpr,
                                         nx, ny,
                                         cell_rad, cell_rad,
                                         x0, y0,
                                         nx, ny,
                                         cell_rad, cell_rad,
                                         x0, y0)

    eps = norm_diff(modelo, model[0])
    log.info(f"Fractional interpolation error is {eps:.3e}")


    # TODO - doesn't work with multiple fields
    # render model to cube
    if opts.out_freqs is not None:
        flow, fhigh, step = list(map(float, opts.out_freqs.split(':')))
        nbando = int((fhigh - flow)/step)
        log.info(f"Rendering model cube to {nbando} output bands")
        freq_out = np.linspace(flow, fhigh, nbando)
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, [ra, dec],
                      freq_out, GuassPar=(1, 1, 0),  # fake for now
                      ms_time=mtimes[0])
        modelo = np.zeros((nbando, nx, ny))
        for b in range(nbando):
            modelo[b] = eval_coeffs_to_slice(mtimes[0],
                                            freq_out[b],
                                            coeffs,
                                            Ix, Iy,
                                            expr,
                                            params,
                                            texpr,
                                            fexpr,
                                            nx, ny,
                                            cell_rad, cell_rad,
                                            x0, y0,
                                            nx, ny,
                                            cell_rad, cell_rad,
                                            x0, y0)
    else:  # we just write it back at input frequencies as a sanity check
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, [ra, dec],
                      mfreqs, GuassPar=(1, 1, 0), casambm=False)
                      
    save_fits(modelo[:, None, :, :],
              fits_name,
              hdr,
              overwrite=True)
