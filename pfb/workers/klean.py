# flake8: noqa
import os
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('KLEAN')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.klean["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.klean["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.klean)
def klean(**kw):
    '''
    Modified single-scale clean.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)

    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    logname = f'{str(opts.log_directory)}/klean_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)


    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url

    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')

    with ExitStack() as stack:
        ti = time.time()
        _klean(**opts)

        dds = xds_from_url(dds_name)

        from pfb.utils.fits import dds2fits, dds2fits_mfs


        if opts.fits_mfs or opts.fits:
            print(f"Writing fits files to {fits_oname}_{opts.suffix}", file=log)

        # convert to fits files
        if opts.fits_mfs:
            dds2fits_mfs(dds,
                         'RESIDUAL',
                         f'{fits_oname}_{opts.suffix}',
                         norm_wsum=True)
            dds2fits_mfs(dds,
                         'MODEL',
                         f'{fits_oname}_{opts.suffix}',
                         norm_wsum=False)

        if opts.fits_cubes:
            dds2fits(dds,
                     'RESIDUAL',
                     f'{fits_oname}_{opts.suffix}',
                     norm_wsum=True)
            dds2fits(dds,
                     'MODEL',
                     f'{fits_oname}_{opts.suffix}',
                     norm_wsum=False)

    print(f"All done after {time.time() - ti}s", file=log)


def _klean(ddsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from functools import partial
    import numpy as np
    from copy import copy, deepcopy
    import xarray as xr
    from pfb.utils.fits import set_wcs, save_fits, load_fits
    from pfb.deconv.clark import clark
    from pfb.utils.naming import xds_from_url
    from pfb.opt.pcg import pcg, pcg_psf
    from pfb.operators.gridding import compute_residual
    from scipy import ndimage
    from pfb.utils.misc import fitcleanbeam, fit_image_cube
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import xds_from_url
    from ducc0.misc import resize_thread_pool, thread_pool_size
    nthreads_tot = opts.nthreads_dask * opts.nvthreads
    resize_thread_pool(nthreads_tot)
    print(f'ducc0 max number of threads set to {thread_pool_size()}', file=log)

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
    dds_store = DaskMSStore(dds_name)
    dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')
    if ddsi is not None:
        dds = []
        for ds in ddsi:
            dds.append(ds.chunk({'row':-1,
                                 'chan':-1,
                                 'x':-1,
                                 'y':-1,
                                 'x_psf':-1,
                                 'y_psf':-1,
                                 'yo2':-1}))
    else:
        # are these sorted correctly?
        dds = xds_from_list(dds_list)

    nx, ny = dds[0].x.size, dds[0].y.size
    nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
    lastsize = ny_psf
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))

    nband = freq_out.size

    # stitch dirty/psf in apparent scale
    # drop_vars to avoid duplicates in memory
    output_type = dds[0].DIRTY.dtype
    if 'RESIDUAL' in dds[0]:
        residual = np.stack([ds.RESIDUAL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('RESIDUAL') for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values for ds in dds], axis=0)
        dds = [ds.drop_vars('DIRTY') for ds in dds]
    if 'MODEL' in dds[0]:
        model = np.stack([ds.MODEL.values for ds in dds], axis=0)
        dds = [ds.drop_vars('MODEL') for ds in dds]
    psf = np.stack([ds.PSF.values for ds in dds], axis=0)
    dds = [ds.drop_vars('PSF') for ds in dds]
    wsums = np.stack([ds.WSUM.values for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands


    wsum = np.sum(wsums)
    psf_mfs = np.sum(psf, axis=0)
    residual_mfs = np.sum(residual, axis=0)

    # for intermediary results (not currently written)
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    x0 = dds[0].x0
    y0 = dds[0].y0
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    if 'niters' in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        mask = load_fits(mask, dtype=output_type).squeeze()
        assert mask.shape == (nx, ny)
        mask = mask.astype(output_type)
        print('Using provided fits mask', file=log)
    else:
        mask = np.ones((nx, ny), dtype=output_type)

    # PCG related options for flux mop
    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack


    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count = 0
    if opts.threshold is None:
        threshold = opts.sigmathreshold * rms
    else:
        threshold = opts.threshold

    print(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(iter0, iter0 + opts.niter):
        print("Cleaning", file=log)
        modelp = deepcopy(model)
        x, status = clark(mask*residual, psf, psfhat, wsums/wsum,
                          threshold=threshold,
                          gamma=opts.gamma,
                          pf=opts.peak_factor,
                          maxit=opts.minor_maxit,
                          subpf=opts.sub_peak_factor,
                          submaxit=opts.subminor_maxit,
                          verbosity=opts.verbose,
                          report_freq=opts.report_freq,
                          sigmathreshold=opts.sigmathreshold,
                          nthreads=opts.nvthreads)
        model += x

        # write component model
        print(f"Writing model at iter {k+1} to "
              f"{basename}_{opts.suffix}_model.mds", file=log)
        try:
            coeffs, Ix, Iy, expr, params, texpr, fexpr = \
                fit_image_cube(time_out, freq_out[fsel], model[None, fsel, :, :],
                               wgt=wsums[None, fsel],
                               nbasisf=int(np.sum(fsel)),
                               method='Legendre')
            # save interpolated dataset
            data_vars = {
                'coefficients': (('par', 'comps'), coeffs),
            }
            coords = {
                'location_x': (('x',), Ix),
                'location_y': (('y',), Iy),
                # 'shape_x':,
                'params': (('par',), params),  # already converted to list
                'times': (('t',), time_out),  # to allow rendering to original grid
                'freqs': (('f',), freq_out)
            }
            attrs = {
                'spec': 'genesis',
                'cell_rad_x': cell_rad,
                'cell_rad_y': cell_rad,
                'npix_x': nx,
                'npix_y': ny,
                'texpr': texpr,
                'fexpr': fexpr,
                'center_x': dds[0].x0,
                'center_y': dds[0].y0,
                'ra': dds[0].ra,
                'dec': dds[0].dec,
                'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
                'parametrisation': expr  # already converted to str
            }

            coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
            coeff_dataset.to_zarr(f"{basename}_{opts.suffix}_model.mds",
                                  mode='w')
        except Exception as e:
            print(f"Exception {e} raised during model fit .", file=log)

        save_fits(np.mean(model[fsel], axis=0),
                  fits_oname + f'_{opts.suffix}_model_{k+1}.fits',
                  hdr_mfs)

        print(f'Computing residual', file=log)
        for ds_name, ds in zip(ds_list, dds):
            b = int(ds.bandid)
            resid = compute_residuals(ds_name,
                                      nx, ny,
                                      cell_rad, cell_rad,
                                      ds_name,
                                      model[b],
                                      x0=ds.x0, y0=ds.y0,
                                      nthreads=opts.nvthreads,
                                      epsilon=opts.epsilon,
                                      do_wgridding=opts.do_wgridding,
                                      double_accum=opts.double_accum)
            residual[b] = resid
        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs,
                  fits_oname + f'_{opts.suffix}_residual_{k+1}.fits',
                  hdr_mfs)

        # report rms where there aren't any model components
        rmsp = rms
        tmp_mask = ~np.any(model, axis=0)
        rms = np.std(residual_mfs[tmp_mask])
        rmax = np.abs(residual_mfs).max()

        # base this on rmax?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        if opts.threshold is None:
            threshold = opts.sigmathreshold * rms
        else:
            threshold = opts.threshold

        # trigger flux mop if clean has stalled, not converged or
        # we have reached the final iteration/threshold
        status |= k == iter0 + opts.niter-1
        status |= rmax <= threshold
        if opts.mop_flux and status:
            print(f"Mopping flux at iter {k+1}", file=log)
            mopmask = np.any(model, axis=0)
            if opts.dirosion:
                struct = ndimage.generate_binary_structure(2, opts.dirosion)
                mopmask = ndimage.binary_dilation(mopmask, structure=struct)
                mopmask = ndimage.binary_erosion(mopmask, structure=struct)
            x0 = np.zeros_like(x)
            x0[:, mopmask] = residual_mfs[mopmask]
            # TODO - applying mask as beam is wasteful
            mopmask = mopmask[None, :, :].astype(residual.dtype)
            x = pcg_psf(psfhat,
                        mopmask*residual,
                        x0,
                        mopmask,
                        lastsize,
                        opts.nvthreads,
                        rmax,  # used as sigmainv
                        cgopts)

            model += opts.mop_gamma*x

            print(f'Computing residual', file=log)
            for ds_name, ds in zip(ds_list, dds):
                b = int(ds.bandid)
                resid = compute_residuals(ds_name,
                                        nx, ny,
                                        cell_rad, cell_rad,
                                        ds_name,
                                        model[b],
                                        x0=ds.x0, y0=ds.y0,
                                        nthreads=opts.nvthreads,
                                        epsilon=opts.epsilon,
                                        do_wgridding=opts.do_wgridding,
                                        double_accum=opts.double_accum)
                residual[b] = resid
            residual /= wsum
            residual_mfs = np.sum(residual, axis=0)

            save_fits(residual_mfs,
                      f'{fits_oname}_{opts.suffix}_postmop{k}_residual_mfs.fits',
                      hdr_mfs)

            save_fits(np.mean(model[fsel], axis=0),
                      f'{fits_oname}_{opts.suffix}_postmop{k}_model_mfs.fits',
                      hdr_mfs)

            rmsp = rms
            tmp_mask = ~np.any(model, axis=0)
            rms = np.std(residual_mfs[tmp_mask])
            rmax = np.abs(residual_mfs).max()

            # base this on rmax?
            if rms < best_rms:
                best_rms = rms
                best_rmax = rmax
                best_model = model.copy()

            if opts.threshold is None:
                threshold = opts.sigmathreshold * rms
            else:
                threshold = opts.threshold

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, rms = {rms:.3e}",
              file=log)

        if rmax <= threshold:
            print("Terminating because final threshold has been reached",
                  file=log)
            break

        if rms > rmsp:
            diverge_count += 1
            if diverge_count > 3:
                print("Algorithm is diverging. Terminating.", file=log)
                break

    return
