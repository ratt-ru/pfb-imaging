# flake8: noqa
from contextlib import ExitStack
from pathlib import Path
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FLUXMOP')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fluxmop["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.fluxmop["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fluxmop)
def fluxmop(**kw):
    '''
    Forward step aka flux mop.

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    logname = f'{str(ldir)}/fluxmop_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.naming import set_output_names
    basedir, oname, fits_output_folder = set_output_names(opts.output_filename,
                                                          opts.product,
                                                          opts.fits_output_folder)

    basename = f'{basedir}/{oname}'
    fits_oname = f'{fits_output_folder}/{oname}'

    opts.output_filename = basename
    dds_name = f'{basename}_{opts.suffix}.dds'
    dds_store = DaskMSStore(dds_name)
    opts.fits_output_folder = fits_output_folder
    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        _fluxmop(**opts)

    print("All done here.", file=log)

def _fluxmop(ddsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import numexpr as ne
    import xarray as xr
    import dask
    import dask.array as da
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, dds2fits_mfs, dds2fits, set_wcs, save_fits
    from pfb.utils.misc import init_mask, dds2cubes
    from pfb.operators.hessian import hessian_xds, hessian_psf_cube
    from pfb.operators.psf import psf_convolve_cube
    from pfb.opt.pcg import pcg_dds
    from ducc0.misc import resize_thread_pool, thread_pool_size
    from ducc0.fft import c2c
    iFs = np.fft.ifftshift
    Fs = np.fft.fftshift
    nthreads_tot = opts.nthreads_dask * opts.nvthreads
    resize_thread_pool(nthreads_tot)
    print(f'ducc0 max number of threads set to {thread_pool_size()}', file=log)

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
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
        dds = xds_from_zarr(dds_name)
        chunks = {'x':-1,
                  'y':-1}
        # These may or may not be present
        if 'PSF' in dds[0]:
            chunks['x_psf'] = -1
            chunks['y_psf'] = -1
            chunks['yo2'] = -1
        if 'WEIGHT' in dds[0]:
            chunks['row'] = -1
            chunks['chan'] = -1
        for i, ds in enumerate(dds):
            dds[i] = ds.chunk(chunks)


    # stitch image space data products
    output_type = dds[0].DIRTY.dtype
    print("Combining slices into cubes", file=log)
    dirty, model, residual, psf, psfhat, beam, wsums, _ = dds2cubes(
                                                               dds,
                                                               opts.nband,
                                                               apparent=False,
                                                               dual=False,
                                                               modelname=opts.model_name,
                                                               residname=opts.residual_name)
    fsel = wsums > 0
    wsum = np.sum(wsums)
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    else:
        residual_mfs = np.sum(residual, axis=0)


    if psf is not None:
        psf_mfs = np.sum(psf, axis=0)
        assert (psf_mfs.max() - 1.0) < 2*opts.epsilon
        lastsize = dds[0].y_psf.size
    freq_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
    freq_out = np.unique(np.array(freq_out))
    nband = opts.nband
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    if model is not None:
        model_mfs = np.mean(model[fsel], axis=0)
    else:
        model_mfs = np.zeros((nx,ny), dtype=output_type)
        model = np.zeros((nband, nx,ny), dtype=output_type)

    # set up vis space Hessian for computing the residual
    # TODO - how to apply beam externally per ds
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['do_wgridding'] = opts.do_wgridding
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0,
                   mask=np.ones((nx, ny), dtype=output_type),
                   compute=True, use_beam=False)

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        if opts.mask=='model':
            mask = np.any(model > opts.min_model, axis=0)
            assert mask.shape == (nx, ny)
            mask = mask.astype(output_type)
            print('Using model > 0 to create mask', file=log)
        else:
            mask = load_fits(opts.mask, dtype=output_type).squeeze()
            assert mask.shape == (nx, ny)
            if opts.or_mask_with_model:
                print("Combining model with input mask", file=log)
                mask = np.logical_or(mask>0, model_mfs>0).astype(output_type)


            mask = mask.astype(output_type)
            print('Using provided fits mask', file=log)
            if opts.zero_model_outside_mask and not opts.or_mask_with_model:
                model[:, mask<1] = 0
                print("Recomputing residual since asked to zero model", file=log)
                convimage = hess(model)
                ne.evaluate('dirty - convimage', out=residual,
                            casting='same_kind')
                ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                            casting='same_kind')
                save_fits(np.mean(model[fsel], axis=0),
                  basename + f'_{opts.suffix}_model_mfs_zeroed.fits',
                  hdr_mfs)
                save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_residual_mfs_zeroed.fits',
                  hdr_mfs)


    else:
        mask = np.ones((nx, ny), dtype=output_type)
        print('Caution - No mask is being applied', file=log)


    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    # cgopts['backtrack'] = opts.backtrack

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Initial peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)

    print("Solving for update", file=log)
    # x0 = np.zeros((nband, nx, ny), dtype=output_type)
    x0 = precondo(j)
    save_fits(np.mean(x0, axis=0),
              basename + f'_{opts.suffix}_x0.fits',
              hdr_mfs)
    for ds in dds:
        x, r, x0 = pcg_dds(hess_pcg,
                           opts.sigmainvsq,
                           opts.sigma,
                           mask=mask,
                           do_wgridding=opts.do_wgridding,
                           epsilon=opts.epsilon,
                           double_accum=opts.double_accum,
                           nthreads=opts.nvthreads,
                           **cgopts)

    modelp = model.copy()
    model += opts.gamma * update

    residualp = residual.copy()
    print("Getting residual", file=log)
    convimage = hess(model)
    ne.evaluate('dirty - convimage', out=residual,
                casting='same_kind')
    ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                casting='same_kind')

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Final peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)

    print("Updating results", file=log)
    dds_out = []
    for ds in dds:
        b = ds.bandid
        r = da.from_array(residual[b]*wsum)
        rp = da.from_array(residualp[b]*wsum)
        m = da.from_array(model[b])
        mp = da.from_array(modelp[b])
        u = da.from_array(update[b])
        # keep previous results in case of failure
        ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                              'RESIDUALP': (('x', 'y'), rp),
                              'MODEL': (('x', 'y'), m),
                              'MODELP': (('x', 'y'), mp),
                              'UPDATE': (('x', 'y'), u)})
        dds_out.append(ds_out)
    writes = xds_to_zarr(dds_out, dds_name,
                         columns=('RESIDUAL', 'RESIDUALP', 'MODEL', 'MODELP', 'UPDATE'),
                         rechunk=True)

    dask.compute(writes)

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})



    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds,
                                    'RESIDUAL',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds,
                                    'MODEL',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=False))
        fitsout.append(dds2fits_mfs(dds,
                                    'UPDATE',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds,
                                'RESIDUAL',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=True))
        fitsout.append(dds2fits(dds,
                                'MODEL',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=False))
        fitsout.append(dds2fits(dds,
                                'UPDATE',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)
