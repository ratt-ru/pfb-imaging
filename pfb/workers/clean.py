# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('CLEAN')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.clean["inputs"].keys():
    defaults[key] = schema.clean["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.clean)
def clean(**kw):
    '''
    Single-scale clean.

    The algorithm always acts on the average apparent dirty image and PSF
    provided by xds. This means there is only ever a single dirty image
    and PSF and the algorithm only provides an approximate apparent model
    that is compatible with them. The intrinsic model can be obtained using
    the forward worker.

    Two variants of single scale clean are currently implemented viz.
    Hogbom and Clark.

    Hogbom is the vanilla clean implementation, the full PSF will be
    subtracted at every iteration.

    Clark clean defines an adaptive mask that changes between iterations.
    The mask is defined by all pixels that are above sub-peak-factor * Imax
    where Imax is the current maximum in the MFS residual. A sub-minor cycle
    is performed only within this mask i.e. peak finding and PSF subtraction
    is confined to the mask until the residual within the mask decreases to
    sub-peak-factor * Imax. At the end of the sub-minor cycle an approximate
    residual is computed as

    IR -= PSF.convolve(model)

    with no mask in place. The mask is then recomputed and the sub-minor cycle
    repeated until the residual reaches peak-factor * Imax0 where Imax0 is the
    peak in the residual at the outset of the minor cycle.

    At the end of each minor cycle we recompute the residual using

    IR = R.H W (V - R x) = ID - R.H W R x

    where ID is the dirty image, R and R.H are degridding and gridding
    operators respectively and W are the effective weights i.e. the weights
    after image weighting, applying calibration solutions and taking the
    weighted sum over correlations. This is usually called a major cycle
    but we get away from explicitly loading in the visibilities by writing
    the residual in terms of the dirty image and an application of the
    Hessian.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'clean_{timestamp}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _clean(**opts)

def _clean(**kw):
    opts = OmegaConf.create(kw)
    # always combine over ds during cleaning
    opts['mean_ds'] = True
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.utils.misc import setup_image_data
    from pfb.deconv.hogbom import hogbom
    from pfb.deconv.clark import clark
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.operators.hessian import hessian
    from pfb.opt.pcg import pcg, pcg_psf
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube, _hessian_reg_psf
    from scipy import ndimage
    from copy import copy

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan':-1})
    if opts.memory_greedy:
        dds = dask.persist(dds)[0]
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    nx_psf, ny_psf = dds[0].nx_psf, dds[0].ny_psf

    # stitch dirty/psf in apparent scale
    output_type = dds[0].DIRTY.dtype
    dirty, residual, wsum, psf, psfhat, _ = setup_image_data(dds,
                                                             opts,
                                                             apparent=True,
                                                             log=log)
    psf_mfs = np.sum(psf, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*opts.epsilon
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    else:
        residual_mfs = np.sum(residual, axis=0)
    try:
        model = getattr(mds, opts.model_name).values
    except Exception as e:
        raise e

    # for intermediary results (not currently written)
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = mds.freq.data
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    # always clean in apparent scale so no beam
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0, mask=np.ones_like(dirty_mfs),
                   compute=True, use_beam=False)

    # set up image space Hessian
    npad_xl = (nx_psf - nx)//2
    npad_xr = nx_psf - nx - npad_xl
    npad_yl = (ny_psf - ny)//2
    npad_yr = ny_psf - ny - npad_yl
    padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    lastsize = ny + np.sum(padding[-1])
    psfopts = {}
    psfopts['nthreads'] = opts.nvthreads
    psfopts['padding'] = padding
    psfopts['unpad_x'] = unpad_x
    psfopts['unpad_y'] = unpad_y
    psfopts['lastsize'] = lastsize
    # psfo = partial(psf_convolve_cube,
    #                psfhat=da.from_array(psfhat, chunks=(1, -1, -1)),
    #                beam=None,
    #                psfopts=psfopts,
    #                wsum=1,  # psf is normalised to sum to one
    #                sigmainv=0,
    #                compute=True)

    hess2opts = copy(psfopts)
    hess2opts['sigmainv'] = 1e-8

    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    psfo = partial(_hessian_reg_psf, beam=None, psfhat=psfhat,
                    nthreads=opts.nthreads, sigmainv=0,
                    padding=padding, unpad_x=unpad_x, unpad_y=unpad_y,
                    lastsize = lastsize)



    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack


    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    if opts.threshold is None:
        threshold = opts.sigmathreshold * rms
    else:
        threshold = opts.threshold

    print("Iter %i: peak residual = %f, rms = %f" % (0, rmax, rms), file=log)
    for k in range(opts.nmiter):
        if opts.algo.lower() == 'clark':
            print("Running Clark", file=log)
            # import cProfile
            # with cProfile.Profile() as pr:
            x, status = clark(residual, psf, psfo,
                            threshold=threshold,
                            gamma=opts.gamma,
                            pf=opts.peak_factor,
                            maxit=opts.clark_maxit,
                            subpf=opts.sub_peak_factor,
                            submaxit=opts.sub_maxit,
                            verbosity=opts.verbose,
                            report_freq=opts.report_freq,
                            sigmathreshold=opts.sigmathreshold)
            # pr.print_stats(sort='cumtime')
            # quit()

        elif opts.algo.lower() == 'hogbom':
            print("Running Hogbom", file=log)
            x, status = hogbom(residual, psf,
                               threshold=threshold,
                               gamma=opts.gamma,
                               pf=opts.peak_factor,
                               maxit=opts.hogbom_maxit,
                               verbosity=opts.verbose,
                               report_freq=opts.report_freq)
        else:
            raise ValueError(f'{opts.algo} is not a valid algo option')

        model += x

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        tmp_mask = ~np.any(model, axis=0)
        rms = np.std(residual_mfs[tmp_mask])
        rmax = np.abs(residual_mfs).max()

        if opts.threshold is None:
            threshold = opts.sigmathreshold * rms
        else:
            threshold = opts.threshold

        # do flux mop if clean has stalled, not converged or
        # we have reached the final iteration/threshold
        status |= k==nmiter-1
        status |= rmax <= threshold
        if opts.mop_flux and status:
            print(f"Mopping flux at iter {k+1}", file=log)
            mask = np.any(model, axis=0)
            if opts.dirosion:
                struct = ndimage.generate_binary_structure(2, opts.dirosion)
                mask = ndimage.binary_dilation(mask, structure=struct)
                mask = ndimage.binary_erosion(mask, structure=struct)
            # hess2opts['sigmainv'] = 1e-8
            x0 = np.zeros_like(x)
            x0[:, mask] = residual_mfs[mask]
            mask = mask[None, :, :].astype(residual.dtype)
            x = pcg_psf(psfhat, mask*residual, x0,
                        mask, hess2opts, cgopts)

            model += x

            save_fits(f'{basename}_premop{k}_resid_mfs.fits', residual_mfs, hdr_mfs)

            print("Getting residual", file=log)
            convimage = hess(model)
            ne.evaluate('dirty - convimage', out=residual,
                        casting='same_kind')
            ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                        casting='same_kind')

            save_fits(f'{basename}_postmop{k}_resid_mfs.fits', residual_mfs, hdr_mfs)

            tmp_mask = ~np.any(model, axis=0)
            rms = np.std(residual_mfs[tmp_mask])
            rmax = np.abs(residual_mfs).max()

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

    print("Saving results", file=log)
    if opts.update_mask:
        mask = np.any(model > rms, axis=0)
        if opts.dirosion:
            struct = ndimage.generate_binary_structure(2, opts.dirosion)
            mask = ndimage.binary_dilation(mask, structure=struct)
            mask = ndimage.binary_erosion(mask, structure=struct)
        if 'MASK' in mds:
            mask = np.logical_or(mask, mds.MASK.values)
        mds = mds.assign(**{
                'MASK': (('x', 'y'), da.from_array(mask, chunks=(-1, -1)))
        })


    model = da.from_array(model, chunks=(1, -1, -1))
    mds = mds.assign(**{
            'CLEAN_MODEL': (('band', 'x', 'y'), model)
    })

    dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))

    if opts.fits_mfs or opts.fits_cubes:
        print("Writing fits files", file=log)
        # construct a header from xds attrs
        ra = dds[0].ra
        dec = dds[0].dec
        radec = [ra, dec]
        cell_rad = dds[0].cell_rad
        cell_deg = np.rad2deg(cell_rad)
        freq_out = mds.freq.data
        ref_freq = np.mean(freq_out)
        hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

        model_mfs = np.mean(model, axis=0)

        save_fits(f'{basename}_clean_model_mfs.fits', model_mfs, hdr_mfs)
        save_fits(f'{basename}_clean_residual_mfs.fits',
                    residual_mfs, hdr_mfs)

        if opts.fits_cubes:
            # need residual in Jy/beam
            hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
            save_fits(f'{basename}_clean_model.fits', model, hdr)
            save_fits(f'{basename}_clean_residual.fits', residual, hdr)

    print("All done here.", file=log)
