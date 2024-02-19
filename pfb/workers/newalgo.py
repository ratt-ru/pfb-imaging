# flake8: noqa
import os
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('NEWALGO')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.newalgo["inputs"].keys():
    defaults[key] = schema.newalgo["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.newalgo)
def newalgo(**kw):
    '''
    newalgo algorithm
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    OmegaConf.set_struct(opts, True)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'newalgo_{timestamp}.log')

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _newalgo(**opts)


def _newalgo(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import xarray as xr
    import numexpr as ne
    import dask
    import dask.array as da
    from pfb.utils.fits import (set_wcs, save_fits, dds2fits,
                                dds2fits_mfs, load_fits)
    from pfb.utils.misc import dds2cubes
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.operators.hessian import hessian_xds
    from pfb.operators.psf import psf_convolve_cube
    from copy import copy, deepcopy
    from ducc0.misc import make_noncritical
    from pfb.utils.misc import fitcleanbeam
    from taivas.drivers.predict_windowed import main as taivas

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}_{opts.postfix}.dds.zarr'
    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan':-1,
                                          'x':-1,
                                          'y':-1,
                                          'x_psf':-1,
                                          'y_psf':-1,
                                          'yo2':-1})
    if opts.memory_greedy:
        dds = dask.persist(dds)[0]

    nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
    lastsize = ny_psf

    # stitch dirty/psf in apparent scale
    output_type = dds[0].DIRTY.dtype
    dirty, model, residual, psf, psfhat, beam, wsums, dual = dds2cubes(
                                                               dds,
                                                               opts.nband,
                                                               apparent=False)

    wsum = np.sum(wsums)
    psf_mfs = np.sum(psf, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*opts.epsilon
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    else:
        residual_mfs = np.sum(residual, axis=0)

    # note mfs images are in Jy/beam
    # to covert the cubes to Jy/beam do something like this
    # fsel = wsums > 0
    # dirty_cube[fsel] *= wsum/wsums[fsel, None, None]

    # for intermediary results (not currently written)
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

    # # TODO - check coordinates match
    # # Add option to interp onto coordinates?
    # if opts.mask is not None:
    #     mask = load_fits(mask, dtype=output_type).squeeze()
    #     assert mask.shape == (nx, ny)
    #     mask = mask.astype(output_type)
    #     print('Using provided fits mask', file=log)
    # else:
    #     mask = np.ones((nx, ny), dtype=output_type)

    # set up vis space Hessian
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['wstack'] = opts.wstack
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads  # nvthreads since dask parallel over band
    # always clean in apparent scale so no beam
    # mask is applied to residual after hessian application
    # hess(x) = R.H W R x  ->  used in major cycle to compute exact gradient
    # R = degridding operator
    # R.H = gridding operator
    # W = weights
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0,
                   mask=np.ones((nx, ny), dtype=output_type),
                   compute=True, use_beam=False)


    # image space hessian
    # pre-allocate arrays for doing FFT's
    xout = np.empty(dirty.shape, dtype=dirty.dtype, order='C')
    xout = make_noncritical(xout)
    xpad = np.empty(psf.shape, dtype=dirty.dtype, order='C')
    xpad = make_noncritical(xpad)
    xhat = np.empty(psfhat.shape, dtype=psfhat.dtype)
    xhat = make_noncritical(xhat)
    # psf_convolve(x) = Z.H F.H hat(PSF) F Z x  - used in minor cycle to compute approximate gradients
    # Z = zero padding
    # F = FFT (real to complex or r2c)
    # hat(PSF) = FT of the PSF
    # F.H = iFFT (complex to real or c2r)
    # Z.H = unpadding
    psf_convolve = partial(psf_convolve_cube, xpad, xhat, xout, psfhat, lastsize,
                           nthreads=opts.nvthreads*opts.nthreads_dask)  # nthreads = nvthreads*nthreads_dask because dask not involved

    # get clean beam area to convert residual units during l1reweighting
    # TODO - could refine this with comparison between dirty and restored
    # if contiuing the deconvolution
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.25, pixsize=1.0)[0]
    pix_per_beam = GaussPar[0]*GaussPar[1]*np.pi/4
    print(f"Number of pixels per beam estimated as {pix_per_beam}",
          file=log)

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    print(f"Iter 0: peak residual = {rmax:.3e}, rms = {rms:.3e}",
          file=log)
    for k in range(opts.niter):
        print('Solving for model', file=log)
        modelp = deepcopy(model)
        # Some things to consider
        # the true gradient is
        # grad_x f(x) = R.H W R x - ID = hess(x) - ID = -IR
        # the approx gradient is
        # approxgrad_x f(x) = psf_convolve(x) - ID = -IR

        # Suppose x = x_k + dx where x_k is the model image at the previous step and dx the update
        # Since both hess and psf_convolve are linear we can also write the gradient as
        # grad_x f(x) = hess(x_k + dx) - ID = hess(x_k) + hess(dx) - ID = hess(dx) - IR_k
        # so we can just keep on deconvoling the residual
        # dx = some_deconv_algo(residual, psf, **deconv_algo_opts)
        # and incrementing the model like so
        # model += dx
        # residual = dirty - hess(model)

        # Can't do this for the approxgrad so we do the below (Landman to think about how to write this down formally)
        # so we have the following trick
        # data = residual + psf_convolve(model)
        # grad = lambda x: psf_convolve(x) - data
        # model = some_deconv_algo(data, psf, **deconv_algo_opts)
        # residual = dirty - hess(model)

        # point to insert deconvolution algo
        # model = taivas(residual, psf, ...)

        import ipdb; ipdb.set_trace()

        save_fits(basename + f'_model_{k+1}.fits', np.mean(model, axis=0), hdr_mfs)

        print("Getting residual", file=log)
        convimage = hess(model)
        ne.evaluate('dirty - convimage', out=residual,
                    casting='same_kind')
        ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                    casting='same_kind')

        save_fits(basename + f'_residual_{k+1}.fits', residual_mfs, hdr_mfs)

        rms = np.std(residual_mfs)
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print(f"Iter {k+1}: peak residual = {rmax:.3e}, "
              f"rms = {rms:.3e}, eps = {eps:.3e}",
              file=log)

        print("Updating results", file=log)
        dds_out = []
        for ds in dds:
            b = ds.bandid
            r = da.from_array(residual[b]*wsum)
            m = da.from_array(model[b])
            d = da.from_array(dual[b])
            ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                                  'MODEL': (('x', 'y'), m)})
            dds_out.append(ds_out)
        writes = xds_to_zarr(dds_out, dds_name,
                             columns=('RESIDUAL', 'MODEL'),
                             rechunk=True)
        dask.compute(writes)

        if eps < opts.tol:
            print(f"Converged after {k+1} iterations.", file=log)
            break
        # if rmax <= threshold:
        #     print("Terminating because final threshold has been reached",
        #           file=log)
        #     break

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
        fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    print("All done here.", file=log)

