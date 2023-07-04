# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('RESTORE')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.restore["inputs"].keys():
    defaults[key] = schema.restore["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create fits image data products (eg. restored images).
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')
    if opts.nworkers is None:
        opts.nworkers = opts.nband

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _restore(**opts)

def _restore(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    from pfb.utils.fits import (save_fits, add_beampars, set_wcs,
                                dds2fits, dds2fits_mfs)
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres, dds2cubes

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'

    dds = xds_from_zarr(dds_name)
    nband = opts.nband
    nx = dds[0].x.size
    ny = dds[0].y.size
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq = []
    for ds in dds:
        freq.append(ds.freq_out)
        assert ds.x.size == nx
        assert ds.y.size == ny
    freq = np.unique(np.array(freq))
    assert freq.size == opts.nband
    nx_psf = dds[0].x_psf.size
    ny_psf = dds[0].y_psf.size

    # init fits headers
    radec = (dds[0].ra, dds[0].dec)
    ref_freq = np.mean(freq)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq)


    dirty, model, residual, psf, _, _, wsums, _ = dds2cubes(dds,
                                                            nband,
                                                            apparent=True)
    wsum = np.sum(wsums)
    output_type = dirty.dtype
    fmask = wsums > 0

    if residual is None:
        print('Warning, no residual in dds. '
              'Using dirty as residual.', file=log)
        residual = dirty.copy()
    residual_mfs = np.sum(residual, axis=0)
    residual[fmask] /= wsums[fmask, None, None]/wsum

    if not model.any():
        print("Warning - model is empty", file=log)
    model_mfs = np.mean(model, axis=0)

    if psf is not None:
        psf_mfs = np.sum(psf, axis=0)
        psf[fmask] /= wsums[fmask, None, None]/wsum
        # sanity check
        assert (psf_mfs.max() - 1.0) < 2e-7
        assert ((np.amax(psf, axis=(1,2)) - 1.0) < 2e-7).all()

        # fit restoring psf
        GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
        GaussPars = fitcleanbeam(psf, level=0.5, pixsize=1.0)  # pixel units

        cpsf_mfs = np.zeros(residual_mfs.shape, dtype=output_type)
        cpsf = np.zeros(residual.shape, dtype=output_type)

        lpsf = -(nx//2) + np.arange(nx)
        mpsf = -(ny//2) + np.arange(ny)
        xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

        cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)

        for v in range(opts.nband):
            cpsf[v] = Gaussian2D(xx, yy, GaussPars[v], normalise=False)


        image_mfs = convolve2gaussres(model_mfs[None], xx, yy,
                                    GaussPar[0], opts.nvthreads,
                                    norm_kernel=False)[0]  # peak of kernel set to unity
        image_mfs += residual_mfs
        image = np.zeros_like(model)
        for b in range(nband):
            image[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                            GaussPars[b], opts.nvthreads,
                                            norm_kernel=False)  # peak of kernel set to unity
            image[b] += residual[b]

            # convert pixel units to deg
            GaussPar = list(GaussPar[0])
            GaussPar[0] *= cell_deg
            GaussPar[1] *= cell_deg
            GaussPar = tuple(GaussPar)

            GaussPars = list(GaussPars)
            for i, gp in enumerate(GaussPars):
                GaussPars[i] = (gp[0]*cell_deg, gp[1]*cell_deg, gp[2])
            GaussPars = tuple(GaussPars)

        hdr_mfs = add_beampars(hdr_mfs, GaussPar)
        hdr = add_beampars(hdr, GaussPar, GaussPars)
    else:
        print('Warning, no psf in dds. '
              'Unable to add resolution info or make restored image. ',
              file=log)

    if 'm' in opts.outputs:
        save_fits(model_mfs,
                  f'{basename}_{opts.postfix}.model_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'M' in opts.outputs:
        save_fits(model,
                  f'{basename}_{opts.postfix}.model.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'r' in opts.outputs:
        save_fits(residual_mfs,
                  f'{basename}_{opts.postfix}.residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'R' in opts.outputs:
        save_fits(residual,
                  f'{basename}_{opts.postfix}.residual.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'd' in opts.outputs:
        dirty_mfs = np.sum(dirty, axis=0)
        save_fits(dirty_mfs,
                  f'{basename}_{opts.postfix}.dirty_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'D' in opts.outputs:
        dirty[fmask] /= wsums[fmask, None, None]/wsum
        save_fits(dirty,
                  f'{basename}_{opts.postfix}.dirty.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'i' in opts.outputs and psf is not None:
        save_fits(image_mfs,
                  f'{basename}_{opts.postfix}.image_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'I' in opts.outputs and psf is not None:
        save_fits(image,
                  f'{basename}_{opts.postfix}.image.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'c' in opts.outputs and psf is not None:
        save_fits(cpsf_mfs,
                  f'{basename}_{opts.postfix}.cpsf_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'C' in opts.outputs and psf is not None:
        save_fits(cpsf,
                  f'{basename}_{opts.postfix}.cpsf.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here", file=log)
