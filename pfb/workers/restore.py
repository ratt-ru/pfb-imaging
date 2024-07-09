# flake8: noqa
from contextlib import ExitStack
from pathlib import Path
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
    defaults[key.replace("-", "_")] = schema.restore["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create fits image cubes from data products (eg. restored images).
    '''
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    logname = f'{str(ldir)}/restore_{timestamp}.log'
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
    from ducc0.fft import c2c
    from ducc0.misc import resize_thread_pool, thread_pool_size
    nthreads_tot = opts.nthreads_dask * opts.nthreads
    resize_thread_pool(nthreads_tot)
    print(f'ducc0 max number of threads set to {thread_pool_size()}', file=log)

    basename = opts.output_filename
    if opts.fits_output_folder is not None:
        fits_oname = opts.fits_output_folder + '/' + basename.split('/')[-1]
    else:
        fits_oname = basename

    dds_name = f'{basename}_{opts.suffix}.dds'
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

    # init fits headers
    radec = (dds[0].ra, dds[0].dec)
    ref_freq = np.mean(freq)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq)

    if opts.overwrite:
        print("Warning! Potentially overwriting output images",
              file=log)

    # stack cubes
    dirty, model, residual, psf, _, _, wsums, _ = dds2cubes(dds,
                                                            nband,
                                                            modelname=opts.model_name,
                                                            residname=opts.residual_name,
                                                            apparent=True,
                                                            dual=False)
    wsum = np.sum(wsums)
    output_type = dirty.dtype
    fmask = wsums > 0
    if (~fmask).all():
        raise ValueError("All data seem to be flagged")

    if residual is None:
        print('Warning, no residual in dds. '
              'Using dirty as residual.', file=log)
        residual = dirty.copy()
    residual_mfs = np.sum(residual, axis=0)
    residual[fmask] /= wsums[fmask, None, None]/wsum

    if not model.any():
        print("Warning - model is empty", file=log)
    model_mfs = np.mean(model[fmask], axis=0)

    # lm in pixel coordinates
    lpsf = -(nx//2) + np.arange(nx)
    mpsf = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

    if psf is not None:
        nx_psf = dds[0].x_psf.size
        ny_psf = dds[0].y_psf.size
        psf_mfs = np.sum(psf, axis=0)
        psf[fmask] /= wsums[fmask, None, None]/wsum
        # sanity check
        try:
            psf_mismatch_mfs = np.abs(psf_mfs.max() - 1.0)
            psf_mismatch = np.abs(np.amax(psf, axis=(1,2))[fmask] - 1.0).max()
            assert psf_mismatch_mfs < 1e-5
            assert psf_mismatch < 1e-5
        except Exception as e:
            max_mismatch = np.maximum(psf_mismatch_mfs, psf_mismatch)
            print(f"Warning - PSF does not normlaise to one. "
                  f"Max mismatch is {max_mismatch:.3e}", file=log)

        # fit restoring psf (pixel units)
        GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
        hdr_mfs = add_beampars(hdr_mfs, GaussPar, unit2deg=cell_deg)
        GaussPars = fitcleanbeam(psf, level=0.5, pixsize=1.0)
        hdr = add_beampars(hdr, GaussPar, GaussPars, unit2deg=cell_deg)

    else:
        print('Warning, no psf in dds. '
              'Unable to add resolution info or make restored image. ',
              file=log)
        GaussPar = None
        GaussPars = None

    if 'm' in opts.outputs:
        save_fits(model_mfs,
                  f'{basename}_{opts.suffix}.model_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'M' in opts.outputs:
        save_fits(model,
                  f'{basename}_{opts.suffix}.model.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'r' in opts.outputs:
        save_fits(residual_mfs,
                  f'{basename}_{opts.suffix}.residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'R' in opts.outputs:
        save_fits(residual,
                  f'{basename}_{opts.suffix}.residual.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'f' in opts.outputs:
        rhat_mfs = c2c(residual_mfs, forward=True,
                       nthreads=opts.nthreads, inorm=0)
        rhat_mfs = np.fft.fftshift(rhat_mfs)
        save_fits(np.abs(rhat_mfs),
                  f'{basename}_{opts.suffix}.abs_fft_residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)
        save_fits(np.angle(rhat_mfs),
                  f'{basename}_{opts.suffix}.phase_fft_residual_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'F' in opts.outputs:
        rhat = c2c(residual, axes=(1,2), forward=True,
                   nthreads=opts.nthreads, inorm=0)
        rhat = np.fft.fftshift(rhat, axes=(1,2))
        save_fits(np.abs(rhat),
                  f'{basename}_{opts.suffix}.abs_fft_residual.fits',
                  hdr,
                  overwrite=opts.overwrite)
        save_fits(np.angle(rhat),
                  f'{basename}_{opts.suffix}.phase_fft_residual.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'd' in opts.outputs:
        dirty_mfs = np.sum(dirty, axis=0)
        save_fits(dirty_mfs,
                  f'{basename}_{opts.suffix}.dirty_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'D' in opts.outputs:
        dirty[fmask] /= wsums[fmask, None, None]/wsum
        save_fits(dirty,
                  f'{basename}_{opts.suffix}.dirty.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'i' in opts.outputs and psf is not None:
        image_mfs = convolve2gaussres(model_mfs[None], xx, yy,
                                      GaussPar[0], opts.nthreads,
                                      norm_kernel=False)[0]  # peak of kernel set to unity
        image_mfs += residual_mfs
        save_fits(image_mfs,
                  f'{basename}_{opts.suffix}.image_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'I' in opts.outputs and psf is not None:
        image = np.zeros_like(model)
        for b in range(nband):
            image[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                            GaussPars[b], opts.nthreads,
                                            norm_kernel=False)  # peak of kernel set to unity
            image[b] += residual[b]
        save_fits(image,
                  f'{basename}_{opts.suffix}.image.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if 'c' in opts.outputs:
        if GaussPar is None:
            raise ValueError("Clean beam in output but no PSF in dds")
        cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)
        save_fits(cpsf_mfs,
                  f'{basename}_{opts.suffix}.cpsf_mfs.fits',
                  hdr_mfs,
                  overwrite=opts.overwrite)

    if 'C' in opts.outputs:
        if GaussPars is None:
            raise ValueError("Clean beam in output but no PSF in dds")
        cpsf = np.zeros(residual.shape, dtype=output_type)
        for v in range(opts.nband):
            gpar = GaussPars[v]
            if not np.isnan(gpar).any():
                cpsf[v] = Gaussian2D(xx, yy, gpar, normalise=False)
        save_fits(cpsf,
                  f'{basename}_{opts.suffix}.cpsf.fits',
                  hdr,
                  overwrite=opts.overwrite)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here", file=log)
