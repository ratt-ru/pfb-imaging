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

@cli.command()
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create restored images.

    Can also be used to convolve images to a common resolution
    and/or perform a primary beam correction.
    '''
    # defaults.update(kw['nworkers'])
    defaults.update(kw)
    args = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{args.output_filename}_{args.product}{args.postfix}.log')
    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(args, stack, log, scheduler=args.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _restore(**args)

def _restore(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)


    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    from pfb.utils.fits import save_fits, add_beampars, set_wcs
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres

    basename = f'{args.output_filename}_{args.product.upper()}'

    dds_name = f'{basename}{args.postfix}.dds.zarr'
    mds_name = f'{basename}{args.postfix}.mds.zarr'



    dds = xds_from_zarr(dds_name)
    # only a single mds (for now)
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq = mds.freq.data
    nx_psf = dds[0].nx_psf
    ny_psf = dds[0].ny_psf
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    output_type = dds[0].RESIDUAL.dtype
    residual = np.zeros((nband, nx, ny), dtype=output_type)
    psf = np.zeros((nband, nx_psf, ny_psf), dtype=output_type)
    wsums = np.zeros(nband)
    for ds in dds:
        b = ds.bandid
        residual[b] += ds.RESIDUAL.values
        psf[b] += ds.PSF.values
        wsums[b] += ds.WSUM.values[0]
    wsum = wsums.sum()
    residual /= wsum
    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)
    residual_mfs = np.sum(residual, axis=0)
    fmask = wsums > 0
    residual[fmask] /= wsums[fmask, None, None]/wsum
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

    for v in range(args.nband):
        cpsf[v] = Gaussian2D(xx, yy, GaussPars[v], normalise=False)

    model = mds.MODEL.values
    model_mfs = np.mean(model, axis=0)

    image_mfs = convolve2gaussres(model_mfs[None], xx, yy,
                                  GaussPar[0], args.nthreads,
                                  norm_kernel=False)[0]  # peak of kernel set to unity
    image_mfs += residual_mfs
    image = np.zeros_like(model)
    for b in range(nband):
        image[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                         GaussPars[b], args.nthreads,
                                         norm_kernel=False)  # peak of kernel set to unity
        image[b] += residual[b]

    # convert pixel units to deg
    GaussPar = list(GaussPar[0])
    GaussPar[0] *= cell_deg
    GaussPar[1] *= cell_deg
    GaussPar = tuple(GaussPar)

    # init fits headers
    radec = (mds.ra, mds.dec)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, np.mean(freq))
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq)

    if 'm' in args.outputs:
        save_fits(f'{basename}{args.postfix}.model_mfs.fits', model_mfs, hdr_mfs)

    if 'M' in args.outputs:
        save_fits(f'{basename}{args.postfix}.model.fits', model, hdr)

    # model does not get resolution info
    hdr_mfs = add_beampars(hdr_mfs, GaussPar)
    hdr = add_beampars(hdr, GaussPar, GaussPars)

    if 'r' in args.outputs:
        save_fits(f'{basename}{args.postfix}.residual_mfs.fits', residual_mfs, hdr_mfs)

    if 'R' in args.outputs:
        save_fits(f'{basename}{args.postfix}.residual.fits', model, hdr)

    if 'i' in args.outputs:
        save_fits(f'{basename}{args.postfix}.image_mfs.fits', image_mfs, hdr_mfs)

    if 'I' in args.outputs:
        save_fits(f'{basename}{args.postfix}.image.fits', image, hdr)

    if 'c' in args.outputs:
        save_fits(f'{basename}{args.postfix}.cpsf_mfs.fits', cpsf_mfs, hdr_mfs)

    if 'C' in args.outputs:
        save_fits(f'{basename}{args.postfix}.cpsf.fits', cpsf, hdr)
