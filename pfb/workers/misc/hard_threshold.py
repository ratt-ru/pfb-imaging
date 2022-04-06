from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('HTHRESH')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.hard_threshold["inputs"].keys():
    defaults[key] = schema.hard_threshold["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.hard_threshold)
def hard_threshold(**kw):
    '''
    Apply hard threshold to model and write out corresponding fits file
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _hard_threshold(**opts)

def _hard_threshold(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny

    model = mds.get(opts.model_name).values

    model = np.where(model < opts.threshold, 0.0, model)

    radec = [mds.ra, mds.dec]
    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = mds.freq.data
    ref_freq = np.mean(freq_out)
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    model_mfs = np.mean(model, axis=0)

    save_fits(f'{basename}_threshold_model.fits', model, hdr)
    save_fits(f'{basename}_threshold_model_mfs.fits', model_mfs, hdr_mfs)
