from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('ZARR2FITS')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.zarr2fits["inputs"].keys():
    defaults[key] = schema.zarr2fits["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.zarr2fits)
def zarr2fits(**kw):
    '''
    Render zarr data set to fits
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'zarr2fits.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    return _zarr2fits(**opts)

def _zarr2fits(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from daskms.experimental.zarr import xds_from_zarr
    import numpy as np
    from astropy.wcs import WCS
    from pfb.utils.fits import save_fits

    xds = xds_from_zarr(opts.zfile)

    if opts.column not in xds[0]:
        raise ValueError(f'Column {opts.column} not in {opts.zfile}')

    scans = []
    bands = []
    freq_out = []
    for ds in xds:
        if ds.scanid not in scans:
            scans.append(ds.scanid)
        if ds.bandid not in bands:
            bands.append(ds.bandid)
            freq_out.append(np.mean(ds.FREQ.values))

    nscan = len(scans)
    nband = len(bands)
    nx, ny = xds[0].get(opts.column).data.shape

    img = np.zeros((nscan, nband, nx, ny), dtype=np.float32)

    scan2idx = {}
    for i, scan in enumerate(scans):
        scan2idx[scan] = i

    for ds in xds:
        img[scan2idx[ds.scanid], ds.bandid] += ds.get(opts.column).values.astype(np.float32)

    ra = xds[0].ra
    dec = xds[0].dec
    radec = [ra, dec]
    cell_rad = xds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq_out = np.array(freq_out)
    assert freq_out.size == nband
    ref_freq = freq_out[0]

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'SCAN']
    w.wcs.cdelt[0] = -cell_rad
    w.wcs.cdelt[1] = cell_rad
    w.wcs.cdelt[2] = freq_out[1] - freq_out[0]
    w.wcs.cdelt[3] = 2
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    w.wcs.crval = [radec[0]*180.0/np.pi, radec[1]*180.0/np.pi, ref_freq, scans[0]]
    # LB - y axis treated differently because of stupid fits convention
    w.wcs.crpix = [1 + nx//2, ny//2, 1, 1]

    header = w.to_header()
    header['RESTFRQ'] = ref_freq
    header['ORIGIN'] = 'pfb-clean'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = 'Jy/beam'
    header['SPECSYS'] = 'TOPOCENT'

    zname = opts.zfile.rstrip('.zarr')
    save_fits(f'{zname}_{opts.column}.fits', img, header)

    print("All done here", file=log)
