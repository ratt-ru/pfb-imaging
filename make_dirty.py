
import numpy as np
import dask
from daskms import xds_from_table
from astropy.io import fits
from pfb.utils import str2bool, set_wcs, save_fits
from pfb.operators import OutMemGridder
import argparse

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("table_name", type=str,
                   help="Name of table to load concatenated Stokes I vis from")
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--outfile", type=str, default='pfb',
                   help='Base name of output file.')
    p.add_argument("--fov", type=float, default=None,
                   help="Field of view in degrees")
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this factor unless specified by cell_size.") 
    p.add_argument("--nx", type=int, default=None,
                   help="Number of x pixels. Computed automatically from fov if None.")
    p.add_argument("--ny", type=int, default=None,
                   help="Number of y pixels. Computed automatically from fov if None.")
    p.add_argument('--cell_size', type=float, default=None,
                   help="Cell size in arcseconds. Computed automatically from super_resolution_factor if None")
    p.add_argument('--precision', default=1e-7, type=float,
                   help='Precision of the gridder.')
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=True,
                   help="Whether to do wide-field correction or not")
    p.add_argument("--channels_out", default=None, type=int,
                   help="Number of channels in output cube")
    p.add_argument("--row_chunks", type=int, default=100000,
                   help="Row chunking when loading in data")
    p.add_argument("--field", type=int, default=0, nargs='+',
                   help="Which fields to image")
    p.add_argument("--ncpu", type=int, default=0)
    return p

def main(args):
    if args.precision > 1e-6:
        real_type = np.float32
        complex_type=np.complex64
    else:
        real_type = np.float64
        complex_type=np.complex128

    # get max uv coords over all fields
    uvw = []
    xds = xds_from_table(args.table_name, group_cols=('FIELD_ID'), columns=('UVW'), chunks={'row':-1})
    for ds in xds:
        uvw.append(ds.UVW.data.compute())
    uvw = np.concatenate(uvw)
    from africanus.constants import c as lightspeed
    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    # del uvw

    # get Nyquist cell size
    freq = xds_from_table(args.table_name+"::FREQ")[0].FREQ.data.compute().squeeze()
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)

    if args.cell_size is not None:
        cell_rad = args.cell_size * np.pi/60/60/180
        print("Super resolution factor = ", cell_N/cell_rad)
    else:
        cell_rad = cell_N/args.super_resolution_factor
        args.cell_size = cell_rad*60*60*180/np.pi
        print("Cell size set to %5.5e arcseconds" % args.cell_size)
    
    if args.nx is None:
        fov = args.fov*3600
        nx = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.nx = next_fast_len(nx)

    if args.ny is None:
        fov = args.fov*3600
        ny = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.ny = next_fast_len(ny)

    if args.channels_out is None:
        args.channels_out = freq.size

    print("Image size set to (%i, %i, %i)"%(args.channels_out, args.nx, args.ny))


    # init gridder
    R = OutMemGridder(args.table_name, args.nx, args.ny, args.cell_size, freq, 
                      nband=args.channels_out, field=args.field, precision=args.precision, 
                      ncpu=args.ncpu, do_wstacking=args.do_wstacking)
    freq_out = R.freq_out

    # get headers
    radec = xds_from_table(args.table_name+"::RADEC")[0].RADEC.data.compute().squeeze()
    hdr = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, freq_out)
    hdr_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, np.mean(freq_out))
    hdr_psf = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, freq_out)
    hdr_psf_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, np.mean(freq_out))
    
    # make psf  LB - TODO: undersized psfs
    psf= R.make_psf()
    nband = R.nband
    psf_max = np.amax(psf.reshape(nband, 4*args.nx*args.ny), axis=1)

    # make dirty
    dirty = R.make_dirty()

    # save dirty and psf images
    save_fits(args.outfile + '_dirty.fits', dirty, hdr, dtype=real_type)
    save_fits(args.outfile + '_psf.fits', psf, hdr_psf, dtype=real_type)
    
    # MFS images
    wsum = np.sum(psf_max)
    dirty_mfs = np.sum(dirty, axis=0)/wsum 
    save_fits(args.outfile + '_dirty_mfs.fits', dirty_mfs, hdr_mfs)

    psf_mfs = np.sum(psf, axis=0)/wsum
    save_fits(args.outfile + '_psf_mfs.fits', psf_mfs, hdr_psf_mfs)

    rmax = np.abs(dirty_mfs).max()
    rms = np.std(dirty_mfs)
    print("Peak of dirty is %f and rms is %f"%(rmax, rms))       


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    main(args)

    