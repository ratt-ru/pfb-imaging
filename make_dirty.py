
import numpy as np
from daskms import xds_from_table
import dask
import dask.array as da
from scipy.fftpack import next_fast_len
import argparse
from astropy.io import fits
from pfb.operators import OutMemGridder, PSF, Prior

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--model_column", default="MODEL_DATA", type=str,
                   help="Column to write model visibilities to.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--nx", type=int, default=None,
                   help="Number of x pixels.")
    p.add_argument("--ny", type=int, default=None,
                   help="Number of y pixels.")
    p.add_argument('--cell_size', type=float, default=None,
                   help="Cell size in arcseconds.")
    p.add_argument("--fov", type=float, default=None,
                   help="Field of view in degrees")
    p.add_argument("--outfile", type=str,
                   help='Directory in which to place the output.')
    p.add_argument("--outname", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument('--precision', default=1e-7, type=float,
                   help='Precision of the gridder.')
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=False,
                   help="Whether to do wide-field correction or not")
    p.add_argument("--channels_out", default=None, type=int,
                   help="Number of channels in output cube")
    p.add_argument("--row_chunks", type=int, default=100000,
                   help="Row chunking when loading in data")
    p.add_argument("--field", type=int, default=0,
                   help="Which field to image")
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this factor")

    return p

def main(args):

    return


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print("Using %i threads"%args.ncpu)

    main(args)