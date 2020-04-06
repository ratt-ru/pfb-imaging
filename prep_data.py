
import numpy as np
from daskms import xds_from_table
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
from scipy.fftpack import next_fast_len
import argparse
from astropy.io import fits
from pfb.operators import OutMemGridder, PSF, Prior
from pfb.utils import str2bool

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--outfile", type=str,
                   help='Directory in which to place the output.')
    p.add_argument("--outname", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument("--row_chunks", type=int, default=100000,
                   help="Row chunking when loading in data")
    p.add_argument("--field", type=int, default=0, nargs='+')
                   help="Which field to image")
    p.add_argument("--ddid", type=int, )
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this factor")

    return p

def main(args):
    """
    Concatenate list of measurement sets to single table holding
    Stokes I data and weights

    ms - list of measurement sets to concatenate
    outname - name of the table to write
    cols - list of column names to concatenate

    """
    # Currently MS's need to have the same frequencies and only a single spw
    freq = None
    radec = None
    for ims in args.ms:
        if freq is None:
            freq = xds_from_table(ims + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()[0]
        else:
            tmpfreq = xds_from_table(ims + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()[0]
            np.testing.assert_array_equal(freq, tmpfreq)

        if radec is None:
            radec = xds_from_table(ims + '::FIELD')[0].PHASE_DIR.data.compute()[0].squeeze()
        else:
            tmpradec = xds_from_table(ims + '::FIELD')[0].PHASE_DIR.data.compute()[0].squeeze()
            np.testing.assert_array_equal(radec, tmpradec)
    nchan = freq.size

    # convert to Stokes I vis and concatenate
    fid = 0
    concat_cols = ('FLAG', 'FLAG_ROW', 'UVW', 'DATA', 'WEIGHT')
    nrows = 0
    dataf = []
    uvwf = []
    weightf = []
    for ims in ms:
        xds = xds_from_ms(ims, 
                          columns=concat_cols,
                          group_cols=["FIELD_ID"],
                          chunks={"row": 10000})[fid]
        
        nrow, _, ncorr = xds.DATA.shape
        nrows += nrow

        data = xds.DATA.data
        weight = xds.WEIGHT.data
        weight = da.tile(weight[:, None, :], (1, nchan, 1))
        uvw = xds.UVW.data
        flag = xds.FLAG.data

        data_I = ((weight[:, :, 0] * data[:, :, 0] + weight[:, :, ncorr-1] * data[:, :, ncorr-1])/(weight[:, :, 0] + weight[:, :, ncorr-1]))
        weight_I = (weight[:, :, 0] + weight[:, :, ncorr-1])
        flag_I = (flag[:, :, 0] | flag[:, :, ncorr-1])

        
        weight_I = da.where(~flag_I, da.sqrt(weight_I), 0.0)
        data_I = da.where(~flag_I, data_I, 0.0j)

        dataf.append(data_I)
        uvwf.append(uvw)
        weightf.append(weight_I)

    data = da.concatenate(dataf, axis=0)
    weight = da.concatenate(weightf, axis=0)
    uvw = da.concatenate(uvwf, axis=0)

    data_vars = {
        'DATA':(('row', 'chan'), data),
        'WEIGHT':(('row', 'chan'), weight),
        'UVW':(('row', 'uvw'), uvw)
    }

    writes = xds_to_table([Dataset(data_vars)], outname, "ALL")
    dask.compute(writes)

    return freq, radec


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

    if args.outfile[-1] != '/':
        args.outfile += '/'    
    
    if args.table_name is None:
        args.table_name = args.outfile + args.outname + ".table"

    main(args)