
import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
import argparse
from pfb.utils import str2bool

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--table_name", type=str,
                   help='Directory in which to place the output.')
    p.add_argument("--field", default=0, nargs='+',
                   help="Which fields to image. Set to 'all' to combine all fields or comma separated list to combine a subset.")
    p.add_argument("--radec", type=float, default=None, nargs='+',
                   help="Measurements will be rephased to have this field center. "
                   "If none all will be rephased to field 0 of first ms")
    p.add_argument("--ddid", type=int, default=0, nargs='+',
                   help="Spectral windows to combine")
    p.add_argument("--pol_products", type=str, default='I',
                   help="Polarisation products to compute. Only Stokes I currently supported")
    p.add_argument("--row_chunks", type=int, default=1000000,
                   help="Row chunking when loading in data")
    p.add_argument("--ncpu", type=int, default=0)
    p.add_argument("--overwrite_table", type=str2bool, nargs='?', const=True, default=False,
                   help="Allow overwriting of existing table")
    return p

# LB - TODO - rephase and interpolate beams
def main(args):
    # use key value pair to keep track of fields which much not have the same id number across measurement sets
    radec_ids = {}
    ifield = 0
    field_id = None
    # only single spectral window supported
    freq = None
    # only single ddid
    ddid_id = 0
    out_datasets = []
    for ms in args.ms:
        datasets = xds_from_ms(ms, 
                               columns=(args.data_column, args.weight_column, 'FLAG', 'FLAG_ROW', 'UVW', 'TIME'),
                               chunks={"row": args.row_chunks})

        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")
        fields = xds_from_table(ms + "::FIELD", group_cols="__row__")
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW", group_cols="__row__")
        pols = xds_from_table(ms + "::POLARIZATION", group_cols="__row__")

        # Get subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        for ds in datasets:
            if ds.FIELD_ID not in args.field:
                continue

            field = fields[ds.FIELD_ID]
            radec = field.PHASE_DIR.data.squeeze()

            if args.radec is None:
                args.radec = radec

            for key in radec_ids.keys():
                if np.array_equal(radec, radec_ids[key]):
                    field_id = key
                else:
                    field_id = None

            if field_id is None:
                radec_ids[ifield] = radec
                field_id = ifield
                ifield += 1

            print("Adding field %i from %s. New field ID is %i"%(ds.FIELD_ID, ms, field_id))

            ddid = ddids[ds.DATA_DESC_ID]

            pol = pols[ddid.POLARIZATION_ID.values[0]]
            corr_type_set = set(pol.CORR_TYPE.data.squeeze())
            if corr_type_set.issubset(set([9, 10, 11, 12])):
                pol_type = 'linear'
            elif corr_type_set.issubset(set([5, 6, 7, 8])):
                pol_type = 'circular'
            else:
                raise ValueError("Cannot determine polarisation type "
                                "from correlations %s. Constructing "
                                "a feed rotation matrix will not be "
                                "possible." % (corr_type_set,))
                        
            spw = spws[ddid.SPECTRAL_WINDOW_ID.values[0]]
            if freq is None:
                freq = spw.CHAN_FREQ
            else:
                try:
                    np.testing.assert_array_equal(freq, spw.CHAN_FREQ)
                    ddid_id = 0
                except:
                    raise ValueError("Frequencies don't match") 

            # get data and convert to output products
            data = getattr(ds, args.data_column).data
            weight = getattr(ds, args.weight_column).data
            if len(weight.shape)<3:  # tile over frequency if no WEIGHT_SPECTRUM
                weight = da.tile(weight[:, None, :], (1, freq.size, 1))
            uvw = ds.UVW.data
            flag = da.logical_or(ds.FLAG.data, ds.FLAG_ROW.data[:, None, None])
            flag = flag[:, :, 0] | flag[:, :, -1]

            

            if 'I' in args.pol_products:
                ncorr = data.shape[-1]
                data = (weight[:, :, 0] * data[:, :, 0] + weight[:, :, ncorr-1] * data[:, :, ncorr-1])
                weight = (weight[:, :, 0] + weight[:, :, ncorr-1])

            # normalise by sum of weights
            weight = da.where(flag, 0.0, weight)
            data = da.where(weight != 0.0, data/weight, 0.0j)

            data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data, field_id)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data, ddid_id)),
                'DATA':(('row', 'chan'), data),
                'WEIGHT':(('row', 'chan'), weight),
                'IMAGING_WEIGHT':(('row', 'chan'), da.sqrt(weight)),
                'UVW':(('row', 'uvw'), uvw)
            }

            out_ds = Dataset(data_vars)

            out_datasets.append(out_ds)
    
    out_freq = Dataset(
            {
                'FREQ':(('row','chan'), da.from_array(freq))
            }
    )
    radec = args.radec[None, :]
    out_radec = Dataset(
            {
                'RADEC':(('row', 'dir'), da.from_array(radec))
            }
    )

    writes = xds_to_table(out_datasets, args.table_name, columns="ALL")
    writes_freq = xds_to_table(out_freq, args.table_name+'::FREQ', columns='ALL')
    writes_radec = xds_to_table(out_radec, args.table_name+'::RADEC', columns='ALL')
    dask.compute(writes, writes_freq, writes_radec)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    if not isinstance(args.field, list):
        args.field = [args.field]

    if 'all' in args.field:
        class tmp(object):
            def __contains__(self, other):
                return True
        args.field = tmp()

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    import os
    if os.path.exists(args.table_name):
        if args.overwrite_table:
            print("Removing %s"%args.table_name)
            import shutil
            shutil.rmtree(args.table_name)
        else:
            raise ValueError("Table %s exists and overwrite_table is prohibited"%args.table_name) 

    main(args)