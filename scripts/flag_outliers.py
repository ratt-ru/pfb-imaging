#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
import argparse
from pfb.utils import str2bool

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+',
                   help="List of measurement sets to image")
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use.")
    p.add_argument("--model_column", default='MODEL_DATA', type=str,
                   help="Column to write model data to")
    p.add_argument("--flag_column", default='FLAG', type=str)
    p.add_argument("--flag_out_column", default='UPDATED_FLAGS', type=str,
                   help="Column to write updated flags to")
    p.add_argument("--sigma_cut", type=float, default=5.0, 
                   help="Whitened residual visibilities with amplitudes larger than"
                   "sigma_cut * global mean will be flagged")
    p.add_argument("--nthreads", type=int, default=0)
    return p


def main(args):
    """
    Flags outliers in data given a model
    """
    radec_ref = None
    # first pass to determine global mean of whitened residual visibility amplitudes
    sum_amps = []
    counts = []
    for ims in args.ms:
        xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                            chunks={"row":-1, "chan":1},
                            columns=('UVW', args.data_column, args.weight_column,
                                     args.model_column, args.flag_column))

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
        pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]
        
        for ds in xds:
            field = fields[ds.FIELD_ID]
            radec = field.PHASE_DIR.data.squeeze()

            # check fields match
            if radec_ref is None:
                radec_ref = radec

            if not np.array_equal(radec, radec_ref):
                continue

            # load in data and compute whitened residuals
            data = getattr(ds, args.data_column).data
            model = getattr(ds, args.model_column).data
            flag = getattr(ds, args.flag_column).data
            weights = getattr(ds, args.weight_column).data
            if len(weights.shape) < 3:
                weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
            
            resid_vis = ~flag*(data - model)*da.sqrt(weights)
            abs_resid_vis_I = (0.5*(resid_vis[:, :, 0] + resid_vis[:, :, -1])).__abs__()

            # get abs value and accumulate 
            sum_amps.append(da.sum(abs_resid_vis_I))
            counts.append(da.sum(~flag))

    # compute mean
    sum_amps = dask.compute(sum_amps)
    counts = dask.compute(counts)
    mean_amp = np.sum(sum_amps)/np.sum(counts)

    # second pass updates flags
    writes  = []
    for ims in args.ms:
        xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                            chunks={"row":-1, "chan":1},
                            columns=('UVW', args.data_column, args.weight_column,
                                     args.model_column, args.flag_column))

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
        pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]
        
        out_data = []
        for ds in xds:
            field = fields[ds.FIELD_ID]
            radec = field.PHASE_DIR.data.squeeze()

            # check fields match
            if radec_ref is None:
                radec_ref = radec

            if not np.array_equal(radec, radec_ref):
                continue

            # load in data and compute whitened residuals
            data = getattr(ds, args.data_column).data
            model = getattr(ds, args.model_column).data
            flag = getattr(ds, args.flag_column).data
            weights = getattr(ds, args.weight_column).data
            if len(weights.shape) < 3:
                weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
            
            resid_vis = ~flag*(data - model)*da.sqrt(weights)
            abs_resid_vis_I = (0.5*(resid_vis[:, :, 0] + resid_vis[:, :, -1])).__abs__()

            # update flags
            tmp = abs_resid_vis_I >= args.sigma_cut * mean_amp
            updated_flag = da.broadcast_to(tmp[:, :, None], flag.shape, chunks=flag.chunks)

            out_ds = ds.assign(**{args.flag_out_column: (("row", "chan", "corr"), updated_flag)})

            out_data.append(out_ds)
        writes.append(xds_to_table(out_data, ims, columns=[args.flag_out_column]))
    dask.compute(writes)








            




if __name__=="__main__":
    args = create_parser().parse_args()

    if args.nthreads:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.nthreads))
    else:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    main(args)