#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import numpy as np
import dask
from dask.diagnostics import ProgressBar
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
    p.add_argument("--weight_out_column", default='UPDATED_WEIGHT', type=str,
                   help="Weight column to use.")
    p.add_argument("--model_column", default='MODEL_DATA', type=str,
                   help="Column to write model data to")
    p.add_argument("--flag_column", default='FLAG', type=str)
    p.add_argument("--flag_out_column", default='UPDATED_FLAG', type=str,
                   help="Column to write updated flags to")
    p.add_argument("--sigma_cut", type=float, default=5.0, 
                   help="Whitened residual visibilities with amplitudes larger than"
                   "sigma_cut * global mean will be flagged")
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--row_chunks", type=int, default=-1)
    p.add_argument("--chan_chunks", type=int, default=-1)
    p.add_argument("--report_means", type=str2bool, nargs='?', const=True, default=False,
                   help="Will report new mean of whitened residual visibility amplitudes"
                   "(requires two passes through the data)")
    return p


def main(args):
    """
    Flags outliers in data given a model and rescale weights so that whitened residuals have a
    mean amplitude of sqrt(2). 
    
    Flags and weights are computed per chunk of data
    """
    radec_ref = None
    writes = []
    for ims in args.ms:
        xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                          chunks={"row":args.row_chunks, "chan":args.chan_chunks},
                          columns=('UVW', args.data_column, args.weight_column,
                                   args.model_column, args.flag_column, 'FLAG_ROW'))

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
            flag = da.logical_or(flag, ds.FLAG_ROW.data[:, None, None])
            weights = getattr(ds, args.weight_column).data
            if len(weights.shape) < 3:
                weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
            
            # Stokes I vis 
            weights = (~flag)*weights
            resid_vis = (data - model)*weights
            wsums = (weights[:, :, 0] + weights[:, :, -1])
            resid_vis_I = da.where(wsums, (resid_vis[:, :, 0] + resid_vis[:, :, -1])/wsums, 0.0j)


            # whiten and take abs
            white_resid = resid_vis_I*da.sqrt(wsums)
            abs_resid_vis_I = (white_resid).__abs__()

            # mean amp  
            sum_amp = da.sum(abs_resid_vis_I)
            count = da.sum(wsums>0)
            mean_amp = sum_amp/count

            flag_I = abs_resid_vis_I >= args.sigma_cut * mean_amp

            # new flags
            updated_flag = da.broadcast_to(flag_I[:, :, None], flag.shape, chunks=flag.chunks)

            # recompute mean amp with new flags
            weights = (~updated_flag)*weights
            resid_vis = (data - model)*weights
            wsums = (weights[:, :, 0] + weights[:, :, -1])
            resid_vis_I = da.where(wsums, (resid_vis[:, :, 0] + resid_vis[:, :, -1])/wsums, 0.0j)
            white_resid = resid_vis_I*da.sqrt(wsums)
            abs_resid_vis_I = (white_resid).__abs__()
            sum_amp = da.sum(abs_resid_vis_I)
            count = da.sum(wsums>0)
            mean_amp = sum_amp/count

            # scale weights (whitened residuals should have mean amplitude of 1/sqrt(2))
            updated_weight = 2**0.5 * weights/mean_amp**2

            ds = ds.assign(**{args.flag_out_column: (("row", "chan", "corr"), updated_flag)})
            ds = ds.assign(**{args.weight_out_column: (("row", "chan", "corr"), updated_weight)})

            out_data.append(ds)
        writes.append(xds_to_table(out_data, ims, columns=[args.flag_out_column, args.weight_out_column]))
    
    with ProgressBar():
        dask.compute(writes)

    # report new mean amp
    if args.report_means:
        radec_ref = None
        mean_amps = []
        for ims in args.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                            chunks={"row":args.row_chunks, "chan":args.chan_chunks},
                            columns=('UVW', args.data_column, args.weight_out_column,
                                    args.model_column, args.flag_out_column, 'FLAG_ROW'))

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
                flag = getattr(ds, args.flag_out_column).data
                flag = da.logical_or(flag, ds.FLAG_ROW.data[:, None, None])
                weights = getattr(ds, args.weight_out_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
                
                # Stokes I vis 
                weights = (~flag)*weights
                resid_vis = (data - model)*weights
                wsums = (weights[:, :, 0] + weights[:, :, -1])
                resid_vis_I = da.where(wsums, (resid_vis[:, :, 0] + resid_vis[:, :, -1])/wsums, 0.0j)


                # whiten and take abs
                white_resid = resid_vis_I*da.sqrt(wsums)
                abs_resid_vis_I = (white_resid).__abs__()

                # mean amp  
                sum_amp = da.sum(abs_resid_vis_I)
                count = da.sum(wsums>0)
                mean_amps.append(sum_amp/count)

        mean_amps = dask.compute(mean_amps)[0]

        print(mean_amps)


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