import time
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import psutil
import sympy as sm
import xarray as xr
from africanus.model.coherency.dask import convert
from dask.distributed import get_client, wait
from dask.graph_manipulation import clone
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_to_storage_table as xds_to_table
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool
from regions import Regions
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify

from pfb_imaging import set_client
from pfb_imaging.operators.gridder import comps2vis
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import set_wcs
from pfb_imaging.utils.misc import construct_mappings
from pfb_imaging.utils.naming import set_output_names, xds_from_url

log = pfb_logging.get_logger("DEGRID")


def degrid(
    ms: list[Path],
    output_filename: str,
    scans: str | None = None,
    ddids: str | None = None,
    fields: str | None = None,
    suffix: str = "main",
    mds: Path | None = None,
    model_column: str = "MODEL_DATA",
    product: str = "I",
    freq_range: str | None = None,
    integrations_per_image: int = -1,
    channels_per_image: int | None = None,
    accumulate: bool = False,
    region_file: str | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    host_address: str | None = None,
    nworkers: int = 1,
    nthreads: int | None = None,
    log_directory: str | None = None,
):
    """
    Predict model visibilities to measurement sets.
    The default behaviour is to read the frequency mapping from the dds and
    degrid one image per band.
    If channels-per-image is provided, the model is evaluated from the mds.
    """

    output_filename, fits_output_folder, log_directory, basedir, oname = set_output_names(
        output_filename,
        product,
        "",  # no fits output for degrid worker
        log_directory,
    )

    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True)
        ncpu = psutil.cpu_count(logical=False)
        # use half by default
        nthreads //= 2
        ncpu //= 2

    msnames = []
    for ms_name in ms:
        msstore = DaskMSStore(ms_name.rstrip("/"))
        mslist = msstore.fs.glob(ms_name.rstrip("/"))
        try:
            assert len(mslist) > 0
            msnames.append(*list(map(msstore.fs.unstrip_protocol, mslist)))
        except Exception:
            log.error_and_raise(f"No MS at {ms_name}", ValueError)
    ms = msnames

    basename = output_filename

    if mds is None:
        mds_store = DaskMSStore(f"{basename}_{suffix}_model.mds")
    else:
        mds_store = DaskMSStore(mds)
        try:
            assert mds_store.exists()
        except Exception:
            log.error_and_raise(f"No mds at {mds}", ValueError)
    mds = mds_store.url

    dds_store = DaskMSStore(f"{basename}_{suffix}.dds")
    if channels_per_image is None and not mds_store.exists():
        try:
            assert dds_store.exists()
        except Exception:
            log.error_and_raise(
                f"There must be a dds at {dds_store.url}. Specify mds and channels-per-image to degrid from mds.",
                ValueError,
            )
    dds = dds_store.url

    remprod = product.upper().strip("IQUV")
    if len(remprod):
        log.error_and_raise(f"Product {remprod} not yet supported", NotImplementedError)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/degrid_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    # pfb_logging.log_options_dict(log, opts)

    # we still need the collections interface for xds_to_table
    client = set_client(nworkers, log, None, client_log_level=log_directory)

    time_start = time.time()

    resize_thread_pool(nthreads)

    client = get_client()

    dds_store = DaskMSStore(dds)
    mds_store = DaskMSStore(mds)

    if channels_per_image is None:
        if dds_store.exists():
            dds, dds_list = xds_from_url(dds_store.url)
            cpi = 0
            for ds in dds:
                cpi = np.maximum(ds.chan.size, cpi)
        else:
            log.error_and_raise("You must supply channels per image in the absence of a dds", ValueError)
    else:
        cpi = channels_per_image

    if freq_range is not None and len(freq_range):
        fmin, fmax = freq_range.strip(" ").split(":")
        if len(fmin) > 0:
            freq_min = float(fmin)
        else:
            freq_min = -np.inf
        if len(fmax) > 0:
            freq_max = float(fmax)
        else:
            freq_max = np.inf
    else:
        freq_min = -np.inf
        freq_max = np.inf

    group_by = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

    log.info("Constructing mapping")
    (
        row_mapping,
        freq_mapping,
        time_mapping,
        freqs,
        utimes,
        ms_chunks,
        gains,
        radecs,
        chan_widths,
        uv_max,
        antpos,
        poltype,
    ) = construct_mappings(ms, None, ipi=integrations_per_image, cpi=cpi, freq_min=freq_min, freq_max=freq_max)
    #    FIELD_IDs=opts.fields,
    #    DDIDs=opts.ddids,
    #    SCANs=opts.scans)

    mds = xr.open_zarr(mds)
    foo = client.scatter(mds, broadcast=True)
    wait(foo)

    # grid spec
    nx = mds.npix_x
    ny = mds.npix_y

    # model func
    params = sm.symbols(("t", "f"))
    params += sm.symbols(tuple(mds.params.values))
    symexpr = parse_expr(mds.parametrisation)
    modelf = lambdify(params, symexpr)
    texpr = parse_expr(mds.texpr)
    tfunc = lambdify(params[0], texpr)
    fexpr = parse_expr(mds.fexpr)
    ffunc = lambdify(params[1], fexpr)

    # load region file if given
    masks = []
    if region_file is not None:
        rfile = Regions.read(region_file)  # should detect format
        # get wcs for model
        wcs = set_wcs(
            np.rad2deg(mds.cell_rad_x),
            np.rad2deg(mds.cell_rad_y),
            mds.npix_x,
            mds.npix_y,
            (mds.ra, mds.dec),
            mds.freqs.values,
            header=False,
        )
        wcs = wcs.dropaxis(-1)
        wcs = wcs.dropaxis(-1)

        mask = np.zeros((nx, ny), dtype=np.float64)
        # get a mask for each region
        for region in rfile:
            pixel_region = region.to_pixel(wcs)
            # why the transpose?
            region_mask = pixel_region.to_mask().to_image((ny, nx))
            region_mask = region_mask.T
            mask += region_mask
            masks.append(region_mask)
        if (mask > 1).any():
            log.error_and_raise("Overlapping regions are not supported", ValueError)
        remainder = 1 - mask
        # place DI component first
        masks = [remainder] + masks
    else:
        masks = [np.ones((nx, ny), dtype=np.float64)]

    input_schema = sorted(product.upper())
    if poltype == "linear":
        output_schema = ["XX", "XY", "YX", "YY"]
    else:
        output_schema = ["RR", "RL", "LR", "LL"]

    writes = []
    for ms_name in ms:
        xds = xds_from_ms(ms_name, chunks=ms_chunks[ms_name], group_cols=group_by)

        for i, mask in enumerate(masks):
            out_data = []
            columns = []
            for k, ds in enumerate(xds):
                fid = ds.FIELD_ID
                ddid = ds.DATA_DESC_ID
                scanid = ds.SCAN_NUMBER
                if (fields is not None) and (fid not in fields):
                    continue
                if (ddids is not None) and (ddid not in ddids):
                    continue
                if (scans is not None) and (scanid not in scans):
                    continue
                idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

                if i == 0:
                    column_name = model_column
                else:
                    column_name = f"{model_column}{i}"
                columns.append(column_name)

                # time <-> row mapping
                utime = da.from_array(utimes[ms_name][idt], chunks=integrations_per_image)
                tidx = da.from_array(time_mapping[ms_name][idt]["start_indices"], chunks=1)
                tcnts = da.from_array(time_mapping[ms_name][idt]["counts"], chunks=1)

                ridx = da.from_array(row_mapping[ms_name][idt]["start_indices"], chunks=integrations_per_image)
                rcnts = da.from_array(row_mapping[ms_name][idt]["counts"], chunks=integrations_per_image)

                # freq <-> band mapping (entire freq axis)
                freq = da.from_array(freqs[ms_name][idt], chunks=ms_chunks[ms_name][k]["chan"])
                fcnts = np.array(ms_chunks[ms_name][k]["chan"])
                fidx = np.concatenate((np.array([0]), np.cumsum(fcnts)))[0:-1]

                fidx = da.from_array(fidx, chunks=1)
                fcnts = da.from_array(fcnts, chunks=1)

                # number of chunks need to match in mapping and coord
                ntime_out = len(tidx.chunks[0])
                assert len(utime.chunks[0]) == ntime_out
                nfreq_out = len(fidx.chunks[0])
                assert len(freq.chunks[0]) == nfreq_out
                # and they need to match the number of row chunks
                uvw = clone(ds.UVW.data)
                assert len(uvw.chunks[0]) == len(tidx.chunks[0])

                ncorr = ds.corr.size

                vis = comps2vis(
                    uvw,
                    utime,
                    freq,
                    ridx,
                    rcnts,
                    tidx,
                    tcnts,
                    fidx,
                    fcnts,
                    mask,
                    mds,
                    modelf,
                    tfunc,
                    ffunc,
                    nthreads=nthreads,
                    epsilon=epsilon,
                    do_wgridding=do_wgridding,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    product=product,
                )

                # convert to single precision to write to MS
                vis = vis.astype(np.complex64)

                if ncorr == 1:
                    out_schema = output_schema[0]
                elif ncorr == 2:
                    out_schema = [output_schema[0], output_schema[-1]]
                else:
                    out_schema = output_schema

                vis = convert(vis, input_schema, out_schema, implicit_stokes=True)

                if accumulate:
                    vis += getattr(ds, column_name).data

                out_ds = ds.assign(**{column_name: (("row", "chan", "corr"), vis)})
                out_data.append(out_ds)

            writes.append(xds_to_table(out_data, ms, columns=columns, rechunk=True))

    # optimize_graph can make things much worse
    log.info("Computing model visibilities")
    dask.compute(writes)  # , optimize_graph=False)

    log.info(f"All done after {time.time() - time_start}s.")

    try:
        client.close()
    except Exception:
        pass
