import json
import os
import time
from glob import glob

import fsspec as fs
import numpy as np
import psutil
import xarray as xr
from astropy.io import fits
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool

from pfb_imaging import pfb_version, set_envs
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import save_fits, set_wcs
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import set_output_names, xds_from_url

log = pfb_logging.get_logger("MODEL2COMPS")


@pfb_logging.log_inputs(log)
def model2comps(
    output_filename: str,
    overwrite: bool = False,
    mds: str | None = None,
    from_fits: str | None = None,
    nbasisf: int | None = None,
    nthreads: int | None = None,
    fit_mode: str = "Legendre",
    min_val: float | None = None,
    suffix: str = "main",
    model_name: str = "MODEL",
    use_wsum: bool = True,
    sigmasq: float = 1e-10,
    model_out: str | None = None,
    out_format: str = "zarr",
    out_freqs: str | None = None,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
):
    """
    Convert model in dds to components.
    """

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )

    nthreads_total = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = nthreads_total // 2
        ncpu = ncpu // 2

    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/model2comps_{timestamp}.log"
    pfb_logging.log_to_file(logname)

    ti = time.time()
    if from_fits:
        _model2comps_fits(
            output_filename=output_filename,
            overwrite=overwrite,
            nbasisf=nbasisf,
            nthreads=nthreads,
            fit_mode=fit_mode,
            min_val=min_val,
            suffix=suffix,
            model_name=model_name,
            use_wsum=use_wsum,
            sigmasq=sigmasq,
            model_out=model_out,
            out_format=out_format,
            out_freqs=out_freqs,
            product=product,
            fits_output_folder=fits_output_folder,
            from_fits=from_fits,
        )
    else:
        _model2comps(
            output_filename=output_filename,
            overwrite=overwrite,
            mds=mds,
            nbasisf=nbasisf,
            nthreads=nthreads,
            fit_mode=fit_mode,
            min_val=min_val,
            suffix=suffix,
            model_name=model_name,
            use_wsum=use_wsum,
            sigmasq=sigmasq,
            model_out=model_out,
            out_format=out_format,
            out_freqs=out_freqs,
            product=product,
            fits_output_folder=fits_output_folder,
        )

    log.info(f"All done after {time.time() - ti}s")


def _model2comps(
    output_filename: str,
    overwrite: bool,
    mds: str | None,
    nbasisf: int | None,
    nthreads: int,
    fit_mode: str,
    min_val: float | None,
    suffix: str,
    model_name: str,
    use_wsum: bool,
    sigmasq: float,
    model_out: str | None,
    out_format: str,
    out_freqs: str | None,
    product: str,
    fits_output_folder: str | None,
):
    basename = output_filename
    dds_name = f"{basename}_{suffix}.dds"
    dds, dds_list = xds_from_url(dds_name)

    if model_out is not None:
        coeff_name = model_out
        fits_name = model_out.rstrip(".mds") + ".fits"
    else:
        coeff_name = f"{basename}_{suffix}_{model_name.lower()}.mds"
        fits_name = f"{basename}_{suffix}_{model_name.lower()}.fits"

    mdsstore = DaskMSStore(coeff_name)
    if mdsstore.exists():
        if overwrite:
            log.info(f"Overwriting {coeff_name}")
            mdsstore.rm(recursive=True)
        else:
            log.error_and_raise(f"{coeff_name} exists. Set --overwrite to overwrite it. ", ValueError)

    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    # get cube info
    nband = 0
    ntime = 0
    mfreqs = []
    mtimes = []
    for ds in dds:
        mfreqs.append(ds.freq_out)
        mtimes.append(ds.time_out)
    mfreqs = np.unique(np.array(mfreqs))
    mtimes = np.unique(np.array(mtimes))
    nband = mfreqs.size
    ntime = mtimes.size

    # stack cube
    nx = dds[0].x.size
    ny = dds[0].y.size
    x0 = dds[0].x0
    y0 = dds[0].y0

    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    wsums = np.zeros((ntime, nband), dtype=np.float64)
    for ds in dds:
        b = int(ds.bandid)
        t = int(ds.timeid)
        model[t, b] = getattr(ds, model_name).values
        wsums[t, b] += ds.WSUM.values[0]

    if not use_wsum:
        wsums[...] = 1.0
    else:
        # normalise so ridge param has more intuitive meaning
        wsums /= wsums.max()

    # render input model onto model grid
    # TODO - this is not perfectly flux conservative
    if mds is not None:
        mdsi_store = DaskMSStore(mds)
        mdsi = xr.open_zarr(mdsi_store.url)
        # we only want to load these once
        coeffsi = mdsi.coefficients.values
        locxi = mdsi.location_x.values
        locyi = mdsi.location_y.values
        paramsi = mdsi.params.values
        for t, mtime in enumerate(mtimes):
            for b, mfreq in enumerate(mfreqs):
                model[t, b] += eval_coeffs_to_slice(
                    mtime,
                    mfreq,
                    coeffsi,
                    locxi,
                    locyi,
                    mdsi.parametrisation,
                    paramsi,
                    mdsi.texpr,
                    mdsi.fexpr,
                    mdsi.npix_x,
                    mdsi.npix_y,
                    mdsi.cell_rad_x,
                    mdsi.cell_rad_y,
                    mdsi.center_x,
                    mdsi.center_y,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    x0,
                    y0,
                )

    if not np.any(model):
        log.error_and_raise("Model is empty", RuntimeError)

    if out_freqs is not None:
        flow, fhigh, step = list(map(float, out_freqs.split(":")))
        fsel = np.all(wsums > 0, axis=0)
        if flow < mfreqs.min():
            log.info(f"Linearly extrapolating to {flow:.3e}Hz")
            # linear extrapolation from first two non-null bands
            low_index = np.argmax(fsel)  # first non-null index
            high_index = low_index + 1
            while not fsel[high_index]:
                high_index += 1
            nudiff = mfreqs[high_index] - mfreqs[low_index]
            slopes = (model[:, high_index] - model[:, low_index]) / nudiff
            intercepts = model[:, high_index] - slopes * mfreqs[high_index]
            mlow = slopes * flow + intercepts
            model = np.concatenate((mlow[:, None], model), axis=1)
            mfreqs = np.concatenate((np.array((flow,)), mfreqs))
            # TODO - duplicate first non-null value?
            wsums = np.concatenate((0.5 * wsums[:, low_index][:, None], wsums), axis=1)
            nband = mfreqs.size
        if fhigh > mfreqs.max():
            log.info(f"Linearly extrapolating to {fhigh:.3e}Hz")
            # linear extrapolation from last two non-null bands
            high_index = nband - np.argmax(fsel[::-1]) - 1  # last non-null index
            low_index = high_index - 1
            while not fsel[low_index]:
                low_index -= 1
            nudiff = mfreqs[high_index] - mfreqs[low_index]
            slopes = (model[:, high_index] - model[:, low_index]) / nudiff
            intercepts = model[:, high_index] - slopes * mfreqs[high_index]
            mhigh = slopes * fhigh + intercepts
            model = np.concatenate((model, mhigh[:, None]), axis=1)
            mfreqs = np.concatenate((mfreqs, np.array((fhigh,))))
            wsums = np.concatenate((wsums, 0.5 * wsums[:, high_index][:, None]), axis=1)
            nband = mfreqs.size

    if min_val is not None:
        model = np.where(model > min_val, model, 0.0)

    if not np.any(model):
        log.error_and_raise(f"Model has no components above {min_val}", RuntimeError)

    if nbasisf is None:
        nbasisf = nband - 1
    else:
        nbasisf = nbasisf

    nbasis = nbasisf
    log.info(f"Fitting {nband} bands with {nbasis} basis functions")
    try:
        coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
            mtimes, mfreqs, model, wgt=wsums, nbasisf=nbasisf, method=fit_mode, sigmasq=sigmasq
        )
    except np.linalg.LinAlgError as e:
        log.error_and_raise("Exception raised during fit .Do you perhaps have empty sub-bands?Decreasing nbasisf", e)

    # save interpolated dataset
    data_vars = {
        "coefficients": (("par", "comps"), coeffs),
    }
    coords = {
        "location_x": (("x",), x_index),
        "location_y": (("y",), y_index),
        # 'shape_x':,
        "params": (("par",), params),  # already converted to list
        "times": (("t",), mtimes),  # to allow rendering to original grid
        "freqs": (("f",), mfreqs),
    }
    attrs = {
        "pfb-imaging-version": pfb_version,
        "spec": "genesis",
        "cell_rad_x": cell_rad,
        "cell_rad_y": cell_rad,
        "npix_x": nx,
        "npix_y": ny,
        "texpr": texpr,
        "fexpr": fexpr,
        "center_x": x0,
        "center_y": y0,
        "flip_u": dds[0].flip_u,
        "flip_v": dds[0].flip_v,
        "flip_w": dds[0].flip_w,
        "ra": dds[0].ra,
        "dec": dds[0].dec,
        "stokes": product,  # I,Q,U,V, IQ/IV, IQUV
        "parametrisation": expr,  # already converted to str
    }

    coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    log.info(f"Writing interpolated model to {coeff_name}")

    if out_format == "zarr":
        coeff_dataset.to_zarr(mdsstore.url)
    elif out_format == "json":
        coeff_dict = coeff_dataset.to_dict()
        with fs.open(mdsstore.url, "w") as f:
            json.dump(coeff_dict, f)

    # interpolation error
    modelo = np.zeros((nband, nx, ny))
    for b in range(nband):
        modelo[b] = eval_coeffs_to_slice(
            mtimes[0],
            mfreqs[b],
            coeffs,
            x_index,
            y_index,
            expr,
            params,
            texpr,
            fexpr,
            nx,
            ny,
            cell_rad,
            cell_rad,
            x0,
            y0,
            nx,
            ny,
            cell_rad,
            cell_rad,
            x0,
            y0,
        )

    # can't import this at the top of the file due to
    # https://github.com/ratt-ru/pfb-imaging/issues/145
    from pfb_imaging.utils.misc import norm_diff

    eps = norm_diff(modelo, model[0])
    log.info(f"Fractional interpolation error is {eps:.3e}")

    # TODO - doesn't work with multiple fields
    # render model to cube
    if out_freqs is not None:
        flow, fhigh, step = list(map(float, out_freqs.split(":")))
        nbando = int((fhigh - flow) / step)
        log.info(f"Rendering model cube to {nbando} output bands")
        freq_out = np.linspace(flow, fhigh, nbando)
        ra = dds[0].ra
        dec = dds[0].dec
        hdr = set_wcs(
            cell_deg,
            cell_deg,
            nx,
            ny,
            [ra, dec],
            freq_out,
            GuassPar=(1, 1, 0),  # fake for now
            ms_time=mtimes[0],
        )
        modelo = np.zeros((nbando, nx, ny))
        for b in range(nbando):
            modelo[b] = eval_coeffs_to_slice(
                mtimes[0],
                freq_out[b],
                coeffs,
                x_index,
                y_index,
                expr,
                params,
                texpr,
                fexpr,
                nx,
                ny,
                cell_rad,
                cell_rad,
                x0,
                y0,
                nx,
                ny,
                cell_rad,
                cell_rad,
                x0,
                y0,
            )

        save_fits(modelo, fits_name, hdr, overwrite=True)


def _model2comps_fits(
    output_filename: str,
    overwrite: bool,
    nbasisf: int | None,
    nthreads: int,
    fit_mode: str,
    min_val: float | None,
    suffix: str,
    model_name: str,
    use_wsum: bool,
    sigmasq: float,
    model_out: str | None,
    out_format: str,
    out_freqs: str | None,
    product: str,
    fits_output_folder: str | None,
    from_fits: str,
):
    basename = output_filename

    if model_out is not None:
        coeff_name = model_out
        fits_name = model_out.rstrip(".mds") + ".fits"
    else:
        coeff_name = f"{basename}_{suffix}_{model_name.lower()}.mds"
        fits_name = f"{basename}_{suffix}_{model_name.lower()}.fits"

    mdsstore = DaskMSStore(coeff_name)
    if mdsstore.exists():
        if overwrite:
            log.info(f"Overwriting {coeff_name}")
            mdsstore.rm(recursive=True)
        else:
            log.error_and_raise(f"{coeff_name} exists. Set --overwrite to overwrite it. ", RuntimeError)

    images_list = sorted(glob(f"{from_fits}-[0-9][0-9][0-9][0-9]-model.fits"), key=os.path.getctime)
    if len(images_list) == 0:
        log.error_and_raise(f"No images found at {from_fits}", ValueError)

    # get cube info
    nband = len(images_list)
    ntime = 1
    mfreqs = []
    model_list = []
    cellx_deg = None
    celly_deg = None
    nx = None
    ny = None
    ra = None
    dec = None
    wsums = []
    for image in images_list:
        log.info(f"Loading {image}")
        hdu = fits.open(image)
        model_list.append(hdu[0].data.squeeze().T)
        mfreqs.append(hdu[0].header["CRVAL3"])
        if "WSCVWSUM" in hdu[0].header:
            wsums.append(hdu[0].header["WSCVWSUM"])
        elif "WSCIMGWG" in hdu[0].header:
            wsums.append(hdu[0].header["WSCIMGWG"])
        else:
            wsums.append(1.0)
        if cellx_deg is None:
            cellx_deg = np.abs(hdu[0].header["CDELT2"])
        else:
            assert cellx_deg == np.abs(hdu[0].header["CDELT2"])
        if celly_deg is None:
            celly_deg = np.abs(hdu[0].header["CDELT1"])
        else:
            assert celly_deg == np.abs(hdu[0].header["CDELT1"])
        if ny is None:
            ny = hdu[0].header["NAXIS2"]
        else:
            assert ny == hdu[0].header["NAXIS2"]
        if nx is None:
            nx = hdu[0].header["NAXIS1"]
        else:
            assert nx == hdu[0].header["NAXIS1"]
        if ra is None:
            ra = hdu[0].header["CRVAL1"]
        else:
            assert ra == hdu[0].header["CRVAL1"]
        if dec is None:
            dec = hdu[0].header["CRVAL2"]
        else:
            assert dec == hdu[0].header["CRVAL2"]
        hdu.close()
    mfreqs = np.unique(np.array(mfreqs))
    assert mfreqs.size == nband
    for b in range(nband):
        log.info(f"Found {mfreqs[b]}Hz model at index {b}")
    assert cellx_deg == celly_deg
    cell_deg = cellx_deg
    cell_rad = np.deg2rad(cell_deg)
    x0 = 0.0
    y0 = 0.0
    wsums = np.array(wsums, dtype=np.float64)[None, :]
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    mtimes = np.ones((1,), dtype=np.float64)
    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    for b in range(nband):
        model[0, b] = model_list[b]

    if not use_wsum:
        wsums[...] = 1.0
    else:
        # normalise so ridge param has more intuitive meaning
        wsums /= wsums.max()

    if not np.any(model):
        log.error_and_raise("Model is empty", ValueError)

    if out_freqs is not None:
        flow, fhigh, step = list(map(float, out_freqs.split(":")))
        fsel = np.all(wsums > 0, axis=0)
        if flow < mfreqs.min():
            log.info(f"Linearly extrapolating to {flow:.3e}Hz")
            # linear extrapolation from first two non-null bands
            low_index = np.argmax(fsel)  # first non-null index
            high_index = low_index + 1
            while not fsel[high_index]:
                high_index += 1
            nudiff = mfreqs[high_index] - mfreqs[low_index]
            slopes = (model[:, high_index] - model[:, low_index]) / nudiff
            intercepts = model[:, high_index] - slopes * mfreqs[high_index]
            mlow = slopes * flow + intercepts
            model = np.concatenate((mlow[:, None], model), axis=1)
            mfreqs = np.concatenate((np.array((flow,)), mfreqs))
            # TODO - duplicate first non-null value?
            wsums = np.concatenate((0.5 * wsums[:, low_index][:, None], wsums), axis=1)
            nband = mfreqs.size
        if fhigh > mfreqs.max():
            log.info(f"Linearly extrapolating to {fhigh:.3e}Hz")
            # linear extrapolation from last two non-null bands
            high_index = nband - np.argmax(fsel[::-1]) - 1  # last non-null index
            low_index = high_index - 1
            while not fsel[low_index]:
                low_index -= 1
            nudiff = mfreqs[high_index] - mfreqs[low_index]
            slopes = (model[:, high_index] - model[:, low_index]) / nudiff
            intercepts = model[:, high_index] - slopes * mfreqs[high_index]
            mhigh = slopes * fhigh + intercepts
            model = np.concatenate((model, mhigh[:, None]), axis=1)
            mfreqs = np.concatenate((mfreqs, np.array((fhigh,))))
            wsums = np.concatenate((wsums, 0.5 * wsums[:, high_index][:, None]), axis=1)
            nband = mfreqs.size

    if min_val is not None:
        model = np.where(model > min_val, model, 0.0)

    if not np.any(model):
        log.error_and_raise(f"Model has no components above {min_val}", ValueError)

    if nbasisf is None:
        nbasisf = nband - 1
    else:
        nbasisf = nbasisf

    nbasis = nbasisf
    log.info(f"Fitting {nband} bands with {nbasis} basis functions")
    try:
        coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
            mtimes, mfreqs, model, wgt=wsums, nbasisf=nbasisf, method=fit_mode, sigmasq=sigmasq
        )
    except np.linalg.LinAlgError as e:
        log.error_and_raise("Exception raised during fit .Do you perhaps have empty sub-bands?Decreasing nbasisf", e)

    # save interpolated dataset
    data_vars = {
        "coefficients": (("par", "comps"), coeffs),
    }
    coords = {
        "location_x": (("x",), x_index),
        "location_y": (("y",), y_index),
        # 'shape_x':,
        "params": (("par",), params),  # already converted to list
        "times": (("t",), mtimes),  # to allow rendering to original grid
        "freqs": (("f",), mfreqs),
    }
    attrs = {
        "pfb-imaging-version": pfb_version,
        "spec": "genesis",
        "cell_rad_x": cell_rad,
        "cell_rad_y": cell_rad,
        "npix_x": nx,
        "npix_y": ny,
        "texpr": texpr,
        "fexpr": fexpr,
        "center_x": x0,
        "center_y": y0,
        "flip_u": False,
        "flip_v": True,
        "flip_w": False,
        "ra": ra,
        "dec": dec,
        "stokes": product,  # I,Q,U,V, IQ/IV, IQUV
        "parametrisation": expr,  # already converted to str
    }

    coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    log.info(f"Writing interpolated model to {coeff_name}")

    if out_format == "zarr":
        coeff_dataset.to_zarr(mdsstore.url)
    elif out_format == "json":
        coeff_dict = coeff_dataset.to_dict()
        with fs.open(mdsstore.url, "w") as f:
            json.dump(coeff_dict, f)

    # interpolation error
    modelo = np.zeros((nband, nx, ny))
    for b in range(nband):
        modelo[b] = eval_coeffs_to_slice(
            mtimes[0],
            mfreqs[b],
            coeffs,
            x_index,
            y_index,
            expr,
            params,
            texpr,
            fexpr,
            nx,
            ny,
            cell_rad,
            cell_rad,
            x0,
            y0,
            nx,
            ny,
            cell_rad,
            cell_rad,
            x0,
            y0,
        )

    # can't import this at the top of the file due to
    # https://github.com/ratt-ru/pfb-imaging/issues/145
    from pfb_imaging.utils.misc import norm_diff

    eps = norm_diff(modelo, model[0])
    log.info(f"Fractional interpolation error is {eps:.3e}")

    # TODO - doesn't work with multiple fields
    # render model to cube
    if out_freqs is not None:
        flow, fhigh, step = list(map(float, out_freqs.split(":")))
        nbando = int((fhigh - flow) / step)
        log.info(f"Rendering model cube to {nbando} output bands")
        freq_out = np.linspace(flow, fhigh, nbando)
        hdr = set_wcs(
            cell_deg,
            cell_deg,
            nx,
            ny,
            [ra, dec],
            freq_out,
            GuassPar=(1, 1, 0),  # fake for now
            ms_time=mtimes[0],
        )
        modelo = np.zeros((nbando, nx, ny))
        for b in range(nbando):
            modelo[b] = eval_coeffs_to_slice(
                mtimes[0],
                freq_out[b],
                coeffs,
                x_index,
                y_index,
                expr,
                params,
                texpr,
                fexpr,
                nx,
                ny,
                cell_rad,
                cell_rad,
                x0,
                y0,
                nx,
                ny,
                cell_rad,
                cell_rad,
                x0,
                y0,
            )
    else:  # we just write it back at input frequencies as a sanity check
        hdr = set_wcs(cell_deg, cell_deg, nx, ny, [ra, dec], mfreqs, GuassPar=(1, 1, 0), casambm=False)

    save_fits(modelo[:, None, :, :], fits_name, hdr, overwrite=True)
