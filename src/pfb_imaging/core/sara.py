# flake8: noqa
import concurrent.futures as cf
import time
from copy import deepcopy
from functools import partial

import numpy as np
import numexpr as ne
import psutil
import xarray as xr
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool, thread_pool_size
from numba import threading_layer

from pfb_imaging import set_envs
from pfb_imaging.operators.gridder import compute_residual
from pfb_imaging.operators.hessian import HessPSF
from pfb_imaging.operators.psi import Psi
from pfb_imaging.opt.power_method import power_method
from pfb_imaging.opt.primal_dual import primal_dual_optimised as primal_dual
from pfb_imaging.prox.prox_21m import prox_21m_numba as prox_21
from pfb_imaging.utils import logging as pfb_logging
from pfb_imaging.utils.fits import dds2fits, save_fits, set_wcs
from pfb_imaging.utils.misc import l1reweight_func
from pfb_imaging.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_imaging.utils.naming import get_opts, set_output_names, xds_from_url

log = pfb_logging.get_logger("SARA")


@pfb_logging.log_inputs(log)
def sara(
    output_filename: str,
    suffix: str = "main",
    bases: str = "self,db1,db2,db3",
    nlevels: int = 2,
    l1_reweight_from: int = 5,
    hess_norm: float | None = None,
    hess_approx: str = "psf",
    rmsfactor: float = 1.0,
    eta: float = 1.0,
    gamma: float = 1.0,
    alpha: float = 2,
    nbasisf: int | None = None,
    positivity: int = 1,
    niter: int = 10,
    nthreads: int | None = None,
    tol: float = 0.0005,
    diverge_count: int = 5,
    skip_model: bool = False,
    rms_outside_model: bool = False,
    init_factor: float = 0.5,
    verbosity: int = 1,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    double_accum: bool = True,
    pd_tol: list[float] | None = None,
    pd_maxit: int = 450,
    pd_verbose: int = 1,
    pd_report_freq: int = 50,
    pm_tol: float = 0.001,
    pm_maxit: int = 100,
    pm_verbose: int = 1,
    pm_report_freq: int = 100,
    cg_tol: float = 0.001,
    cg_maxit: int = 150,
    cg_minit: int = 10,
    cg_verbose: int = 1,
    cg_report_freq: int = 10,
    backtrack: bool = False,
    log_directory: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
    fits_mfs: bool = True,
    fits_cubes: bool = True,
):
    """
    Deconvolution using SARA regularisation
    """

    output_filename, fits_output_folder, log_directory, oname = set_output_names(
        output_filename,
        product,
        fits_output_folder,
        log_directory,
    )

    ncpu = psutil.cpu_count(logical=False)
    if nthreads is None:
        nthreads = psutil.cpu_count(logical=True) // 2
        ncpu = ncpu // 2
    else:
        ncpu = np.minimum(nthreads, psutil.cpu_count(logical=False))

    resize_thread_pool(nthreads)
    set_envs(nthreads, ncpu)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"{str(log_directory)}/sara_{timestamp}.log"
    pfb_logging.log_to_file(logname)
    log.info(f"Logs will be written to {logname}")

    basename = output_filename
    fits_oname = f"{fits_output_folder}/{oname}"
    dds_name = f"{basename}_{suffix}.dds"

    time_start = time.time()

    dds, dds_list = xds_from_url(dds_name)

    if dds[0].corr.size > 1:
        log.error_and_raise(
            "Joint polarisation deconvolution not yet supported for sara algorithm", NotImplementedError
        )

    nx, ny = dds[0].x.size, dds[0].y.size
    freq_out = []
    time_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
        time_out.append(ds.time_out)
    freq_out = np.unique(np.array(freq_out))
    time_out = np.unique(np.array(time_out))
    if time_out.size > 1:
        log.error_and_raise("Only static models currently supported", NotImplementedError)

    nband = freq_out.size

    # drop_vars after access to avoid duplicates in memory
    # and avoid unintentional side effects
    if "RESIDUAL" in dds[0]:
        residual = np.stack([ds.RESIDUAL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("DIRTY") for ds in dds]
        dds = [ds.drop_vars("RESIDUAL") for ds in dds]
    else:
        residual = np.stack([ds.DIRTY.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("DIRTY") for ds in dds]
    if "MODEL" in dds[0]:
        model = np.stack([ds.MODEL.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("MODEL") for ds in dds]
    else:
        model = np.zeros((nband, nx, ny))
    if "UPDATE" in dds[0]:
        update = np.stack([ds.UPDATE.values[0] for ds in dds], axis=0)
        dds = [ds.drop_vars("UPDATE") for ds in dds]
    else:
        update = np.zeros((nband, nx, ny))
    abspsf = np.stack([np.abs(ds.PSFHAT.values[0]) for ds in dds], axis=0)
    dds = [ds.drop_vars("PSFHAT") for ds in dds]
    beam = np.stack([ds.BEAM.values[0] for ds in dds], axis=0)
    wsums = np.stack([ds.WSUM.values[0] for ds in dds], axis=0)
    fsel = wsums > 0  # keep track of empty bands

    wsum = np.sum(wsums)
    wsums /= wsum
    abspsf /= wsum
    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)
    model_mfs = np.mean(model[fsel], axis=0)

    # for intermediary results
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq, casambm=False)
    if "niters" in dds[0].attrs:
        iter0 = dds[0].niters
    else:
        iter0 = 0

    # Allow calling with pd_tol as float or list
    if pd_tol is None:
        pd_tol = [3e-4]
    try:
        ntol = len(pd_tol)
    except TypeError:
        assert isinstance(pd_tol, float)
        ntol = 1
        pd_tol = [pd_tol]
    niters = niter
    if ntol <= niters:
        pd_tolf = pd_tol[-1]
        pd_tol += [pd_tolf] * niters  # no harm in too many

    # image space hessian
    # pre-allocate arrays for doing FFT's
    real_type = "f8"
    precond = HessPSF(
        nx,
        ny,
        abspsf,
        beam=beam,
        eta=eta * wsums,
        nthreads=nthreads,
        cgtol=cg_tol,
        cgmaxit=cg_maxit,
        cgverbose=cg_verbose,
        cgrf=cg_report_freq,
        taper_width=np.minimum(int(0.1 * nx), 32),
    )

    if hess_norm is None:
        # if the grid worker had been rerun hess_norm won't be in attrs
        if "hess_norm" in dds[0].attrs:
            hess_norm = dds[0].hess_norm
            log.info(f"Using previously estimated hess_norm of {hess_norm:.3e}")
        else:
            log.info("Finding spectral norm of Hessian approximation")
            hess_norm, hessbeta = power_method(
                precond.dot,
                (nband, nx, ny),
                tol=pm_tol,
                maxit=pm_maxit,
                verbosity=pm_verbose,
                report_freq=pm_report_freq,
            )
            # inflate slightly for stability
            hess_norm *= 1.05
    else:
        hess_norm = hess_norm
        log.info(f"Using provided hess-norm of = {hess_norm:.3e}")

    log.info("Setting up dictionary")
    bases_tuple = tuple(bases.split(","))
    nbasis = len(bases_tuple)
    psi = Psi(nband, nx, ny, bases_tuple, nlevels, nthreads)
    Nxmax = psi.Nxmax
    Nymax = psi.Nymax

    log.info(f"Using {psi.nthreads_per_band} numba threads for each band")
    log.info(f"Using {thread_pool_size()} threads for gridding")

    # number of frequency basis functions
    if nbasisf is None:
        nbasisf = int(np.sum(fsel))
    else:
        nbasisf = nbasisf
    log.info(f"Using {nbasisf} frequency basis functions")

    # a value less than zero turns L1 reweighting off
    # we'll start on convergence or at the iteration
    # indicated by l1-reweight-from, whichever comes first
    l1reweight_active = False
    # we need an array to put the components in for reweighting
    outvar = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=real_type)

    dual = np.zeros((nband, nbasis, Nymax, Nxmax), dtype=real_type)
    if l1_reweight_from == 0:
        log.info("Initialising with L1 reweighted")
        if not update.any():
            log.error_and_raise("Cannot reweight before any updates have been performed", ValueError)
        psi.dot(update, outvar)
        tmp = np.sum(outvar, axis=0)
        # exclude zeros from padding DWT's
        # rms_comps = np.std(tmp[tmp!=0])
        # log.info(f'rms_comps updated to {rms_comps}')
        # per basis rms_comps
        rms_comps = np.ones((nbasis,), dtype=float)
        for i, base in enumerate(bases_tuple):
            tmpb = tmp[i]
            rms_comps[i] = np.std(tmpb[tmpb != 0])
            log.info(f"rms_comps for base {base} is {rms_comps[i]}")
        reweighter = partial(
            l1reweight_func, psiH=psi.dot, outvar=outvar, rmsfactor=rmsfactor, rms_comps=rms_comps, alpha=alpha
        )
        l1weight = reweighter(model)
        l1reweight_active = True
    else:
        l1weight = np.ones((nbasis, Nymax, Nxmax), dtype=real_type)
        reweighter = None
        l1reweight_active = False

    if rms_outside_model and model.any():
        rms_mask = model_mfs == 0
        rms = np.std(residual_mfs[rms_mask])
    else:
        rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    best_rms = rms
    best_rmax = rmax
    best_model = model.copy()
    diverge_count_curr = 0
    eps = 1.0
    write_futures = None
    log.info(f"Iter {iter0}: peak residual = {rmax:.3e}, rms = {rms:.3e}")
    mrange = range(iter0, iter0 + niter)
    for k in mrange:
        log.info("Solving for update")
        residual *= beam  # avoid copy
        update = precond.idot(residual, mode=hess_approx, x0=update if update.any() else None)
        update_mfs = np.mean(update, axis=0)
        save_fits(update_mfs, fits_oname + f"_{suffix}_update_{k + 1}.fits", hdr_mfs)

        modelp = deepcopy(model)
        xtilde = model + gamma * update
        grad21 = lambda x: -precond.dot(xtilde - x) / gamma
        if iter0 == 0:
            lam = init_factor * rmsfactor * rms
        else:
            lam = rmsfactor * rms
        log.info(f"Solving for model with lambda = {lam}")
        model, dual = primal_dual(
            model,
            dual,
            lam,
            psi.hdot,
            psi.dot,
            hess_norm,
            prox_21,
            l1weight,
            reweighter,
            grad21,
            nu=nbasis,
            positivity=positivity,
            tol=pd_tol[k - iter0],
            maxit=pd_maxit,
            verbosity=pd_verbose,
            report_freq=pd_report_freq,
            gamma=gamma,
        )

        # write component model
        log.info(f"Writing model to {basename}_{suffix}_model.mds")
        try:
            coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
                time_out,
                freq_out[fsel],
                model[None, fsel, :, :],
                wgt=wsums[None, fsel],
                nbasisf=nbasisf,
                method="Legendre",
                sigmasq=1e-6,
            )
            # save interpolated dataset
            data_vars = {
                "coefficients": (("par", "comps"), coeffs),
            }
            coords = {
                "location_x": (("x",), x_index),
                "location_y": (("y",), y_index),
                # 'shape_x':,
                "params": (("par",), params),  # already converted to list
                "times": (("t",), time_out),  # to allow rendering to original grid
                "freqs": (("f",), freq_out),
            }
            mattrs = {
                "spec": "genesis",
                "cell_rad_x": cell_rad,
                "cell_rad_y": cell_rad,
                "npix_x": nx,
                "npix_y": ny,
                "texpr": texpr,
                "fexpr": fexpr,
                "center_x": dds[0].x0,
                "center_y": dds[0].y0,
                "flip_u": dds[0].flip_u,
                "flip_v": dds[0].flip_v,
                "flip_w": dds[0].flip_w,
                "ra": dds[0].ra,
                "dec": dds[0].dec,
                "stokes": product,  # I,Q,U,V, IQ/IV, IQUV
                "parametrisation": expr,  # already converted to str
            }

            coeff_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=mattrs)
            coeff_dataset.to_zarr(f"{basename}_{suffix}_model.mds", mode="w")

            # this is to make the model consistent with the fitted polynomial coefficients
            for b in range(nband):
                model[b] = eval_coeffs_to_slice(
                    time_out[0],
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
                    dds[0].x0,
                    dds[0].y0,
                    nx,
                    ny,
                    cell_rad,
                    cell_rad,
                    dds[0].x0,
                    dds[0].y0,
                )
        except Exception as e:
            log.info(f"Exception {e} raised during model fit .")

        model_mfs = np.mean(model[fsel], axis=0)
        save_fits(model_mfs, fits_oname + f"_{suffix}_model_{k + 1}.fits", hdr_mfs)

        # make sure write futures have finished
        if write_futures is not None:
            cf.wait(write_futures)

        log.info(f"Computing residual")
        write_futures = []
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            resid, fut = compute_residual(
                ds_name,
                nx,
                ny,
                cell_rad,
                cell_rad,
                ds_name,
                model[b][None, :, :],  # add corr axis
                nthreads=nthreads,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                verbosity=verbosity,
            )
            write_futures.append(fut)
            residual[b] = resid[0]  # remove corr axis

        residual /= wsum
        residual_mfs = np.sum(residual, axis=0)
        save_fits(residual_mfs, fits_oname + f"_{suffix}_residual_{k + 1}.fits", hdr_mfs)
        rmsp = rms
        if rms_outside_model:
            rms_mask = model_mfs == 0
            rms = np.std(residual_mfs[rms_mask])
        else:
            rms = np.std(residual_mfs)
        rmaxp = rmax
        rmax = np.abs(residual_mfs).max()
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        # what to base this on?
        if rms < best_rms:
            best_rms = rms
            best_rmax = rmax
            best_model = model.copy()

        # these are not updated in compute_residual
        for ds_name, ds in zip(dds_list, dds):
            b = int(ds.bandid)
            ds["UPDATE"] = (("corr", "x", "y"), update[b][None, :, :])
            # don't write unecessarily
            for var in ds.data_vars:
                if var != "UPDATE":
                    ds = ds.drop_vars(var)

            if (model == best_model).all():
                ds["MODEL_BEST"] = (("corr", "x", "y"), best_model[b][None, :, :])

            attrs = {}
            attrs["rms"] = best_rms
            attrs["rmax"] = best_rmax
            attrs["niters"] = k + 1
            attrs["hess_norm"] = hess_norm
            ds = ds.assign_attrs(**attrs)

            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(ds.to_zarr, ds_name, mode="a")
                write_futures.append(fut)

            # ds.to_zarr(ds_name, mode='a')

        log.info(f"Iter {k + 1}: peak residual = {rmax:.3e}, rms = {rms:.3e}, eps = {eps:.3e}")

        if eps < tol:
            # do not converge prematurely
            if l1_reweight_from > 0 and not l1reweight_active:  # only happens once
                # start reweighting
                l1_reweight_from = k + 1 - iter0
                l1reweight_active = True
            else:
                log.info(f"Converged after {k + 1} iterations.")
                break

        if (k + 1 - iter0 >= l1_reweight_from) and (k + 1 - iter0 < niter):
            log.info("Computing L1 weights")
            psi.dot(update, outvar)
            tmp = np.sum(outvar, axis=0)
            # exclude zeros from padding DWT's
            # rms_comps = np.std(tmp[tmp!=0])
            rms_comps = np.ones((nbasis,), dtype=float)
            # log.info(f'rms_comps updated to {rms_comps}')
            for i, base in enumerate(bases_tuple):
                tmpb = tmp[i]
                rms_comps[i] = np.std(tmpb[tmpb != 0])
                log.info(f"rms_comps for base {base} is {rms_comps[i]}")
            reweighter = partial(
                l1reweight_func, psiH=psi.dot, outvar=outvar, rmsfactor=rmsfactor, rms_comps=rms_comps, alpha=alpha
            )
            l1weight = reweighter(model)
            l1reweight_active = True

        if (rms > rmsp) and (rmax > rmaxp):
            diverge_count_curr += 1
            if diverge_count_curr > diverge_count:
                log.info("Algorithm is diverging. Terminating.")
                break

    # make sure write futures have finished
    cf.wait(write_futures)

    # Write FITS outputs
    dds, dds_list = xds_from_url(dds_name)

    if fits_mfs or fits_cubes:
        # get the psfpars for the mfs cube
        dds_store = DaskMSStore(dds_name)
        if "://" in dds_store.url:
            protocol = dds_store.url.split("://")[0]
        else:
            protocol = "file"
        psfpars_mfs = get_opts(dds_store.url, protocol, name="psfparsn_mfs.pkl")
        log.info(f"Writing fits files to {fits_oname}_{suffix}")

        dds2fits(
            dds_list,
            "RESIDUAL",
            f"{fits_oname}_{suffix}",
            norm_wsum=True,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing RESIDUAL")
        dds2fits(
            dds_list,
            "MODEL",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing MODEL")
        dds2fits(
            dds_list,
            "UPDATE",
            f"{fits_oname}_{suffix}",
            norm_wsum=False,
            nthreads=nthreads,
            do_mfs=fits_mfs,
            do_cube=fits_cubes,
            psfpars_mfs=psfpars_mfs,
        )
        log.info("Done writing UPDATE")

    log.info(f"Numba use the {threading_layer()} threading layer")
    log.info(f"All done after {time.time() - time_start}s")
