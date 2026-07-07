from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListStr,
    StimelaMeta,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

Directory = NewType("Directory", Path)


@stimela_cab(
    name="deconv",
    info="General deconvolution of the imager DataTree via composable forward/backward algorithms.",
)
@stimela_output(
    dtype="Directory",
    name="dt-out",
    info="DataTree dataset updated in place.",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.dt",
    must_exist=False,
)
@stimela_output(
    dtype="Directory",
    name="mds-out",
    info="Output component model.",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.mds",
    must_exist=False,
)
@stimela_output(
    dtype="Directory",
    name="log-directory",
    info="Directory to write logs and performance reports to.",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
@stimela_output(
    dtype="Directory",
    name="fits-output-folder",
    info="Optional path to write fits files to. "
    "Set to output-filename if not provided. "
    "The same naming conventions apply.",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
@stimela_output(
    dtype="Directory",
    name="numba-cache-dir",
    info="Directory to use for numba caching. Currently not configurable. Exists to ensure the directory is mounted.",
    implicit="/tmp/numba",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
)
def deconv(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output.",
            rich_help_panel="Naming",
        ),
    ],
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products. "
            "This is useful for distinguishing runs with different imaging paramaters. "
            "For example, different image sizes of robustness factors.",
            rich_help_panel="Naming",
        ),
    ] = "main",
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    fits_mfs: Annotated[
        bool,
        typer.Option(
            help="Output MFS fits files",
            rich_help_panel="Fits",
        ),
    ] = True,
    fits_cubes: Annotated[
        bool,
        typer.Option(
            help="Output fits cubes",
            rich_help_panel="Fits",
        ),
    ] = True,
    minor_cycle: Annotated[
        Literal["sara", "ista"],
        typer.Option(
            help="Which minor cycle algorithm to use.",
            rich_help_panel="PFB",
        ),
    ] = "sara",
    opt_backend: Annotated[
        Literal["primal-dual", "forward-backward"],
        typer.Option(
            help="Optimization backend for the inner backward step.",
            rich_help_panel="PFB",
        ),
    ] = "primal-dual",
    bases: Annotated[
        ListStr,
        typer.Option(
            parser=parse_list_str,
            help="Wavelet bases to use. Give as comma separated str.",
            rich_help_panel="SARA",
        ),
    ] = "self,db1,db2,db3",
    nlevels: Annotated[
        int,
        typer.Option(
            help="Wavelet decomposition level.",
            rich_help_panel="SARA",
        ),
    ] = 2,
    l1_reweight_from: Annotated[
        int,
        typer.Option(
            help="L1 reweighting will kick in either at convergence or after this many iterations. "
            "Set to a negative value to disable L1 reweighting.",
            rich_help_panel="SARA",
        ),
    ] = 5,
    alpha: Annotated[
        float,
        typer.Option(
            help="Controls how aggressively the l1reweighting is applied. "
            "Larger values correspond to more agressive reweighting.",
            rich_help_panel="SARA",
        ),
    ] = 2.0,
    hess_norm: Annotated[
        float | None,
        typer.Option(
            help="Spectral norm of hessian approximation.",
            rich_help_panel="Hessian",
        ),
    ] = None,
    rmsfactor: Annotated[
        float,
        typer.Option(
            help="By default will threshold by rmsfactor*rms at every iteration.",
            rich_help_panel="PFB",
        ),
    ] = 1.0,
    eta: Annotated[
        float,
        typer.Option(
            help="Will use eta*wsum to regularise the inversion of the Hessian approximation.",
            rich_help_panel="PFB",
        ),
    ] = 1.0,
    gamma: Annotated[
        float,
        typer.Option(
            help="Step size of update.",
            rich_help_panel="PFB",
        ),
    ] = 1.0,
    nbasisf: Annotated[
        int | None,
        typer.Option(
            help="Number of basis functions to use while fitting the frequency axis. "
            "Default is to use the number of non-null imaging bands (interpolation).",
            rich_help_panel="PFB",
        ),
    ] = None,
    positivity: Annotated[
        int,
        typer.Option(
            help="How to apply positivity constraint. "
            "0 -> no positivity. "
            "1 -> normal positivity constraint. "
            "2 -> strong positivity (zero all pixels if < 0 in any band).",
            rich_help_panel="PFB",
        ),
    ] = 1,
    niter: Annotated[
        int,
        typer.Option(
            help="Number of iterations.",
            rich_help_panel="PFB",
        ),
    ] = 10,
    tol: Annotated[
        float,
        typer.Option(
            help="Tolerance at which to terminate algorithm. Will stop when norm(x-xp)/norm(x) < tol.",
            rich_help_panel="PFB",
        ),
    ] = 0.0005,
    diverge_count: Annotated[
        int,
        typer.Option(
            help="Will terminate the algorithm if the rms increases this many times. "
            "Set to larger than niter to disable this check.",
            rich_help_panel="PFB",
        ),
    ] = 5,
    rms_outside_model: Annotated[
        bool,
        typer.Option(
            help="Mask residual where model is non-zero when computing rms. "
            "This is not recommended for largely non-empty fields.",
            rich_help_panel="PFB",
        ),
    ] = False,
    init_factor: Annotated[
        float,
        typer.Option(
            help="Reduce the regularisation strength by this fraction at the outset.",
            rich_help_panel="PFB",
        ),
    ] = 0.5,
    verbosity: Annotated[
        int,
        typer.Option(
            help="Set to larger than 1 to report timings during residual computation.",
            rich_help_panel="PFB",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Total number of threads to use. Defaults to half the total number available.",
            rich_help_panel="Performance",
        ),
    ] = None,
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of Ray workers.",
            rich_help_panel="Performance",
        ),
    ] = 1,
    ray_address: Annotated[
        str,
        typer.Option(
            help="Address of the Ray cluster to connect to.",
            rich_help_panel="Performance",
        ),
    ] = "local",
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy.",
            rich_help_panel="WGridder",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking.",
            rich_help_panel="WGridder",
        ),
    ] = True,
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto grid using double precision. Only has an affect when using single precision.",
            rich_help_panel="WGridder",
        ),
    ] = True,
    pd_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of primal dual algorithm.",
            rich_help_panel="PrimalDual",
        ),
    ] = 0.0003,
    pd_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for primal dual algorithm.",
            rich_help_panel="PrimalDual",
        ),
    ] = 450,
    pd_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of primal dual algorithm. Set to > 1 for debugging, 0 for silence",
            rich_help_panel="PrimalDual",
        ),
    ] = 1,
    pd_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of primal dual algorithm",
            rich_help_panel="PrimalDual",
        ),
    ] = 50,
    fb_tol: Annotated[
        float,
        typer.Option(
            help="Tolerance of forward-backward algorithm.",
            rich_help_panel="ForwardBackward",
        ),
    ] = 0.0003,
    fb_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for forward-backward algorithm.",
            rich_help_panel="ForwardBackward",
        ),
    ] = 450,
    fb_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of forward-backward algorithm. Set to > 1 for debugging, 0 for silence.",
            rich_help_panel="ForwardBackward",
        ),
    ] = 1,
    fb_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of forward-backward algorithm.",
            rich_help_panel="ForwardBackward",
        ),
    ] = 50,
    acceleration: Annotated[
        bool,
        typer.Option(
            help="Enable FISTA acceleration for forward-backward backend.",
            rich_help_panel="ForwardBackward",
        ),
    ] = True,
    pm_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of power method",
            rich_help_panel="PowerMethod",
        ),
    ] = 0.001,
    pm_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for power method",
            rich_help_panel="PowerMethod",
        ),
    ] = 100,
    pm_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of power method. Set to > 1 for debugging, 0 for silence",
            rich_help_panel="PowerMethod",
        ),
    ] = 1,
    pm_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of power method algorithm",
            rich_help_panel="PowerMethod",
        ),
    ] = 100,
    cg_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 0.001,
    cg_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 150,
    cg_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of conjugate gradient algorithm. Set to > 1 for debugging, 0 for silence",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 1,
    cg_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 10,
    log_directory: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Directory to write logs and performance reports to.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=False,
            mkdir=False,
            path_policies={
                "write_parent": True,
            },
        ),
    ] = None,
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=False,
            mkdir=False,
            path_policies={
                "write_parent": True,
            },
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    General deconvolution of the imager DataTree via composable forward/backward algorithms.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                deconv,
                dict(
                    output_filename=output_filename,
                    suffix=suffix,
                    product=product,
                    fits_mfs=fits_mfs,
                    fits_cubes=fits_cubes,
                    minor_cycle=minor_cycle,
                    opt_backend=opt_backend,
                    bases=bases,
                    nlevels=nlevels,
                    l1_reweight_from=l1_reweight_from,
                    alpha=alpha,
                    hess_norm=hess_norm,
                    rmsfactor=rmsfactor,
                    eta=eta,
                    gamma=gamma,
                    nbasisf=nbasisf,
                    positivity=positivity,
                    niter=niter,
                    tol=tol,
                    diverge_count=diverge_count,
                    rms_outside_model=rms_outside_model,
                    init_factor=init_factor,
                    verbosity=verbosity,
                    nthreads=nthreads,
                    nworkers=nworkers,
                    ray_address=ray_address,
                    epsilon=epsilon,
                    do_wgridding=do_wgridding,
                    double_accum=double_accum,
                    pd_tol=pd_tol,
                    pd_maxit=pd_maxit,
                    pd_verbose=pd_verbose,
                    pd_report_freq=pd_report_freq,
                    fb_tol=fb_tol,
                    fb_maxit=fb_maxit,
                    fb_verbose=fb_verbose,
                    fb_report_freq=fb_report_freq,
                    acceleration=acceleration,
                    pm_tol=pm_tol,
                    pm_maxit=pm_maxit,
                    pm_verbose=pm_verbose,
                    pm_report_freq=pm_report_freq,
                    cg_tol=cg_tol,
                    cg_maxit=cg_maxit,
                    cg_verbose=cg_verbose,
                    cg_report_freq=cg_report_freq,
                    log_directory=log_directory,
                    fits_output_folder=fits_output_folder,
                ),
            )

            # Lazy import the core implementation
            from pfb_imaging.core.deconv import deconv as deconv_core  # noqa: E402

            # Call the core function with all parameters
            deconv_core(
                output_filename,
                suffix=suffix,
                product=product,
                fits_mfs=fits_mfs,
                fits_cubes=fits_cubes,
                minor_cycle=minor_cycle,
                opt_backend=opt_backend,
                bases=bases,
                nlevels=nlevels,
                l1_reweight_from=l1_reweight_from,
                alpha=alpha,
                hess_norm=hess_norm,
                rmsfactor=rmsfactor,
                eta=eta,
                gamma=gamma,
                nbasisf=nbasisf,
                positivity=positivity,
                niter=niter,
                tol=tol,
                diverge_count=diverge_count,
                rms_outside_model=rms_outside_model,
                init_factor=init_factor,
                verbosity=verbosity,
                nthreads=nthreads,
                nworkers=nworkers,
                ray_address=ray_address,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                pd_tol=pd_tol,
                pd_maxit=pd_maxit,
                pd_verbose=pd_verbose,
                pd_report_freq=pd_report_freq,
                fb_tol=fb_tol,
                fb_maxit=fb_maxit,
                fb_verbose=fb_verbose,
                fb_report_freq=fb_report_freq,
                acceleration=acceleration,
                pm_tol=pm_tol,
                pm_maxit=pm_maxit,
                pm_verbose=pm_verbose,
                pm_report_freq=pm_report_freq,
                cg_tol=cg_tol,
                cg_maxit=cg_maxit,
                cg_verbose=cg_verbose,
                cg_report_freq=cg_report_freq,
                log_directory=log_directory,
                fits_output_folder=fits_output_folder,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("pfb-imaging")
    if image is None:
        raise RuntimeError("No Container URL in pfb-imaging metadata.")

    run_in_container(
        deconv,
        dict(
            output_filename=output_filename,
            suffix=suffix,
            product=product,
            fits_mfs=fits_mfs,
            fits_cubes=fits_cubes,
            minor_cycle=minor_cycle,
            opt_backend=opt_backend,
            bases=bases,
            nlevels=nlevels,
            l1_reweight_from=l1_reweight_from,
            alpha=alpha,
            hess_norm=hess_norm,
            rmsfactor=rmsfactor,
            eta=eta,
            gamma=gamma,
            nbasisf=nbasisf,
            positivity=positivity,
            niter=niter,
            tol=tol,
            diverge_count=diverge_count,
            rms_outside_model=rms_outside_model,
            init_factor=init_factor,
            verbosity=verbosity,
            nthreads=nthreads,
            nworkers=nworkers,
            ray_address=ray_address,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            pd_tol=pd_tol,
            pd_maxit=pd_maxit,
            pd_verbose=pd_verbose,
            pd_report_freq=pd_report_freq,
            fb_tol=fb_tol,
            fb_maxit=fb_maxit,
            fb_verbose=fb_verbose,
            fb_report_freq=fb_report_freq,
            acceleration=acceleration,
            pm_tol=pm_tol,
            pm_maxit=pm_maxit,
            pm_verbose=pm_verbose,
            pm_report_freq=pm_report_freq,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
            cg_verbose=cg_verbose,
            cg_report_freq=cg_report_freq,
            log_directory=log_directory,
            fits_output_folder=fits_output_folder,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
