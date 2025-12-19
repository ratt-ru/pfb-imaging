from pathlib import Path
from typing import Annotated, NewType
from typing import Literal

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="sara",
    info="",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="dds-out",
    info="",
)
@stimela_output(
    dtype="Directory",
    name="mds-out",
    info="",
)
def sara(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    bases: Annotated[
        str,
        typer.Option(
            help="Wavelet bases to use. Give as comma separated str eg.",
        ),
    ] = "self,db1,db2,db3",
    nlevels: Annotated[
        int,
        typer.Option(
            help="Wavelet decomposition level",
        ),
    ] = 2,
    l1_reweight_from: Annotated[
        int,
        typer.Option(
            help="L1 reweighting will kick in either at convergence or after this many iterations. "
            "Set to a negative value to disbale L1 reweighting.",
        ),
    ] = 5,
    hess_norm: Annotated[
        float | None,
        typer.Option(
            help="Spectral norm of hessian approximation",
        ),
    ] = None,
    hess_approx: Annotated[
        Literal["wgt", "psf", "direct"],
        typer.Option(
            help="Which Hessian approximation to use. "
            "Set to wgt for vis space approximation, psf for zero-padded image space approximation and direct for direct inversion.",
        ),
    ] = "psf",
    rmsfactor: Annotated[
        float,
        typer.Option(
            help="By default will threshold by rmsfactor*rms at every iteration",
        ),
    ] = 1.0,
    eta: Annotated[
        float,
        typer.Option(
            help="Will use eta*wsum to regularise the inversion of the Hessian approximation.",
        ),
    ] = 1.0,
    gamma: Annotated[
        float,
        typer.Option(
            help="Step size of update",
        ),
    ] = 1.0,
    alpha: Annotated[
        float,
        typer.Option(
            help="Controls how aggressively the l1reweighting is applied. "
            "Larger values correspond to more agressive reweighting.",
        ),
    ] = 2,
    nbasisf: Annotated[
        int | None,
        typer.Option(
            help="Number of basis functions to use while fitting the frequency axis. "
            "Default is to use the number of non-null imaging bands i.e. "
            "interpolation.",
        ),
    ] = None,
    positivity: Annotated[
        int,
        typer.Option(
            help="How to apply positivity constraint 0 -> no positivity, 1 -> normal positivity constraint 2 -> strong positivity i.e. "
            "all pixels in a band > 0",
        ),
    ] = 1,
    niter: Annotated[
        int,
        typer.Option(
            help="Number of iterations.",
        ),
    ] = 10,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Total number of threads to use. Defaults to half the total number available.",
        ),
    ] = None,
    tol: Annotated[
        float,
        typer.Option(
            help="Tolerance at which to terminate algorithm. Will stop when norm(x-xp)/norm(x) < tol",
        ),
    ] = 0.0005,
    diverge_count: Annotated[
        int,
        typer.Option(
            help="Will terminate the algorithm if the rms increases this many times. "
            "Set to larger than niter to disable this check.",
        ),
    ] = 5,
    skip_model: Annotated[
        bool,
        typer.Option(
            help="Skip model creation i.e. use with do-res to do-res only",
        ),
    ] = False,
    rms_outside_model: Annotated[
        bool,
        typer.Option(
            help="Mask residual where model is non-zero when computing rms. "
            "This is not recommended for largely non-empty fields.",
        ),
    ] = False,
    init_factor: Annotated[
        float,
        typer.Option(
            help="Reduce the regularisation strength by this fraction at the outset.",
        ),
    ] = 0.5,
    verbosity: Annotated[
        int,
        typer.Option(
            help="Set to larger than 1 to report timings during residual computation",
        ),
    ] = 1,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking",
        ),
    ] = True,
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto grid using double precision. Only has an affect when using single precision.",
        ),
    ] = True,
    pd_tol: Annotated[
        str,
        typer.Option(
            help="Tolreance of primal dual algorithm. Stimela dtype: List[float]",
        ),
    ] = ["3e-4"],
    pd_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for primal dual algorithm",
        ),
    ] = 450,
    pd_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of primal dual algorithm. Set to > 1 for debugging, 0 for silence",
        ),
    ] = 1,
    pd_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of primal dual algorithm",
        ),
    ] = 50,
    pm_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of power method",
        ),
    ] = 0.001,
    pm_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for power method",
        ),
    ] = 100,
    pm_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of primal power method. Set to > 1 for debugging, 0 for silence",
        ),
    ] = 1,
    pm_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of power method algorithm",
        ),
    ] = 100,
    cg_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of conjugate gradient algorithm",
        ),
    ] = 0.001,
    cg_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for conjugate gradient algorithm",
        ),
    ] = 150,
    cg_minit: Annotated[
        int,
        typer.Option(
            help="Minimum iterations for conjugate gradient algorithm",
        ),
    ] = 10,
    cg_verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity of conjugate gradient algorithm. Set to > 1 for debugging, 0 for silence",
        ),
    ] = 1,
    cg_report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency of conjugate gradient algorithm",
        ),
    ] = 10,
    backtrack: Annotated[
        bool,
        typer.Option(
            help="Ensure residual decreases at every iteration",
        ),
    ] = False,
    log_directory: Annotated[
        str | None,
        typer.Option(
            help="Directory to write logs and performance reports to.",
        ),
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
        ),
    ] = "I",
    fits_output_folder: Annotated[
        str | None,
        typer.Option(
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
        ),
    ] = None,
    fits_mfs: Annotated[
        bool,
        typer.Option(
            help="Output MFS fits files",
        ),
    ] = True,
    fits_cubes: Annotated[
        bool,
        typer.Option(
            help="Output fits cubes",
        ),
    ] = True,
):
    # Lazy import the core implementation
    from pfb_imaging.workers.sara import sara as sara_core  # noqa: E402

    # Parse pd_tol if provided as comma-separated string
    pd_tol_list = None
    if pd_tol is not None:
        pd_tol_list = [float(x.strip()) for x in pd_tol.split(",")]

    # Call the core function with all parameters
    sara_core(
        output_filename,
        suffix=suffix,
        bases=bases,
        nlevels=nlevels,
        l1_reweight_from=l1_reweight_from,
        hess_norm=hess_norm,
        hess_approx=hess_approx,
        rmsfactor=rmsfactor,
        eta=eta,
        gamma=gamma,
        alpha=alpha,
        nbasisf=nbasisf,
        positivity=positivity,
        niter=niter,
        nthreads=nthreads,
        tol=tol,
        diverge_count=diverge_count,
        skip_model=skip_model,
        rms_outside_model=rms_outside_model,
        init_factor=init_factor,
        verbosity=verbosity,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
        pd_tol=pd_tol_list,
        pd_maxit=pd_maxit,
        pd_verbose=pd_verbose,
        pd_report_freq=pd_report_freq,
        pm_tol=pm_tol,
        pm_maxit=pm_maxit,
        pm_verbose=pm_verbose,
        pm_report_freq=pm_report_freq,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        cg_minit=cg_minit,
        cg_verbose=cg_verbose,
        cg_report_freq=cg_report_freq,
        backtrack=backtrack,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )
