from pathlib import Path
from typing import Annotated, NewType

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="kclean",
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
def kclean(
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
            help="Can be used to specify a custom name for the image space data products.",
        ),
    ] = "main",
    mask: Annotated[
        str | None,
        typer.Option(
            help="Path to mask.fits",
        ),
    ] = None,
    dirosion: Annotated[
        int,
        typer.Option(
            help="Perform dilation followed by erosion with structure element of this width",
        ),
    ] = 1,
    mop_flux: Annotated[
        bool,
        typer.Option(
            help="Trigger PCG based flux mop if minor cycle stalls, the final threshold is reached or on the final iteration.",
        ),
    ] = True,
    mop_gamma: Annotated[
        float,
        typer.Option(
            help="Step size for flux mop. Should be between (0,1). A value of 1 is most aggressive.",
        ),
    ] = 0.65,
    niter: Annotated[
        int,
        typer.Option(
            help="Number of major iterations",
        ),
    ] = 5,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Total number of threads to use. Defaults to half the total number available.",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Absolute threshold at which to stop cleaning. "
            "By default it is set automatically using rmsfactor parameter",
        ),
    ] = None,
    rmsfactor: Annotated[
        float,
        typer.Option(
            help="Multiple of the rms at which to stop cleaning",
        ),
    ] = 3.0,
    eta: Annotated[
        float,
        typer.Option(
            help="Will use eta*wsum to regularise the inversion of the Hessian approximation.",
        ),
    ] = 0.001,
    gamma: Annotated[
        float,
        typer.Option(
            help="Minor loop gain",
        ),
    ] = 0.1,
    peak_factor: Annotated[
        float,
        typer.Option(
            help="Peak factor",
        ),
    ] = 0.15,
    sub_peak_factor: Annotated[
        float,
        typer.Option(
            help="Peak factor of sub-minor loop",
        ),
    ] = 0.75,
    minor_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum number of PSF convolutions between major cycles",
        ),
    ] = 50,
    subminor_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations for the sub-minor cycle",
        ),
    ] = 1000,
    verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity level. Set to 2 for maximum verbosity, 0 for silence",
        ),
    ] = 1,
    report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency for minor cycles",
        ),
    ] = 10,
    cg_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of conjugate gradient algorithm",
        ),
    ] = 0.01,
    cg_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for conjugate gradient algorithm",
        ),
    ] = 100,
    cg_minit: Annotated[
        int,
        typer.Option(
            help="Minimum iterations for conjugate gradient algorithm",
        ),
    ] = 1,
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
    ] = 100,
    backtrack: Annotated[
        bool,
        typer.Option(
            help="Ensure residual decreases at every iteration",
        ),
    ] = False,
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
    from pfb_imaging.core.kclean import kclean as kclean_core  # noqa: E402

    # Call the core function with all parameters
    kclean_core(
        output_filename,
        suffix=suffix,
        mask=mask,
        dirosion=dirosion,
        mop_flux=mop_flux,
        mop_gamma=mop_gamma,
        niter=niter,
        nthreads=nthreads,
        threshold=threshold,
        rmsfactor=rmsfactor,
        eta=eta,
        gamma=gamma,
        peak_factor=peak_factor,
        sub_peak_factor=sub_peak_factor,
        minor_maxit=minor_maxit,
        subminor_maxit=subminor_maxit,
        verbose=verbose,
        report_freq=report_freq,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        cg_minit=cg_minit,
        cg_verbose=cg_verbose,
        cg_report_freq=cg_report_freq,
        backtrack=backtrack,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )
