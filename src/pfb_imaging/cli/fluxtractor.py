from pathlib import Path
from typing import Annotated, NewType

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="fluxtractor",
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
def fluxtractor(
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
            help="Can be used to specify a custom name for the image space data product",
        ),
    ] = "main",
    mask: Annotated[
        str | None,
        typer.Option(
            help="Either path to mask.fits or set to model to construct from model",
        ),
    ] = None,
    zero_model_outside_mask: Annotated[
        bool,
        typer.Option(
            help="Make sure the input model is zero outside the mask. "
            "Only has an effect if an external mask has been passed in. "
            "A major cycle will be triggered to recompute the residual after zeroing.",
        ),
    ] = False,
    or_mask_with_model: Annotated[
        bool,
        typer.Option(
            help="Make a new mask consisting of the union of input mask and where the MFS model is larger than zero.",
        ),
    ] = False,
    min_model: Annotated[
        float,
        typer.Option(
            help="If using mask to construct model construct it where model > min-model",
        ),
    ] = 1e-05,
    eta: Annotated[
        float,
        typer.Option(
            help="Standard deviation of assumed GRF prior",
        ),
    ] = 1e-05,
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of the model to update",
        ),
    ] = "MODEL",
    residual_name: Annotated[
        str,
        typer.Option(
            help="Name of the residual to use",
        ),
    ] = "RESIDUAL",
    gamma: Annotated[
        float,
        typer.Option(
            help="Step size of update",
        ),
    ] = 1.0,
    use_psf: Annotated[
        bool,
        typer.Option(
            help="Whether to approximate the Hessian as a convolution by the PSF",
        ),
    ] = True,
    memory_greedy: Annotated[
        bool,
        typer.Option(
            help="Holds data in memory if set",
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
    host_address: Annotated[
        str | None,
        typer.Option(
            help="Address where the distributed client lives. Uses LocalCluster if no address is provided.",
        ),
    ] = None,
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of worker processes. Use with distributed scheduler.",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads used to scale vertically (eg. "
            "for FFTs and gridding). "
            "Each dask thread can in principle spawn this many threads. "
            "Will attempt to use half the available threads by default.",
        ),
    ] = None,
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
    from pfb_imaging.core.fluxtractor import fluxtractor as fluxtractor_core  # noqa: E402

    # Call the core function with all parameters
    fluxtractor_core(
        output_filename,
        suffix=suffix,
        mask=mask,
        zero_model_outside_mask=zero_model_outside_mask,
        or_mask_with_model=or_mask_with_model,
        min_model=min_model,
        eta=eta,
        model_name=model_name,
        residual_name=residual_name,
        gamma=gamma,
        use_psf=use_psf,
        memory_greedy=memory_greedy,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        cg_minit=cg_minit,
        cg_verbose=cg_verbose,
        cg_report_freq=cg_report_freq,
        backtrack=backtrack,
        host_address=host_address,
        nworkers=nworkers,
        nthreads=nthreads,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )
