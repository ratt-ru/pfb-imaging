from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="airi",
    info="AIRI algorithm with PFB imaging.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="dds-out",
    info="",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.dds",
    must_exist=False,
)
@stimela_output(
    dtype="Directory",
    name="mds-out",
    info="",
    implicit="{current.output-filename}_{current.product}_{current.suffix}_model.mds",
    must_exist=False,
)
def airi(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output.",
        ),
    ],
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    hess_norm: Annotated[
        float | None,
        typer.Option(
            help="Spectral norm of hessian approximation",
        ),
        {
            "stimela": {
                "abbreviation": "hess-norm",
            },
        },
    ] = None,
    hess_approx: Annotated[
        Literal["wgt", "psf", "direct"],
        typer.Option(
            help="Which Hessian approximation to use. "
            "wgt -> vis space approximation. "
            "psf -> for zero-padded image space approximation. "
            "direct -> for direct inversion.",
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
            help="How to apply positivity constraint. "
            "0 -> no positivity. "
            "1 -> normal positivity constraint. "
            "2 -> strong positivity (zero all pixels if < 0 in any band).",
        ),
    ] = 1,
    niter: Annotated[
        int,
        typer.Option(
            help="Number of iterations.",
        ),
        {
            "stimela": {
                "abbreviation": "niter",
            },
        },
    ] = 10,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Total number of threads to use. Defaults to half the total number available.",
        ),
        {
            "stimela": {
                "abbreviation": "nt",
            },
        },
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
    verbosity: Annotated[
        int,
        typer.Option(
            help="Set to larger than 1 to report timings during residual computation",
        ),
    ] = 1,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy.",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking.",
        ),
    ] = True,
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto grid using double precision. Only has an affect when using single precision.",
        ),
    ] = True,
    # pd_tol: Annotated[
    #     float,
    #     typer.Option(
    #         help="Tolreance of primal dual algorithm.",
    #     ),
    # ] = 0.0003,
    # pd_maxit: Annotated[
    #     int,
    #     typer.Option(
    #         help="Maximum iterations for primal dual algorithm.",
    #     ),
    # ] = 450,
    # pd_verbose: Annotated[
    #     int,
    #     typer.Option(
    #         help="Verbosity of primal dual algorithm. Set to > 1 for debugging, 0 for silence",
    #     ),
    # ] = 1,
    # pd_report_freq: Annotated[
    #     int,
    #     typer.Option(
    #         help="Report frequency of primal dual algorithm",
    #     ),
    # ] = 50,
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
    shelf_path: Annotated[
        str | None,
        typer.Option(
            help="Path to CSV file containing the AIRI denoiser shelf.",
        ),
        {
            "stimela": {
                "Format": "sigma,/path/to/network.onnx",
            },
        },
    ] = None,
    airi_heuristic_scale: Annotated[
        float,
        typer.Option(
            help="Scaling factor for the heuristic noise level estimation. "
            "Can be used to fine-tune denoiser selection.",
        ),
    ] = 1.0,
    airi_adapt_update: Annotated[
        bool,
        typer.Option(
            help="Adaptively update denoiser selection based on evolving peak estimates.",
        ),
    ] = True,
    airi_peak_tol: Annotated[
        float,
        typer.Option(
            help="Relative tolerance for peak variation before updating denoiser.",
        ),
    ] = 0.1,
    airi_device: Annotated[
        Literal["cpu", "cuda"],
        typer.Option(
            help="Device to run AIRI denoisers on (cpu or cuda).",
        ),
    ] = "cpu",
    airi_tile_size: Annotated[
        int | None,
        typer.Option(
            help="Tile size for processing large images. "
            "If set, the image will be split into overlapping tiles of this size (inner region) for denoising.",
        ),
    ] = None,
    airi_tile_margin: Annotated[
        int,
        typer.Option(
            help="Margin (overlap) on each side of tiles. Full tile size is tile_size + 2*margin.",
        ),
    ] = 32,
):
    """
    AIRI algorithm with PFB imaging.
    """
    # Lazy import the core implementation
    from pfb_imaging.core.airi import airi as airi_core  # noqa: E402

    # Call the core function with all parameters
    airi_core(
        output_filename,
        suffix=suffix,
        hess_norm=hess_norm,
        hess_approx=hess_approx,
        rmsfactor=rmsfactor,
        eta=eta,
        gamma=gamma,
        nbasisf=nbasisf,
        positivity=positivity,
        niter=niter,
        nthreads=nthreads,
        tol=tol,
        diverge_count=diverge_count,
        verbosity=verbosity,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        double_accum=double_accum,
        # pd_tol=pd_tol,
        # pd_maxit=pd_maxit,
        # pd_verbose=pd_verbose,
        # pd_report_freq=pd_report_freq,
        pm_tol=pm_tol,
        pm_maxit=pm_maxit,
        pm_verbose=pm_verbose,
        pm_report_freq=pm_report_freq,
        cg_tol=cg_tol,
        cg_maxit=cg_maxit,
        cg_verbose=cg_verbose,
        cg_report_freq=cg_report_freq,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
        shelf_path=shelf_path,
        airi_heuristic_scale=airi_heuristic_scale,
        airi_adapt_update=airi_adapt_update,
        airi_peak_tol=airi_peak_tol,
        airi_device=airi_device,
        airi_tile_size=airi_tile_size,
        airi_tile_margin=airi_tile_margin,
    )
