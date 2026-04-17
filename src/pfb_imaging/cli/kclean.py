from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="kclean",
    info="Modified single scale clean algorithm.",
)
@stimela_output(
    dtype="Directory",
    name="dds-out",
    info="Output dataset directory.",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.dds",
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
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
def kclean(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
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
    mask: Annotated[
        str | None,
        typer.Option(
            help="Path to mask.fits",
            rich_help_panel="Input",
        ),
    ] = None,
    dirosion: Annotated[
        int,
        typer.Option(
            help="Perform dilation followed by erosion with structure element of this width",
            rich_help_panel="Masking",
        ),
    ] = 1,
    mop_flux: Annotated[
        bool,
        typer.Option(
            help="Use PCG based flux mop.",
            rich_help_panel="PFB",
        ),
    ] = True,
    mop_gamma: Annotated[
        float,
        typer.Option(
            help="Step size for flux mop. Should be between (0,1). A value of 1 is most aggressive.",
            rich_help_panel="PFB",
        ),
    ] = 0.65,
    niter: Annotated[
        int,
        typer.Option(
            help="Number of major iterations",
            rich_help_panel="PFB",
        ),
    ] = 5,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Total number of threads to use. Defaults to half the total number available.",
            rich_help_panel="Performance",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Absolute threshold at which to stop cleaning. "
            "By default it is set automatically using rmsfactor parameter",
            rich_help_panel="PFB",
        ),
    ] = None,
    rmsfactor: Annotated[
        float,
        typer.Option(
            help="Multiple of the rms at which to stop cleaning",
            rich_help_panel="PFB",
        ),
    ] = 3.0,
    eta: Annotated[
        float,
        typer.Option(
            help="Will use eta*wsum to regularise the inversion of the Hessian approximation.",
            rich_help_panel="PFB",
        ),
    ] = 0.001,
    gamma: Annotated[
        float,
        typer.Option(
            help="Minor loop gain",
            rich_help_panel="PFB",
        ),
    ] = 0.1,
    peak_factor: Annotated[
        float,
        typer.Option(
            help="Peak factor",
            rich_help_panel="Clean",
        ),
    ] = 0.15,
    sub_peak_factor: Annotated[
        float,
        typer.Option(
            help="Peak factor of sub-minor loop",
            rich_help_panel="Clean",
        ),
    ] = 0.75,
    minor_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum number of PSF convolutions between major cycles",
            rich_help_panel="Clean",
        ),
    ] = 50,
    subminor_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations for the sub-minor cycle",
            rich_help_panel="Clean",
        ),
    ] = 1000,
    verbose: Annotated[
        int,
        typer.Option(
            help="Verbosity level. Set to 2 for maximum verbosity, 0 for silence",
            rich_help_panel="Reporting",
        ),
    ] = 1,
    report_freq: Annotated[
        int,
        typer.Option(
            help="Report frequency for minor cycles",
            rich_help_panel="Reporting",
        ),
    ] = 10,
    cg_tol: Annotated[
        float,
        typer.Option(
            help="Tolreance of conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 0.01,
    cg_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 100,
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
    ] = 100,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy",
            rich_help_panel="WGridder",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking",
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
    log_directory: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Directory to write logs and performance reports to.",
            rich_help_panel="Output",
        ),
        {
            "stimela": {
                "mkdir": False,
                "path_policies": {
                    "write_parent": True,
                },
            },
        },
    ] = None,
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
            rich_help_panel="Output",
        ),
        {
            "stimela": {
                "mkdir": False,
                "path_policies": {
                    "write_parent": True,
                },
            },
        },
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        {"stimela": {"skip": True}},
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        {"stimela": {"skip": True}},
    ] = False,
):
    """
    Modified single scale clean algorithm.
    """
    if backend == "native" or backend == "auto":
        try:
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
                cg_verbose=cg_verbose,
                cg_report_freq=cg_report_freq,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                product=product,
                fits_mfs=fits_mfs,
                fits_cubes=fits_cubes,
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
        kclean,
        dict(
            output_filename=output_filename,
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
            cg_verbose=cg_verbose,
            cg_report_freq=cg_report_freq,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            product=product,
            fits_mfs=fits_mfs,
            fits_cubes=fits_cubes,
            log_directory=log_directory,
            fits_output_folder=fits_output_folder,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
