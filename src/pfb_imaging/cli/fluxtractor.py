from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="fluxtractor",
    info="Otherwise knows as the fluxmop.",
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
def fluxtractor(
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
            help="Either path to mask.fits or set to model to construct from model",
            rich_help_panel="Input",
        ),
    ] = None,
    zero_model_outside_mask: Annotated[
        bool,
        typer.Option(
            help="Make sure the input model is zero outside the mask. "
            "Only has an effect if an external mask has been passed in. "
            "A major cycle will be triggered to recompute the residual after zeroing.",
            rich_help_panel="Masking",
        ),
    ] = False,
    or_mask_with_model: Annotated[
        bool,
        typer.Option(
            help="Make a new mask consisting of the union of input mask and where the MFS model is larger than zero.",
            rich_help_panel="Masking",
        ),
    ] = False,
    min_model: Annotated[
        float,
        typer.Option(
            help="If using mask to construct model construct it where model > min-model",
            rich_help_panel="Masking",
        ),
    ] = 1e-05,
    eta: Annotated[
        float,
        typer.Option(
            help="Standard deviation of assumed GRF prior",
            rich_help_panel="Imaging",
        ),
    ] = 1e-05,
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of the model to update",
            rich_help_panel="Input",
        ),
    ] = "MODEL",
    residual_name: Annotated[
        str,
        typer.Option(
            help="Name of the residual to use",
            rich_help_panel="Input",
        ),
    ] = "RESIDUAL",
    use_psf: Annotated[
        bool,
        typer.Option(
            help="Whether to approximate the Hessian as a convolution by the PSF",
            rich_help_panel="Imaging",
        ),
    ] = True,
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
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of worker processes. Use with distributed scheduler.",
            rich_help_panel="Performance",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads used to scale vertically (for FFTs and gridding). "
            "Each dask thread can in principle spawn this many threads. "
            "Will attempt to use half the available threads by default.",
            rich_help_panel="Performance",
        ),
    ] = None,
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
    Otherwise knows as the fluxmop.
    """
    if backend == "native" or backend == "auto":
        try:
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
                use_psf=use_psf,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                cg_tol=cg_tol,
                cg_maxit=cg_maxit,
                cg_verbose=cg_verbose,
                cg_report_freq=cg_report_freq,
                nworkers=nworkers,
                nthreads=nthreads,
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
        fluxtractor,
        dict(
            output_filename=output_filename,
            suffix=suffix,
            mask=mask,
            zero_model_outside_mask=zero_model_outside_mask,
            or_mask_with_model=or_mask_with_model,
            min_model=min_model,
            eta=eta,
            model_name=model_name,
            residual_name=residual_name,
            use_psf=use_psf,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
            cg_verbose=cg_verbose,
            cg_report_freq=cg_report_freq,
            nworkers=nworkers,
            nthreads=nthreads,
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
