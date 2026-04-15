from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="model2comps",
    info="Convert model image to components.",
)
@stimela_output(
    dtype="Directory",
    name="mds-out",
    info="Output component model.",
    implicit="=IFSET(current.model-out, current.model-out, {current.output-filename}_{current.product}_{current.suffix}_{current.model-name}.mds)",  # noqa: E501
    must_exist=False,
)
def model2comps(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
            rich_help_panel="Naming",
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of existing model",
            rich_help_panel="Control",
        ),
    ] = False,
    mds: Annotated[
        str | None,
        typer.Option(
            help="An optional input model to append to the model. "
            "Will be rendered to the same resolution as the model in the dds.",
            rich_help_panel="Input",
        ),
    ] = None,
    from_fits: Annotated[
        str | None,
        typer.Option(
            help="An optional fits input model. An mds will be created that matches the model resolution.",
            rich_help_panel="Input",
        ),
    ] = None,
    nbasisf: Annotated[
        int | None,
        typer.Option(
            help="Order of interpolating polynomial for frequency axis. One less than the number of bands by default.",
            rich_help_panel="Fitting",
        ),
    ] = None,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads",
            rich_help_panel="Performance",
        ),
    ] = None,
    fit_mode: Annotated[
        str,
        typer.Option(
            help="",
            rich_help_panel="Fitting",
        ),
    ] = "Legendre",
    min_val: Annotated[
        float | None,
        typer.Option(
            help="Only fit components above this flux level",
            rich_help_panel="Fitting",
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products. "
            "This is useful for distinguishing runs with different imaging paramaters. "
            "For example, different image sizes of robustness factors.",
            rich_help_panel="Naming",
        ),
    ] = "main",
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of model in mds",
            rich_help_panel="Input",
        ),
    ] = "MODEL",
    use_wsum: Annotated[
        bool,
        typer.Option(
            help="Use wsum as weights during fit",
            rich_help_panel="Fitting",
        ),
    ] = True,
    sigmasq: Annotated[
        float,
        typer.Option(
            help="Multiple of the identity to add to the hessian for stability",
            rich_help_panel="Fitting",
        ),
    ] = 1e-10,
    model_out: Annotated[
        str | None,
        typer.Option(
            help="Optional explicit output name. Otherwise the default naming convention is used.",
            rich_help_panel="Output",
        ),
    ] = None,
    out_format: Annotated[
        Literal["zarr", "json"],
        typer.Option(
            help="Format to dump model to.",
            rich_help_panel="Output",
        ),
    ] = "zarr",
    out_freqs: Annotated[
        str | None,
        typer.Option(
            help="A string flow:fhigh:step of frequencies in hertz where the output cube needs to be evaluated.",
            rich_help_panel="Output",
        ),
    ] = None,
    log_directory: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Directory to write logs and performance reports to.",
            rich_help_panel="Output",
        ),
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
            rich_help_panel="Naming",
        ),
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
    Convert model image to components.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.model2comps import model2comps as model2comps_core  # noqa: E402

            # Call the core function with all parameters
            model2comps_core(
                output_filename,
                overwrite=overwrite,
                mds=mds,
                from_fits=from_fits,
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
                log_directory=log_directory,
                product=product,
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
        model2comps,
        dict(
            output_filename=output_filename,
            overwrite=overwrite,
            mds=mds,
            from_fits=from_fits,
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
            log_directory=log_directory,
            product=product,
            fits_output_folder=fits_output_folder,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
