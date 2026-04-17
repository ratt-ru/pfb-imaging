from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import ListInt, parse_list_int, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="restore",
    info="Restore model images and.or convolved images to common resolution.",
)
@stimela_output(
    dtype="File",
    name="mfs-image",
    info="",
    implicit="{current.output-filename}_{current.product}_{current.suffix}_image_mfs.fits",
)
@stimela_output(
    dtype="File",
    name="image",
    info="",
    implicit="{current.output-filename}_{current.product}_{current.suffix}_image.fits",
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
def restore(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
            rich_help_panel="Naming",
        ),
    ],
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of model in dds",
            rich_help_panel="Input",
        ),
    ] = "MODEL",
    residual_name: Annotated[
        str,
        typer.Option(
            help="Name of residual in dds",
            rich_help_panel="Input",
        ),
    ] = "RESIDUAL",
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products. "
            "This is useful for distinguishing runs with different imaging paramaters. "
            "For example, different image sizes of robustness factors.",
            rich_help_panel="Naming",
        ),
    ] = "main",
    outputs: Annotated[
        str,
        typer.Option(
            help="Output products (m)odel, (r)esidual, (i)mage, (c)lean beam, (d)irty, (f)ft_residuals. "
            "Use capitals to produce corresponding cubes.",
            rich_help_panel="Output",
        ),
    ] = "iI",
    gausspar: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="Gaussian parameters (e-major, e-minor, position-angle) specifying restoring resolution. "
            "All parameters need to be specified in units of degrees. "
            "The default resolution is the native resolution in each imaging band. "
            "This parameter can be used to homogenise the resolution of the cubes. "
            "Set to (0,0,0) to use the resolution of the lowest band.",
            rich_help_panel="Restoration",
        ),
    ] = None,
    drop_bands: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of bands to discard.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
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
    Restore model images and.or convolved images to common resolution.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.restore import restore as restore_core  # noqa: E402

            # Call the core function with all parameters
            restore_core(
                output_filename,
                model_name=model_name,
                residual_name=residual_name,
                suffix=suffix,
                outputs=outputs,
                gausspar=gausspar,
                drop_bands=drop_bands,
                nworkers=nworkers,
                nthreads=nthreads,
                product=product,
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
        restore,
        dict(
            output_filename=output_filename,
            model_name=model_name,
            residual_name=residual_name,
            suffix=suffix,
            outputs=outputs,
            gausspar=gausspar,
            drop_bands=drop_bands,
            nworkers=nworkers,
            nthreads=nthreads,
            product=product,
            log_directory=log_directory,
            fits_output_folder=fits_output_folder,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
