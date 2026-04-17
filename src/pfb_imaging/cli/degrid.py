from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import ListInt, parse_list_int, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="degrid",
    info="Degrid visibilities from model image(s) into measurement set. "
    "The model image needs to be in component format.",
)
@stimela_output(
    dtype="Directory",
    name="log-directory",
    info="Directory to write logs and performance reports to.",
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
def degrid(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set.",
            rich_help_panel="Input",
        ),
        {
            "stimela": {
                "writable": True,
            },
        },
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
            rich_help_panel="Naming",
        ),
    ],
    scans: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    ddids: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    fields: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of FIELD_ID's to image. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
            rich_help_panel="Data Selection",
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
    mds: Annotated[
        str | None,
        typer.Option(
            help="Optional path to mds to use for degridding. By default mds is inferred from output-filename.",
            rich_help_panel="Input",
        ),
    ] = None,
    model_column: Annotated[
        str,
        typer.Option(
            help="Column to write model data to",
            rich_help_panel="Output",
        ),
    ] = "MODEL_DATA",
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    freq_range: Annotated[
        str | None,
        typer.Option(
            help="Frequency range to image in Hz. Specify as a string with colon delimiter ('1e9:1.1e9').",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations corresponding to each image. "
            "Default -1 (equivalently 0 or None) implies degrid per scan.",
            rich_help_panel="Imaging",
        ),
    ] = -1,
    channels_per_image: Annotated[
        int | None,
        typer.Option(
            help="Number of channels per image. Default (None) -> read mapping from dds. (-1, 0) -> one band per SPW.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    accumulate: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto model column",
            rich_help_panel="Output",
        ),
    ] = False,
    region_file: Annotated[
        str | None,
        typer.Option(
            help="A region file containing regions that need to be converted to separate measurement set columns. "
            "Each region in the file will end up in a separate column labelled as model-column{#}. "
            "The remainder of the fields goes into model-column.",
            rich_help_panel="Input",
        ),
    ] = None,
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
    host_address: Annotated[
        str | None,
        typer.Option(
            help="Address where the distributed client lives. Uses LocalCluster if no address is provided.",
            rich_help_panel="Distribution",
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
    Degrid visibilities from model image(s) into measurement set.
    The model image needs to be in component format.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.degrid import degrid as degrid_core  # noqa: E402

            # Call the core function with all parameters
            degrid_core(
                ms,
                output_filename,
                scans=scans,
                ddids=ddids,
                fields=fields,
                suffix=suffix,
                mds=mds,
                model_column=model_column,
                product=product,
                freq_range=freq_range,
                integrations_per_image=integrations_per_image,
                channels_per_image=channels_per_image,
                accumulate=accumulate,
                region_file=region_file,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                host_address=host_address,
                nworkers=nworkers,
                nthreads=nthreads,
                log_directory=log_directory,
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
        degrid,
        dict(
            ms=ms,
            output_filename=output_filename,
            scans=scans,
            ddids=ddids,
            fields=fields,
            suffix=suffix,
            mds=mds,
            model_column=model_column,
            product=product,
            freq_range=freq_range,
            integrations_per_image=integrations_per_image,
            channels_per_image=channels_per_image,
            accumulate=accumulate,
            region_file=region_file,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            host_address=host_address,
            nworkers=nworkers,
            nthreads=nthreads,
            log_directory=log_directory,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
