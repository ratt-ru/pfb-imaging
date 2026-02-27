from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import ListInt, parse_list_int, stimela_cab

URI = NewType("URI", Path)


@stimela_cab(
    name="degrid",
    info="Degrid visibilities from model image(s) into measurement set. "
    "The model image needs to be in component format.",
)
def degrid(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set.",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    scans: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
        ),
    ] = None,
    ddids: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
        ),
    ] = None,
    fields: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of FIELD_ID's to image. "
            "Defaults to all. "
            "Input as comma separated string '0,2' if running from CLI.",
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    mds: Annotated[
        str | None,
        typer.Option(
            help="Optional path to mds to use for degridding. By default mds is inferred from output-filename.",
        ),
    ] = None,
    model_column: Annotated[
        str,
        typer.Option(
            help="Column to write model data to",
        ),
    ] = "MODEL_DATA",
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
        ),
    ] = "I",
    freq_range: Annotated[
        str | None,
        typer.Option(
            help="Frequency range to image in Hz. Specify as a string with colon delimiter ('1e9:1.1e9').",
        ),
    ] = None,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations corresponding to each image. "
            "Default -1 (equivalently 0 or None) implies degrid per scan.",
        ),
    ] = -1,
    channels_per_image: Annotated[
        int | None,
        typer.Option(
            help="Number of channels per image. Default (None) -> read mapping from dds. (-1, 0) -> one band per SPW.",
        ),
    ] = None,
    accumulate: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto model column",
        ),
    ] = False,
    region_file: Annotated[
        str | None,
        typer.Option(
            help="A region file containing regions that need to be converted to separate measurement set columns. "
            "Each region in the file will end up in a separate column labelled as model-column{#}. "
            "The remainder of the fields goes into model-column.",
        ),
    ] = None,
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
            help="Number of threads used to scale vertically (for FFTs and gridding). "
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
):
    """
    Degrid visibilities from model image(s) into measurement set.
    The model image needs to be in component format.
    """
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
