from pathlib import Path
from typing import Annotated, NewType
from typing import Literal

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="init",
    info="",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="xds-out",
    info="",
)
def init(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set",
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
        str | None,
        typer.Option(
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI. "
            "Stimela dtype: List[int]",
        ),
    ] = None,
    ddids: Annotated[
        str | None,
        typer.Option(
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI. "
            "Stimela dtype: List[int]",
        ),
    ] = None,
    fields: Annotated[
        str | None,
        typer.Option(
            help="List of FIELD_ID's to image. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI. "
            "Stimela dtype: List[int]",
        ),
    ] = None,
    freq_range: Annotated[
        str | None,
        typer.Option(
            help="Frequency range to image in Hz. Specify as a string with colon delimiter eg. '1e9:1.1e9'",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of output xds",
        ),
    ] = False,
    radec: Annotated[
        str | None,
        typer.Option(
            help="Rephase all images to this radec specified in radians",
        ),
    ] = None,
    data_column: Annotated[
        str,
        typer.Option(
            help="Data column to image. "
            "Must be the same across MSs. "
            "Simple arithmetic is supported eg. "
            "'CORRECTED_DATA-MODEL_DATA'. "
            "When gains are present this column will be corrected.",
        ),
    ] = "DATA",
    weight_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing natural weights. Must be the same across MSs",
        ),
    ] = None,
    sigma_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing standard devations. "
            "Will be used to initialise natural weights if detected. "
            "Must be the same across MSs",
        ),
    ] = None,
    flag_column: Annotated[
        str,
        typer.Option(
            help="Column containing data flags. Must be the same across MSs",
        ),
    ] = "FLAG",
    gain_table: Annotated[
        list[URI] | None,
        typer.Option(
            parser=Path,
            help="Path to Quartical gain table containing NET gains. "
            "There must be a table for each MS and glob(ms) and glob(gt) should match up when running from CLI.",
        ),
    ] = None,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations per image. Default (-1, 0, None) -> dataset per scan.",
        ),
    ] = -1,
    channels_per_image: Annotated[
        int,
        typer.Option(
            help="Number of channels per image for degridding resolution. "
            "Any of (-1, 0, None) implies single dataset per spw.",
        ),
    ] = -1,
    precision: Annotated[
        Literal["single", "double"],
        typer.Option(
            help="Gridding precision",
        ),
    ] = "double",
    bda_decorr: Annotated[
        float,
        typer.Option(
            help="BDA decorrelation factor. Only has an effect if less than one",
        ),
    ] = 1.0,
    max_field_of_view: Annotated[
        float,
        typer.Option(
            help="The maximum field of view that will be considered. Used to compute decorrelation due to BDA.",
        ),
    ] = 3.0,
    beam_model: Annotated[
        str | None,
        typer.Option(
            help="Beam model to use. Only katbeam currently supported.",
        ),
    ] = None,
    chan_average: Annotated[
        int,
        typer.Option(
            help="Average this number if channels together",
        ),
    ] = 1,
    target: Annotated[
        str | None,
        typer.Option(
            help="This can be predefined celestial objects known to astropy or a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)",
        ),
    ] = None,
    progressbar: Annotated[
        bool,
        typer.Option(
            help="Display progress. Use --no-progressbar to deactivate.",
        ),
    ] = True,
    check_ants: Annotated[
        bool,
        typer.Option(
            help="Check that ANTENNA1 and ANTENNA2 tables are consistent with the ANTENNA table.",
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
):
    # Lazy import the core implementation
    from pfb_imaging.core import initinit as initinit_core  # noqa: E402

    # Parse scans if provided as comma-separated string
    scans_list = None
    if scans is not None:
        scans_list = [int(x.strip()) for x in scans.split(",")]

    # Parse ddids if provided as comma-separated string
    ddids_list = None
    if ddids is not None:
        ddids_list = [int(x.strip()) for x in ddids.split(",")]

    # Parse fields if provided as comma-separated string
    fields_list = None
    if fields is not None:
        fields_list = [int(x.strip()) for x in fields.split(",")]

    # Call the core function with all parameters
    initinit_core(
        ms,
        output_filename,
        scans=scans_list,
        ddids=ddids_list,
        fields=fields_list,
        freq_range=freq_range,
        overwrite=overwrite,
        radec=radec,
        data_column=data_column,
        weight_column=weight_column,
        sigma_column=sigma_column,
        flag_column=flag_column,
        gain_table=gain_table,
        integrations_per_image=integrations_per_image,
        channels_per_image=channels_per_image,
        precision=precision,
        bda_decorr=bda_decorr,
        max_field_of_view=max_field_of_view,
        beam_model=beam_model,
        chan_average=chan_average,
        target=target,
        progressbar=progressbar,
        check_ants=check_ants,
        log_directory=log_directory,
        product=product,
        host_address=host_address,
        nworkers=nworkers,
        nthreads=nthreads,
    )
